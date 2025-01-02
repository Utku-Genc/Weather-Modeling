import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
from torch.utils.data import DataLoader, TensorDataset
import time

# CSV dosyasını yükle
df = pd.read_csv('./daily_data.csv')

# LSTNet Modeli
class LSTNet(nn.Module):
    def __init__(self, input_size, cnn_kernel, hidden_size, output_size, dropout=0.2):
        super(LSTNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=(cnn_kernel, input_size))
        self.relu = nn.ReLU()
        self.gru = nn.GRU(16, hidden_size, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = x.unsqueeze(1)  # (batch_size, 1, seq_len, input_size)
        x = self.conv1(x)
        x = self.relu(x)
        x = x.squeeze(3).permute(0, 2, 1)  # (batch_size, seq_len, num_filters)
        x, _ = self.gru(x)
        x = self.dropout(x)
        x = self.fc(x[:, -1, :])  # Son zaman adımının çıkışı
        return x

# Model parametreleri
input_size = 5  # Özellik sayısı
cnn_kernel = 6
hidden_size = 32
output_size = 5
learning_rate = 0.001
epochs = 10
patience = 7

# Modeli oluştur
model = LSTNet(input_size=input_size, cnn_kernel=cnn_kernel, hidden_size=hidden_size, output_size=output_size)

# Kayıp fonksiyonu ve optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

# Veriyi yüklemek ve ölçeklendirmek
scaler = StandardScaler()
data = df[['Temperature', 'Dew Point', 'Humidity', 'Wind Speed', 'Pressure']].values
scaled_data = scaler.fit_transform(data)

# Veriyi eğitim ve test olarak ayırma
train_size = int(len(scaled_data) * 0.8)
train_data = scaled_data[:train_size]
test_data = scaled_data[train_size:]

# Zaman serisi verisini 3D hale getirmek
def create_sequences(data, seq_length=30):
    sequences = []
    labels = []
    for i in range(len(data) - seq_length):
        seq = data[i:i+seq_length]
        label = data[i+seq_length]
        sequences.append(seq)
        labels.append(label)
    return np.array(sequences), np.array(labels)

# Eğitim ve test verisi oluşturma
X_train, y_train = create_sequences(train_data)
X_test, y_test = create_sequences(test_data)

# Tensor formatına çevirme
X_train_tensor = torch.Tensor(X_train)
y_train_tensor = torch.Tensor(y_train)
X_test_tensor = torch.Tensor(X_test)
y_test_tensor = torch.Tensor(y_test)

# DataLoader oluşturma
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
validation_dataset = TensorDataset(X_test_tensor, y_test_tensor)
validation_loader = DataLoader(validation_dataset, batch_size=16, shuffle=False)

# Eğitim döngüsü
train_losses = []
validation_losses = []
best_val_loss = float('inf')
patience_counter = 0

start_training_time = time.time()

for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for batch in train_loader:
        inputs, targets = batch
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    train_loss = running_loss / len(train_loader)
    train_losses.append(train_loss)

    model.eval()
    validation_loss = 0.0
    with torch.no_grad():
        for val_batch in validation_loader:
            val_inputs, val_targets = val_batch
            val_outputs = model(val_inputs)
            val_loss = criterion(val_outputs, val_targets)
            validation_loss += val_loss.item()

    val_loss = validation_loss / len(validation_loader)
    validation_losses.append(val_loss)

    scheduler.step()

    print(f"Epoch {epoch + 1}/{epochs} - Training Loss: {train_loss:.4f} - Validation Loss: {val_loss:.4f}")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        torch.save(model.state_dict(), 'best_lstnet_model.pth')  # En iyi modeli kaydet
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print("Early stopping triggered.")
            break

end_training_time = time.time()
training_time = end_training_time - start_training_time

# Test seti tahminlerini yapmak
model.eval()
with torch.no_grad():
    y_test_predictions = model(X_test_tensor).numpy()

y_test_actual = y_test_tensor.numpy()

# Performans metriklerini hesaplama
mse_test = mean_squared_error(y_test_actual, y_test_predictions)
mae_test = mean_absolute_error(y_test_actual, y_test_predictions)
rmse_test = np.sqrt(mse_test)
r2_test = r2_score(y_test_actual, y_test_predictions)
mape_test = mean_absolute_percentage_error(y_test_actual, y_test_predictions)

# Performans metriklerini yazdırma
print(f"=== Test Seti Performansı ===")
print(f"MSE: {mse_test:.4f}")
print(f"MAE: {mae_test:.4f}")
print(f"RMSE: {rmse_test:.4f}")
print(f"R²: {r2_test:.4f}")
print(f"MAPE: {mape_test:.4f}")

# Eğitim ve doğrulama kayıplarını çizme
plt.figure(figsize=(10, 6))
plt.plot(range(len(train_losses)), train_losses, label='Eğitim Kaybı (Train Loss)', marker='o')
plt.plot(range(len(validation_losses)), validation_losses, label='Doğrulama Kaybı (Validation Loss)', marker='x')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Eğitim ve Doğrulama Kaybı Grafiği')
plt.legend()
plt.grid()
plt.show()

# Gelecek tahminler için fonksiyon
def forecast_future(model, last_sequence, scaler, input_size, future_days):
    model.eval()
    future_predictions = []
    current_sequence = last_sequence

    for _ in range(future_days):
        with torch.no_grad():
            input_tensor = torch.Tensor(current_sequence).unsqueeze(0)  # Batch boyutunu ekle
            prediction = model(input_tensor).squeeze(0).numpy()  # Tahmin yap
            future_predictions.append(prediction)
            current_sequence = np.vstack([current_sequence[1:], prediction])

    future_predictions = scaler.inverse_transform(future_predictions)
    return future_predictions

last_sequence = X_test[-1]
future_days = 7

model.load_state_dict(torch.load('best_lstnet_model.pth'))
future_predictions = forecast_future(model, last_sequence, scaler, input_size, future_days)

# Sayısal sütunların isimleri
feature_names = ['Temperature', 'Dew Point', 'Humidity', 'Wind Speed', 'Pressure']

# Tahminleri düzenli bir şekilde yazdırma
print("7 Günlük Tahminler:")
for i, feature in enumerate(feature_names):
    print(f"\n{feature} için tahminler:")
    for day, prediction in enumerate(future_predictions[:, i], start=1):
        print(f"Gün {day}: {prediction:.2f}")

        
# Tahminleri çizme
dates = pd.date_range(start=df['Date'].iloc[-1], periods=future_days + 1, freq='D')[1:]
plt.figure(figsize=(10, 6))
for i, label in enumerate(feature_names):
    plt.plot(dates, future_predictions[:, i], label=f'{label} Tahmini')

plt.xlabel('Tarih')
plt.ylabel('Değer')
plt.title('Gelecek Tahminler')
plt.legend()
plt.grid()
plt.show()

# Orijinal veri ve tahminleri karşılaştırma
original_dates = pd.to_datetime(df['Date'])
plt.figure(figsize=(12, 8))
for i, label in enumerate(feature_names):
    plt.plot(original_dates, df[label], label=f'Orijinal {label}')
    plt.plot(dates, future_predictions[:, i], linestyle='--', label=f'{label} Tahmini')

plt.xlabel("Tarih")
plt.ylabel("Değer")
plt.title("Orijinal Veri ve 7 Günlük Tahminler")
plt.legend()
plt.grid()
plt.show()

# Son 10 günlük gerçek veri ve tahminlerin karşılaştırılması
zoomed_original_dates = original_dates.iloc[-10:]
zoomed_original_data = df.iloc[-10:][feature_names]

plt.figure(figsize=(12, 8))
for i, label in enumerate(feature_names):
    plt.plot(zoomed_original_dates, zoomed_original_data[label], label=f'Son 10 Gün Gerçek {label}')
    plt.plot(dates, future_predictions[:, i], linestyle='--', label=f'{label} Tahmini')

plt.xlabel("Tarih")
plt.ylabel("Değer")
plt.title("Son 10 Günlük Gerçek Veri ve Tahminler")
plt.legend()
plt.grid()
plt.show()

# Çıkarım zamanı ölçümü
start_inference_time = time.time()
_ = forecast_future(model, last_sequence, scaler, input_size, future_days)
end_inference_time = time.time()
inference_time = end_inference_time - start_inference_time

print(f"Training Time: {training_time:.2f} seconds")
print(f"Inference Time: {inference_time:.2f} seconds")
