#TFT MODELİ TÜM DEĞERLER
import pandas as pd
import numpy as np
import torch
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
from pytorch_forecasting.data.encoders import GroupNormalizer
from pytorch_forecasting.metrics import RMSE
from lightning.pytorch.callbacks import EarlyStopping
from lightning.pytorch.loggers import CSVLogger
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
import matplotlib.pyplot as plt
from lightning.pytorch import Trainer
import time

# 1. Veriyi Yükleme ve Hazırlama
df = pd.read_csv('./daily_data.csv')
df = df[['Date', 'Temperature', 'Dew Point', 'Humidity', 'Wind Speed', 'Pressure']].copy()
df['time_idx'] = range(len(df))
df['time_idx'] = df['time_idx'].astype('int64')
df['group_id'] = 0  # Sabit grup kimliği

# Tüm sayısal sütunlar hedef olarak ayarlanır
target = ['Temperature', 'Dew Point', 'Humidity', 'Wind Speed', 'Pressure']
time_varying_known_reals = ['time_idx']
time_varying_unknown_reals = target

# 2. Ölçeklendirme
data = df[['time_idx', 'group_id'] + target].copy()
scaler = StandardScaler()
scaled_features = scaler.fit_transform(data[target])
data[target] = scaled_features

from pytorch_forecasting.data.encoders import GroupNormalizer, MultiNormalizer

max_encoder_length = 30  # Girdi uzunluğu
max_prediction_length = 7  # Tahmin uzunluğu


# TemporalFusionTransformer Veri Seti
dataset = TimeSeriesDataSet(
    data,
    time_idx="time_idx",
    target=target,
    group_ids=["group_id"],
    min_encoder_length=max_encoder_length // 2,
    max_encoder_length=max_encoder_length,
    max_prediction_length=max_prediction_length,
    static_categoricals=[],
    time_varying_known_reals=time_varying_known_reals,
    time_varying_unknown_reals=time_varying_unknown_reals,
    target_normalizer=MultiNormalizer(
        [GroupNormalizer(groups=["group_id"]) for _ in target]
    ),
    add_relative_time_idx=True,
    add_target_scales=True,
    add_encoder_length=True,
)

train_dataloader = dataset.to_dataloader(train=True, batch_size=16, num_workers=0)
val_dataloader = dataset.to_dataloader(train=False, batch_size=16, num_workers=0)

# TemporalFusionTransformer Modeli
tft = TemporalFusionTransformer.from_dataset(
    dataset,
    learning_rate=0.03,
    hidden_size=16,
    attention_head_size=1,
    dropout=0.1,
    hidden_continuous_size=8,
    output_size=[1] * len(target),  # Her hedef değişken için ayrı bir çıktı boyutu
    loss=RMSE(reduction="mean"),
    reduce_on_plateau_patience=4,
)
print(f"Modeldeki parametre sayısı: {tft.size()}")

# 5. Eğitim Ayarları ve Trainer
early_stop_callback = EarlyStopping(monitor="val_loss", patience=3, mode="min")
logger = CSVLogger("logs", name="tft-training")

trainer = Trainer(
    max_epochs=10,
    accelerator="cpu",  # GPU kullanımını aktif etmek için "gpu" yapabilirsiniz
    gradient_clip_val=0.1,
    logger=logger,
    callbacks=[early_stop_callback],
)

# Eğitim süresini ölçme
start_time = time.time()
trainer.fit(tft, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
training_time = time.time() - start_time

# 6. Eğitim ve doğrulama sonrası metrikler
print("\n=== Eğitim Tamamlandı, Performans Metrikleri Hesaplanıyor ===")

# Doğrulama seti tahminlerini yapmak
start_inference_time = time.time()  # Inferans süresi başlangıcı
val_predictions = tft.predict(val_dataloader)
inference_time = time.time() - start_inference_time  # Inferans süresi hesaplama

# Val actuals için doğru yapıyı oluşturma
val_actuals = torch.cat(
    [torch.cat(y[0], dim=0) for x, y in iter(val_dataloader) if y[0] is not None],
    dim=0
).numpy()

# Tahminler
val_predictions = tft.predict(val_dataloader)

# Eğer `val_predictions` bir liste ise tensöre dönüştür
if isinstance(val_predictions, list):
    val_predictions = torch.cat([torch.tensor(pred) for pred in val_predictions], dim=0)

# Ölçeklendirmeyi geri alma
actuals_rescaled = scaler.inverse_transform(val_actuals.reshape(-1, len(target)))
predicted_values_rescaled = scaler.inverse_transform(val_predictions.numpy().reshape(-1, len(target)))
# Performans Metriklerini Hesaplama
from tabulate import tabulate  # Sonuçları tablo olarak göstermek için

metrics_table = []

for i, col in enumerate(target):
    # Hedef değişken için metrikleri hesaplama
    mse = mean_squared_error(actuals_rescaled[:, i], predicted_values_rescaled[:, i])
    mae = mean_absolute_error(actuals_rescaled[:, i], predicted_values_rescaled[:, i])
    rmse = np.sqrt(mse)
    r2 = r2_score(actuals_rescaled[:, i], predicted_values_rescaled[:, i])
    mape = mean_absolute_percentage_error(actuals_rescaled[:, i], predicted_values_rescaled[:, i])

    # Sonuçları tabloya ekleme
    metrics_table.append([col, mse, mae, rmse, r2, mape])

# Ortalama Metrikler
average_mse = np.mean([row[1] for row in metrics_table])
average_mae = np.mean([row[2] for row in metrics_table])
average_rmse = np.mean([row[3] for row in metrics_table])
average_r2 = np.mean([row[4] for row in metrics_table])
average_mape = np.mean([row[5] for row in metrics_table])

metrics_table.append(["Average", average_mse, average_mae, average_rmse, average_r2, average_mape])

# Tabloyu Görselleştirme
headers = ["Target", "MSE", "MAE", "RMSE", "R²", "MAPE"]
print(tabulate(metrics_table, headers=headers, floatfmt=".4f"))


print(f"Training Time: {training_time:.2f} seconds")
print(f"Inference Time: {inference_time:.2f} seconds")


# Loss Grafiği Oluşturma
log_path = logger.log_dir + "/metrics.csv"
metrics_df = pd.read_csv(log_path)

plt.figure(figsize=(10, 6))

# Train loss çizimi (step bazında)
if "train_loss_step" in metrics_df.columns:
    plt.plot(metrics_df["step"], metrics_df["train_loss_step"], label="Train Loss (Step)", alpha=0.7)

# Validation loss çizimi (epoch bazında)
if "val_loss" in metrics_df.columns:
    val_loss_data = metrics_df.dropna(subset=["val_loss"])
    plt.plot(val_loss_data["epoch"], val_loss_data["val_loss"], label="Validation Loss (Epoch)", linewidth=2, marker='o')

plt.xlabel("Step / Epoch")
plt.ylabel("Loss")
plt.title("Training and Validation Loss")
plt.legend()
plt.grid(True)
plt.show()


# Tahmin ve Görselleştirme
predictions_and_x = tft.predict(val_dataloader, mode="raw", return_x=True)
raw_predictions = predictions_and_x.output
x = predictions_and_x.x

# `raw_predictions` içindeki doğru öğeyi seçme
if isinstance(raw_predictions, tuple):
    predictions = raw_predictions[2]  # Ana tahminlerin bulunduğu öğe

# predictions_squeezed boyutlarını kontrol etme
predictions_squeezed = predictions.squeeze(2)  # Üçüncü ekseni sıkıştır
print(f"predictions_squeezed shape: {predictions_squeezed.shape}")

# Görselleştirme - Tek bir grafik üzerinde tüm tahminler
plt.figure(figsize=(10, 6))  # Grafik boyutunu ayarlıyoruz

for idx in range(3):  # İlk 3 örnek için
    for i, col in enumerate(target):
        try:
            # predictions_squeezed[idx, i, :] ile doğru veriyi alıyoruz ve numpy formatına çeviriyoruz
            prediction_data = predictions_squeezed[idx, i, :].cpu().numpy()  # PyTorch tensor ise numpy'ye dönüştür
            # Her hedef değişken için çizim ekliyoruz
            plt.plot(prediction_data, label=f"{col} at index {idx}")
        except Exception as e:
            print(f"Error while plotting for index {idx}, column {col}: {e}")

# Grafik başlık ve etiketlerini ekliyoruz
plt.title("Predictions for Multiple Variables")
plt.xlabel('Time Steps')
plt.ylabel('Value')
plt.legend()  # Etiketleri gösteriyoruz
plt.grid(True)  # Izgara çizgilerini ekliyoruz
plt.show()


# Gelecek Tahminleri
future_predictions = tft.predict(val_dataloader, mode="prediction")

# Eğer future_predictions bir liste ise tensöre dönüştür
if isinstance(future_predictions, list):
    future_predictions = torch.cat([pred.clone().detach() for pred in future_predictions], dim=0)

# Ölçeklendirmeyi geri alma
future_predictions_rescaled = scaler.inverse_transform(future_predictions.numpy().reshape(-1, len(target)))

print("Gelecek tahminler (yeniden ölçeklendirilmiş):")
print(future_predictions_rescaled)

# 8. Tahminlerin Görselleştirilmesi
for i, col in enumerate(target):
    plt.figure(figsize=(10, 6))
    plt.plot(actuals_rescaled[:, i], label=f"Actual - {col}", color="blue")
    plt.plot(predicted_values_rescaled[:, i], label=f"Predicted - {col}", color="orange")
    plt.title(f"Comparison of Actual and Predicted for {col}")
    plt.xlabel("Time Step")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True)
    plt.show()

# 9. Gelecek Tahminleri
future_predictions = tft.predict(val_dataloader, mode="prediction")
if isinstance(future_predictions, list):
    future_predictions = torch.cat([pred.clone().detach() for pred in future_predictions], dim=0)

future_predictions_rescaled = scaler.inverse_transform(future_predictions.numpy().reshape(-1, len(target)))

# Gelecek Tahminleri Görselleştirme
future_index = range(len(future_predictions_rescaled))
for i, col in enumerate(target):
    plt.figure(figsize=(10, 6))
    plt.plot(future_index, future_predictions_rescaled[:, i], label=f"Future Prediction - {col}", color="green")
    plt.title(f"Future Predictions for {col}")
    plt.xlabel("Time Step")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True)
    plt.show()


    # Tüm tahminler ve gerçek değerleri tek bir grafik üzerinde gösterme
plt.figure(figsize=(12, 8))  # Grafik boyutunu ayarla

for i, col in enumerate(target):
    plt.plot(
        range(len(predicted_values_rescaled[:, i])),
        predicted_values_rescaled[:, i],
        label=f"Predicted - {col}",
        linestyle='--'
    )
    plt.plot(
        range(len(actuals_rescaled[:, i])),
        actuals_rescaled[:, i],
        label=f"Actual - {col}",
        alpha=0.7
    )

plt.title("Actual vs Predicted for All Targets")
plt.xlabel("Time Step")
plt.ylabel("Value")
plt.legend(loc="upper right", fontsize=9)  # Her hedef için etiketi göster
plt.grid(True)
plt.show()

# Doğrulama veri setindeki tarihleri alalım
dates = df["Date"].iloc[-len(actuals_rescaled):]  # Doğrulama setine denk gelen tarihler

# 7 günlük tahmin ve gerçek değerleri çizme
plt.figure(figsize=(12, 6))

for i, col in enumerate(target):
    # Son 7 gün için tahmin ve gerçek değerleri al
    last_7_actuals = actuals_rescaled[-7:, i]
    last_7_predictions = predicted_values_rescaled[-7:, i]
    last_7_dates = dates.iloc[-7:]

    # Gerçek değerleri çiz
    plt.plot(last_7_dates, last_7_actuals, label=f"Actual - {col}", marker="o", linestyle="-")
    # Tahminleri çiz
    plt.plot(last_7_dates, last_7_predictions, label=f"Predicted - {col}", marker="x", linestyle="--")

plt.title("7 Günlük Tahmin ve Gerçek Değerler (Tarih Bazında)")
plt.xlabel("Tarih")
plt.ylabel("Değer")
plt.legend(loc="upper left", fontsize=9)
plt.xticks(rotation=45)  # Tarih etiketlerini döndür
plt.grid(True)
plt.tight_layout()
plt.show()