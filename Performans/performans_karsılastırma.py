# PERFORMANS KARŞILAŞTIRMALARI
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler

# Verileri oluşturma
data = {
    'Model': ['LSTNet', 'Bayesian Reformer', 'Autoformer', 'Reformer', 'TFT', 'Vanilla', 'TST', 'Informer'],
    'MSE': [0.3283, 0.3368, 0.3404075503349304, 0.3340, 66.4438, 0.3330, 0.3531, 0.4121],
    'MAE': [0.4007, 0.4032, 0.40983375906944275, 0.4007, 5.5517, 0.4025, 0.4150, 0.4569],
    'RMSE': [0.5730, 0.5803, 0.5834445563504131, 0.5779, 6.8729, 0.5770, 0.5943, 0.6420],
    'R2': [0.6745, 0.6657, 0.6616665124893188, 0.6683, -0.3621, 0.6694, 0.6502, 0.5869],
    'MAPE': [2.0265, 2.1397, 2.004774332046509, 2.0910, 35352.6643, 2.0644, 2.0166, 2.2587],
    'Training Time (s)': [120.14, 1206.08, 487.60, 834.09, 1036.72, 83.81, 285.38, 212.57],
    'Inference Time (s)': [0.02, 0.13, 1.21, 0.12, 21.17, 0.02, 0.05, 0.04]
}

df = pd.DataFrame(data)

# Performans karşılaştırmasını görselleştirme
# 1. MSE Karşılaştırması
plt.figure(figsize=(10, 6))
sns.barplot(data=df, x='Model', y='MSE', palette='viridis')
plt.title('Model MSE Comparison')
plt.xticks(rotation=45)
plt.ylabel('Mean Squared Error (MSE)')
plt.show()

# 2. Eğitim Süresi Karşılaştırması
plt.figure(figsize=(10, 6))
sns.barplot(data=df, x='Model', y='Training Time (s)', palette='viridis')
plt.title('Model Training Time Comparison')
plt.xticks(rotation=45)
plt.ylabel('Training Time (seconds)')
plt.show()

# 3. R2 Değerleri Karşılaştırması
plt.figure(figsize=(10, 6))
sns.barplot(data=df, x='Model', y='R2', palette='viridis')
plt.title('Model R2 Score Comparison')
plt.xticks(rotation=45)
plt.ylabel('R2 Score')
plt.show()

# 4. MAPE Karşılaştırması (Log Ölçeği ile, büyük değer farkları nedeniyle)
plt.figure(figsize=(10, 6))
sns.barplot(data=df, x='Model', y='MAPE', palette='viridis', log=True)
plt.title('Model MAPE Comparison (Log Scale)')
plt.xticks(rotation=45)
plt.ylabel('Mean Absolute Percentage Error (MAPE)')
plt.show()

# Normalleştirme
metrics = ['MSE', 'MAE', 'RMSE', 'R2', 'MAPE', 'Training Time (s)', 'Inference Time (s)']
scaler = MinMaxScaler()
scaled_metrics = scaler.fit_transform(df[metrics])

# Normalize edilmiş metriklerle genel skor hesaplama
normalized_df = pd.DataFrame(scaled_metrics, columns=metrics)
normalized_df['Model'] = df['Model']
normalized_df['Overall Score'] = normalized_df[['MSE', 'MAE', 'RMSE', 'MAPE', 'Training Time (s)', 'Inference Time (s)']].mean(axis=1) - normalized_df['R2']

plt.figure(figsize=(10, 6))
sns.barplot(data=normalized_df, x='Model', y='Overall Score', palette='viridis')
plt.title('Overall Model Evaluation Scores')
plt.xticks(rotation=45)
plt.ylabel('Overall Score (Lower is Better)')
plt.show()

# Genel skorları orijinal tabloya ekleme
df['Overall Score'] = normalized_df['Overall Score']

# Tabloyu görüntüleme
from IPython.display import display
display(df)

# Tabloyu CSV olarak kaydetme
df.to_csv('model_performance_with_overall_score.csv', index=False)

# CSV dosyası kaydedildi
print("CSV file 'model_performance_with_overall_score.csv' has been saved.")
