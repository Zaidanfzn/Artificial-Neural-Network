import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.initializers import HeNormal
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

data = {
    "Suhu": [60, 60, 60, 60, 60, 70, 70, 70, 70, 70, 80, 80, 80, 80, 80, 90, 90, 90, 90, 90, 100, 100, 100, 100, 100, 60, 70, 80, 90, 100],
    "Waktu": [5, 10, 15, 20, 30, 5, 10, 15, 20, 30, 5, 10, 15, 20, 30, 5, 10, 15, 20, 30, 5, 10, 15, 20, 30, 25, 25, 25, 25, 25],
    "Kadar_Air": [80, 75, 70, 65, 60, 80, 75, 70, 65, 60, 80, 75, 70, 65, 60, 80, 75, 70, 65, 60, 80, 75, 70, 65, 60, 68, 72, 66, 62, 60],
    "Kadar_Nutrisi": [20, 25, 30, 28, 27, 32, 35, 40, 45, 43, 48, 50, 55, 60, 58, 53, 56, 62, 68, 65, 70, 73, 75, 78, 80, 26, 41, 57, 66, 72]
}

df = pd.DataFrame(data)
X = df[["Suhu", "Waktu", "Kadar_Air"]]
y = df["Kadar_Nutrisi"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = Sequential()
model.add(Dense(10, input_dim=3, activation='relu', kernel_initializer=HeNormal()))
model.add(Dense(8, activation='relu', kernel_initializer=HeNormal()))
model.add(Dense(1, activation='linear'))
model.compile(optimizer='adam', loss='mean_squared_error')
history = model.fit(X_train, y_train, epochs=500, batch_size=32, validation_data=(X_test, y_test))

loss = model.evaluate(X_test, y_test)
print("Loss pada data uji (NN):", loss)

plt.figure(figsize=(10,6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss (MSE)')
plt.legend()
plt.grid(True)
plt.show()

predictions = model.predict(X_test)
plt.figure(figsize=(10,6))
plt.scatter(y_test, predictions, color='blue')
plt.xlabel("Nilai Aktual")
plt.ylabel("Prediksi")
plt.title("Hasil Prediksi vs Nilai Aktual")
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r')
plt.grid(True)
plt.show()

# Pembagian data latih (70%) dan data uji (30%) dengan proporsi yang benar
data_latih = [
    {'suhu': 60, 'waktu': 5, 'kadar_air': 80, 'nutrisi': 20.0},
    {'suhu': 60, 'waktu': 10, 'kadar_air': 75, 'nutrisi': 25.0},
    {'suhu': 60, 'waktu': 15, 'kadar_air': 70, 'nutrisi': 30.0},
    {'suhu': 70, 'waktu': 5, 'kadar_air': 80, 'nutrisi': 32.0},
    {'suhu': 70, 'waktu': 10, 'kadar_air': 75, 'nutrisi': 35.0},
    {'suhu': 80, 'waktu': 5, 'kadar_air': 80, 'nutrisi': 48.0},
    {'suhu': 80, 'waktu': 10, 'kadar_air': 75, 'nutrisi': 50.0},
    {'suhu': 90, 'waktu': 5, 'kadar_air': 80, 'nutrisi': 53.0},
    {'suhu': 90, 'waktu': 10, 'kadar_air': 75, 'nutrisi': 56.0},
    {'suhu': 90, 'waktu': 15, 'kadar_air': 70, 'nutrisi': 62.0},
    {'suhu': 100, 'waktu': 5, 'kadar_air': 80, 'nutrisi': 70.0},
    {'suhu': 100, 'waktu': 10, 'kadar_air': 75, 'nutrisi': 73.0},
    {'suhu': 100, 'waktu': 15, 'kadar_air': 70, 'nutrisi': 75.0},
    {'suhu': 80, 'waktu': 20, 'kadar_air': 65, 'nutrisi': 65.0},
    {'suhu': 70, 'waktu': 20, 'kadar_air': 65, 'nutrisi': 43.0},
    {'suhu': 90, 'waktu': 20, 'kadar_air': 65, 'nutrisi': 65.0},
    {'suhu': 80, 'waktu': 25, 'kadar_air': 65, 'nutrisi': 48.0},
    {'suhu': 85, 'waktu': 15, 'kadar_air': 75, 'nutrisi': 60.0},
    {'suhu': 95, 'waktu': 18, 'kadar_air': 80, 'nutrisi': 62.0},
    {'suhu': 92, 'waktu': 22, 'kadar_air': 72, 'nutrisi': 58.0},
    {'suhu': 100, 'waktu': 20, 'kadar_air': 65, 'nutrisi': 78.0},
    {'suhu': 70, 'waktu': 15, 'kadar_air': 70, 'nutrisi': 40.0},
    {'suhu': 80, 'waktu': 20, 'kadar_air': 65, 'nutrisi': 58.0},
    {'suhu': 100, 'waktu': 25, 'kadar_air': 60, 'nutrisi': 60.0},
]

X_latih = np.array([[data['suhu'], data['waktu'], data['kadar_air']] for data in data_latih])
y_latih = np.array([data['nutrisi'] for data in data_latih])

lr_model = LinearRegression()
lr_model.fit(X_latih, y_latih)

data_uji = [
    {'suhu': 90, 'waktu': 20, 'kadar_air': 70, 'nutrisi': 55.0},
    {'suhu': 85, 'waktu': 18, 'kadar_air': 77, 'nutrisi': 61.0},
    {'suhu': 100, 'waktu': 30, 'kadar_air': 65, 'nutrisi': 65.0},
    {'suhu': 60, 'waktu': 25, 'kadar_air': 68, 'nutrisi': 26.0},
    {'suhu': 70, 'waktu': 25, 'kadar_air': 72, 'nutrisi': 41.0},
    {'suhu': 80, 'waktu': 25, 'kadar_air': 66, 'nutrisi': 57.0},
]

X_uji = np.array([[data['suhu'], data['waktu'], data['kadar_air']] for data in data_uji])
y_uji = np.array([data['nutrisi'] for data in data_uji])

y_pred_lr = lr_model.predict(X_uji)

for i in range(len(data_uji)):
    print(f"Suhu: {data_uji[i]['suhu']}°C, Waktu: {data_uji[i]['waktu']} menit, Kadar Air: {data_uji[i]['kadar_air']}%, "
          f"Prediksi Kadar Nutrisi (LR): {y_pred_lr[i]:.2f} mg/100g, Kadar Nutrisi Sesungguhnya: {y_uji[i]} mg/100g")
    error = abs(y_pred_lr[i] - y_uji[i])
    print(f"Selisih Prediksi (LR): {error:.2f} mg/100g\n")

mae_lr = mean_absolute_error(y_uji, y_pred_lr)
rmse_lr = np.sqrt(mean_squared_error(y_uji, y_pred_lr))

print(f"Mean Absolute Error (MAE) LR: {mae_lr:.2f}")
print(f"Root Mean Squared Error (RMSE) LR: {rmse_lr:.2f}")