import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Data latih yang digunakan untuk melatih model (contoh data)
data_latih = [
    {'suhu': 90, 'waktu': 20, 'kadar_air': 70, 'nutrisi': 55.0},
    {'suhu': 80, 'waktu': 25, 'kadar_air': 65, 'nutrisi': 48.0},
    {'suhu': 85, 'waktu': 15, 'kadar_air': 75, 'nutrisi': 60.0},
    {'suhu': 95, 'waktu': 18, 'kadar_air': 80, 'nutrisi': 62.0},
    {'suhu': 92, 'waktu': 22, 'kadar_air': 72, 'nutrisi': 58.0},
    # Tambahkan lebih banyak data sesuai kebutuhan
]

# Menyiapkan input dan target untuk data latih
X_latih = np.array([[data['suhu'], data['waktu'], data['kadar_air']] for data in data_latih])
y_latih = np.array([data['nutrisi'] for data in data_latih])

# Membuat model Linear Regression dan melatihnya
model = LinearRegression()
model.fit(X_latih, y_latih)

# Data uji yang diketahui (input data dan nilai nutrisi sesungguhnya)
data_uji = [
    {'suhu': 90, 'waktu': 20, 'kadar_air': 70, 'nutrisi': 55.0},
    {'suhu': 80, 'waktu': 25, 'kadar_air': 65, 'nutrisi': 48.0},
    {'suhu': 85, 'waktu': 18, 'kadar_air': 77, 'nutrisi': 61.0},
    {'suhu': 100, 'waktu': 30, 'kadar_air': 65, 'nutrisi': 65.0},
    # Tambahkan lebih banyak data uji sesuai kebutuhan
]

# Menyiapkan input dan target untuk data uji
X_uji = np.array([[data['suhu'], data['waktu'], data['kadar_air']] for data in data_uji])
y_uji = np.array([data['nutrisi'] for data in data_uji])

# Menguji model dengan data uji dan menghitung error
y_pred = model.predict(X_uji)

# Mencetak hasil prediksi dan nilai sesungguhnya
for i in range(len(data_uji)):
    print(f"Suhu: {data_uji[i]['suhu']}°C, Waktu: {data_uji[i]['waktu']} menit, Kadar Air: {data_uji[i]['kadar_air']}%, "
          f"Prediksi Kadar Nutrisi: {y_pred[i]:.2f} mg/100g, Kadar Nutrisi Sesungguhnya: {y_uji[i]} mg/100g")
    
    # Menghitung selisih antara prediksi dan nilai sesungguhnya
    error = abs(y_pred[i] - y_uji[i])
    print(f"Selisih Prediksi: {error:.2f} mg/100g\n")

# Menghitung dan menampilkan metrik evaluasi MAE dan RMSE
mae = mean_absolute_error(y_uji, y_pred)
rmse = np.sqrt(mean_squared_error(y_uji, y_pred))

print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
