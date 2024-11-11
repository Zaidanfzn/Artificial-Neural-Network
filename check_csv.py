import pandas as pd

# Load data CSV
data = pd.read_csv('Data Hasil Pengukuran.csv')

# Menampilkan nama-nama kolom dalam DataFrame
print("Nama kolom sebelum perbaikan:")
print(data.columns)

# Menghapus spasi ekstra dari nama kolom
data.columns = data.columns.str.strip()

# Menampilkan nama kolom setelah perbaikan
print("\nNama kolom setelah perbaikan:")
print(data.columns)

# Menyimpan file CSV yang sudah diperbaiki jika diperlukan
data.to_csv('Data_Hasil_Pengukuran_Perbaikan.csv', index=False)
