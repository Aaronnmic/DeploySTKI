import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from io import StringIO

# Judul Aplikasi
st.title("Klasifikasi Cuaca Berdasarkan Deskripsi")
st.write("""
Aplikasi ini bertujuan untuk mengklasifikasikan cuaca berdasarkan data iklim.
Kategori cuaca yang digunakan adalah sebagai berikut:
- **Cerah**: Curah hujan = 0 mm
- **Berawan**: Curah hujan di antara 0 dan 20 mm
- **Hujan**: Curah hujan lebih dari 20 mm
""")

# Baca Dataset Langsung dari File Lokal
try:
    data = pd.read_csv("climate_data.csv")
    st.write("Dataset berhasil dimuat! Berikut adalah beberapa baris awal:")
    st.write(data.head())

    # Eksplorasi Data Awal (EDA)
    st.subheader("Informasi Dataset")
    st.write("Berikut adalah struktur dataset yang digunakan:")
    buffer = StringIO()
    data.info(buf=buffer)
    info_str = buffer.getvalue()
    st.text(info_str)
    st.write("Statistik Dataset")
    st.write(data.describe())

    # Preprocessing Data
    data = data.dropna()  # Menghapus data dengan nilai kosong
    data['date'] = pd.to_datetime(data['date'], errors='coerce')

    # Labeling Dataset
    def label_weather(row):
        if row['RR'] == 0:
            return 'Cerah'
        elif row['RR'] > 0 and row['RR'] <= 20:
            return 'Berawan'
        else:
            return 'Hujan'

    data['weather_category'] = data.apply(label_weather, axis=1)

    st.subheader("Distribusi Kategori Cuaca")
    st.write("""
Distribusi kategori cuaca menggambarkan jumlah data untuk setiap label:
- **Cerah**: Hari tanpa curah hujan.
- **Berawan**: Hari dengan curah hujan ringan (antara 0 dan 20 mm).
- **Hujan**: Hari dengan curah hujan tinggi (lebih dari 20 mm).
    """)
    st.bar_chart(data['weather_category'].value_counts())

    # Pemodelan dan Evaluasi
    X = data[['Tn', 'Tx', 'Tavg', 'RH_avg']]  # Fitur numerik utama
    y = data['weather_category']  # Label kategori

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    # Evaluasi Model
    y_pred = model.predict(X_test)

    st.subheader("Laporan Klasifikasi")
    st.write("""
Laporan klasifikasi menunjukkan performa model untuk setiap label:
- **Precision**: Proporsi prediksi benar terhadap total prediksi untuk label tersebut.
- **Recall**: Proporsi prediksi benar terhadap total data aktual untuk label tersebut.
- **F1-Score**: Rata-rata harmonis antara precision dan recall.
""")
    st.text(classification_report(y_test, y_pred))

    # Menampilkan Confusion Matrix
    conf_matrix = confusion_matrix(y_test, y_pred)
    st.subheader("Confusion Matrix")
    st.write("""
Confusion Matrix menunjukkan jumlah prediksi benar dan salah untuk setiap label.
- Baris mewakili label sebenarnya (Actual).
- Kolom mewakili label yang diprediksi (Predicted).
""")
    fig, ax = plt.subplots()
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=model.classes_, yticklabels=model.classes_, ax=ax)
    st.pyplot(fig)

except FileNotFoundError:
    st.error("Dataset tidak ditemukan. Pastikan file climate_data.csv ada di direktori aplikasi.")
