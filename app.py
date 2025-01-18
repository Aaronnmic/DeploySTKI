import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

# Judul Aplikasi
st.title("Klasifikasi Cuaca Berdasarkan Deskripsi")
st.write("Proyek ini bertujuan untuk mengklasifikasikan cuaca berdasarkan data iklim seperti suhu, kelembapan, dan curah hujan.")

# Upload Dataset
uploaded_file = st.file_uploader("Upload Dataset CSV", type="csv")
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)  # Memperbaiki kesalahan membaca dataset
    st.write("Dataset berhasil dimuat! Berikut adalah beberapa baris awal:")
    st.write(data.head())

    # Eksplorasi Data Awal (EDA)
    st.subheader("Informasi Dataset")
    st.text("Info Dataset:")
    buffer = []
    data.info(buf=buffer)
    info_str = "\n".join(buffer)
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
    st.text(classification_report(y_test, y_pred))

    # Menampilkan Confusion Matrix
    conf_matrix = confusion_matrix(y_test, y_pred)
    st.subheader("Confusion Matrix")
    fig, ax = plt.subplots()
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=model.classes_, yticklabels=model.classes_, ax=ax)
    st.pyplot(fig)
