import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
import streamlit as st

# --- Konfigurasi Halaman ---
st.set_page_config(
    page_title="Prediksi Risiko Diabetes",
    page_icon="🩺",
    layout="centered"
)

# --- Data Latihan (10 Pasien sesuai Studi Kasus) ---
raw_data = {
    'Pasien': ['P01', 'P02', 'P03', 'P04', 'P05', 'P06', 'P07', 'P08', 'P09', 'P10'],
    'Usia': [25, 30, 35, 40, 45, 50, 55, 28, 38, 48],
    'BMI': [21.5, 23.1, 25.0, 27.5, 30.2, 31.5, 33.0, 22.5, 26.8, 29.5],
    'Glukosa': [90, 95, 105, 120, 140, 155, 165, 92, 115, 145],
    'Tekanan_Darah': [110, 115, 120, 125, 135, 140, 145, 112, 122, 130],
    'Riwayat_Keluarga': [0, 0, 0, 1, 1, 1, 1, 0, 0, 1],
    'Kelas': [0, 0, 0, 1, 1, 1, 1, 0, 0, 1]
}

df_train = pd.DataFrame(raw_data)

# Label untuk tampilan tabel
df_display = df_train.copy()
df_display['Riwayat_Keluarga'] = df_display['Riwayat_Keluarga'].map({0: 'Tidak ada (0)', 1: 'Ada (1)'})
df_display['Kelas'] = df_display['Kelas'].map({0: 'Risiko Rendah', 1: 'Risiko Tinggi'})

# --- Header Aplikasi ---
st.markdown("### SISTEM PENDUKUNG KEPUTUSAN")
st.title("Prediksi Risiko Diabetes")
st.markdown("Klasifikasi menggunakan algoritma K-Nearest Neighbors (KNN)")

st.divider()

# --- Form Input Data Pasien (Model Kotak Tengah seperti Foto 2) ---
st.subheader("Input Data Pasien")
st.markdown("Lengkapi nilai berikut untuk menjalankan klasifikasi.")

with st.form("form_prediksi"):
    col1, col2 = st.columns(2)
    
    with col1:
        usia = st.number_input("Usia (tahun)", min_value=1, max_value=120, value=40)
        bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=28.0, step=0.1)
        riwayat_keluarga = st.selectbox(
            "Riwayat Diabetes dalam Keluarga",
            options=[0, 1],
            format_func=lambda x: "Tidak ada (0)" if x == 0 else "Ada (1)"
        )
        
    with col2:
        glukosa = st.number_input("Glukosa (mg/dL)", min_value=50, max_value=300, value=130)
        tekanan_darah = st.number_input("Tekanan Darah", min_value=50, max_value=250, value=128)
        k_value = st.selectbox("Nilai K (jumlah tetangga)", options=[1, 3, 5, 7], index=1)

    submit_button = st.form_submit_button(label="Proses Prediksi", use_container_width=True)

# --- Proses Prediksi dan Perhitungan ---
if submit_button:
    features = ['Usia', 'BMI', 'Glukosa', 'Tekanan_Darah', 'Riwayat_Keluarga']
    X = df_train[features]
    y = df_train['Kelas']

    # Standarisasi data agar perhitungan jarak akurat
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    data_baru = np.array([[usia, bmi, glukosa, tekanan_darah, riwayat_keluarga]])
    data_baru_scaled = scaler.transform(data_baru)

    # Model KNN
    knn = KNeighborsClassifier(n_neighbors=k_value)
    knn.fit(X_scaled, y)

    pred_class = knn.predict(data_baru_scaled)[0]
    pred_prob = knn.predict_proba(data_baru_scaled)[0]

    st.markdown("---")
    st.subheader("Hasil Analisis & Perhitungan Jarak K-NN")
    
    if pred_class == 1:
        st.error(f"⚠️ **Risiko Tinggi** (Kelas 1) — Probabilitas: **{pred_prob[1]*100:.1f}%**")
    else:
        st.success(f"✅ **Risiko Rendah** (Kelas 0) — Probabilitas: **{pred_prob[0]*100:.1f}%**")

    # Menghitung jarak Euclidean ke seluruh data latih
    distances = np.linalg.norm(X_scaled - data_baru_scaled, axis=1)
    df_calc = df_train.copy()
    df_calc['Jarak_Euclidean'] = distances
    df_sorted = df_calc.sort_values(by='Jarak_Euclidean').reset_index(drop=True)
    
    st.write(f"**Rincian {k_value} Tetangga Terdekat (Nearest Neighbors):**")
    df_neighbor_show = df_sorted.head(k_value)[['Pasien', 'Usia', 'BMI', 'Glukosa', 'Tekanan_Darah', 'Riwayat_Keluarga', 'Kelas', 'Jarak_Euclidean']]
    df_neighbor_show['Kelas'] = df_neighbor_show['Kelas'].map({0: 'Rendah', 1: 'Tinggi'})
    st.dataframe(df_neighbor_show, use_container_width=True, hide_index=True)

st.markdown("---")

# --- Expander Dataset Latihan (10 Data) ---
with st.expander("📂 Lihat dataset latihan (10 data)"):
    st.markdown("Data studi kasus yang digunakan sebagai data latihan model KNN[cite: 3].")
    st.dataframe(df_display, use_container_width=True, hide_index=True)

st.markdown("---")
st.caption("Metode yang digunakan: StandardScaler, KNN, dan jarak Euclidean. Dataset berisi 10 data studi kasus dan hanya digunakan untuk keperluan akademik. Hasil klasifikasi bukan diagnosis medis[cite: 3].")