import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
import streamlit as st

# --- 1. Konfigurasi Halaman ---
st.set_page_config(
    page_title="Sistem Prediksi Diabetes - KNN",
    page_icon="🧪",
    layout="wide"
)

# --- 2. Data Latihan (10 Pasien sesuai Studi Kasus) ---
raw_data = {
    'Pasien': ['P01', 'P02', 'P03', 'P04', 'P05', 'P06', 'P07', 'P08', 'P09', 'P10'],
    'Usia': [25, 30, 35, 40, 45, 50, 55, 28, 38, 48],
    'BMI': [21.5, 23.1, 25.0, 27.5, 30.2, 31.5, 33.0, 22.5, 26.8, 29.5],
    'Glukosa': [90, 95, 105, 120, 140, 155, 165, 92, 115, 145],
    'Tekanan_Darah': [110, 115, 120, 125, 135, 140, 145, 112, 122, 130],
    'Riwayat_Keluarga': [0, 0, 0, 1, 1, 1, 1, 0, 0, 1], # 0 = Tidak Ada, 1 = Ada
    'Kelas': [0, 0, 0, 1, 1, 1, 1, 0, 0, 1]              # 0 = Rendah, 1 = Tinggi
}

df_train = pd.DataFrame(raw_data)

# Deskrpsi label untuk tampilan tabel
df_display = df_train.copy()
df_display['Riwayat_Keluarga'] = df_display['Riwayat_Keluarga'].map({0: 'Tidak Ada (0)', 1: 'Ada (1)'})
df_display['Kelas'] = df_display['Kelas'].map({0: 'Risiko Rendah', 1: 'Risiko Tinggi'})

# --- 3. Header Aplikasi ---
st.title("🧪 Prediksi Risiko Diabetes berbasis KNN")
st.caption("Aplikasi Klasifikasi Kesehatan Menggunakan Algoritma K-Nearest Neighbors (KNN)")

st.divider()

# --- 4. Sidebar: Parameter & Input Pasien ---
with st.sidebar:
    st.header("⚙️ Pengaturan & Input")
    
    # Pilihan Nilai K
    k_value = st.slider("Pilih Nilai K (Jumlah Tetangga):", min_value=1, max_value=9, value=3, step=2)
    st.info(f"Model akan mengambil **{k_value} tetangga terdekat**.")
    
    st.subheader("Data Pasien Baru")
    input_usia = st.number_input("Usia (tahun)", min_value=1, max_value=120, value=42)
    input_bmi = st.number_input("BMI (kg/m²)", min_value=10.0, max_value=60.0, value=28.5, step=0.1)
    input_glukosa = st.number_input("Glukosa Darah (mg/dL)", min_value=50, max_value=300, value=130)
    input_td = st.number_input("Tekanan Darah (mmHg)", min_value=50, max_value=250, value=128)
    input_riwayat = st.selectbox("Riwayat Diabetes Keluarga", options=[0, 1], format_func=lambda x: "Ada (1)" if x == 1 else "Tidak Ada (0)")

    btn_predict = st.button("🚀 Jalankan Prediksi", type="primary", use_container_width=True)

# --- 5. Main Layout dengan Tab ---
tab1, tab2, tab3 = st.tabs(["📊 Hasil Klasifikasi", "📋 Dataset Latihan (10 Pasien)", "🧮 Detail Perhitungan Jarak"])

features = ['Usia', 'BMI', 'Glukosa', 'Tekanan_Darah', 'Riwayat_Keluarga']
X = df_train[features]
y = df_train['Kelas']

# Prapemrosesan Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

data_baru = np.array([[input_usia, input_bmi, input_glukosa, input_td, input_riwayat]])
data_baru_scaled = scaler.transform(data_baru)

# Melatih Model KNN dengan K pilihan user
knn = KNeighborsClassifier(n_neighbors=k_value)
knn.fit(X_scaled, y)

# Prediksi Hasil
pred_class = knn.predict(data_baru_scaled)[0]
pred_prob = knn.predict_proba(data_baru_scaled)[0]

# --- TAB 1: HASIL PREDIKSI ---
with tab1:
    st.subheader("Ringkasan Hasil Analisis Pasien")
    
    # Menampilkan ringkasan data input pasien baru
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Usia", f"{input_usia} th")
    c2.metric("BMI", f"{input_bmi}")
    c3.metric("Glukosa", f"{input_glukosa} mg/dL")
    c4.metric("Tekanan Darah", f"{input_td} mmHg")
    c5.metric("Riwayat Keluarga", "Ada" if input_riwayat == 1 else "Tidak")
    
    st.markdown("---")
    
    res_col1, res_col2 = st.columns([2, 1])
    
    with res_col1:
        if pred_class == 1:
            st.error("### ⚠️ Risiko Tinggi (Kelas 1)")
            st.write("Berdasarkan perhitungan KNN, pasien memiliki karakteristik kesehatan yang mendekati kelompok berisiko tinggi diabetes.")
        else:
            st.success("### ✅ Risiko Rendah (Kelas 0)")
            st.write("Berdasarkan perhitungan KNN, pasien berada pada kelompok berisiko rendah diabetes.")

    with res_col2:
        st.metric("Probabilitas Risiko Tinggi", f"{pred_prob[1]*100:.1f}%")
        st.metric("Probabilitas Risiko Rendah", f"{pred_prob[0]*100:.1f}%")

# --- TAB 2: DATASET LATIHAN ---
with tab2:
    st.subheader("Data Latihan (10 Pasien)")
    st.write("Berikut adalah 10 data historis pasien yang digunakan sebagai acuan dasar klasifikasi model KNN:")
    st.dataframe(df_display, use_container_width=True, hide_index=True)

# --- TAB 3: DETAIL PERHITUNGAN JARAK ---
with tab3:
    st.subheader(f"Jarak Euclidean ke-{len(df_train)} Pasien (Standar Ter-Skala)")
    st.write(f"Berikut urutan jarak dari data pasien baru ke 10 data acuan untuk menentukan {k_value} tetangga terdekat:")
    
    # Perhitungan Jarak Euclidean Sederhana untuk transparansi
    distances = np.linalg.norm(X_scaled - data_baru_scaled, axis=1)
    df_calc = df_train.copy()
    df_calc['Jarak_Euclidean'] = distances
    df_calc['Status_Tetangga'] = "Bukan"
    
    # Mengurutkan berdasarkan jarak terkecil
    df_sorted = df_calc.sort_values(by='Jarak_Euclidean').reset_index(drop=True)
    df_sorted.loc[:k_value-1, 'Status_Tetangga'] = f"Top {k_value} Tetangga"
    
    df_sorted['Kelas_Label'] = df_sorted['Kelas'].map({0: 'Risiko Rendah', 1: 'Risiko Tinggi'})
    
    st.dataframe(
        df_sorted[['Pasien', 'Usia', 'BMI', 'Glukosa', 'Tekanan_Darah', 'Kelas_Label', 'Jarak_Euclidean', 'Status_Tetangga']],
        use_container_width=True,
        hide_index=True
    )

# Footer
st.caption("---")
st.caption("📌 Catatan: Aplikasi ini dibuat untuk keperluan akademik/pembelajaran klasifikasi Data Mining menggunakan metode KNN.")