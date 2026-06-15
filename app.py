import streamlit as st
import pickle
import numpy as np

# Konfigurasi halaman
st.set_page_config(
    page_title="Prediksi Prestasi Akademik Siswa",
    page_icon="🎓",
    layout="wide"
)

# Load model dan scaler
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

# Header
st.markdown("""
# 🎓 Prediksi Prestasi Akademik Siswa
### Menggunakan Metode Supervised Learning

Masukkan data siswa untuk memprediksi nilai GPA.
""")

st.divider()

# Layout 2 kolom
col1, col2 = st.columns(2)

with col1:
    st.subheader("📚 Faktor Akademik")

    age = st.slider("Usia", 15, 25, 17)

    study_time = st.slider(
        "Jam Belajar per Minggu",
        0, 40, 10
    )

    absences = st.slider(
        "Jumlah Ketidakhadiran",
        0, 30, 5
    )

with col2:
    st.subheader("⭐ Faktor Pendukung")

    parental_support = st.slider(
        "Dukungan Orang Tua",
        0, 4, 2
    )

    activity_score = st.slider(
        "Activity Score",
        0, 10, 3
    )

    academic_engagement = st.slider(
        "Academic Engagement",
        0, 20, 5
    )

st.divider()

# Tombol Prediksi
if st.button("🔍 Prediksi GPA", use_container_width=True):

    data = np.array([
        age,
        study_time,
        absences,
        parental_support,
        academic_engagement,
        activity_score
    ]).reshape(1, -1)

    data = scaler.transform(data)

    prediksi = model.predict(data)

    st.success(
        f"🎯 Prediksi GPA Siswa: {prediksi[0]:.2f}"
    )

    if prediksi[0] >= 3.5:
        st.balloons()
        st.info("Kategori: Prestasi Sangat Baik")
    elif prediksi[0] >= 3.0:
        st.info("Kategori: Prestasi Baik")
    elif prediksi[0] >= 2.0:
        st.warning("Kategori: Prestasi Cukup")
    else:
        st.error("Kategori: Perlu Pendampingan Belajar")
