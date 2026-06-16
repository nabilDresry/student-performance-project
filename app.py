import streamlit as st
import pickle
import numpy as np

# =========================
# KONFIGURASI HALAMAN
# =========================

st.set_page_config(
    page_title="Prediksi Prestasi Akademik Siswa",
    page_icon="🎓",
    layout="wide"
)

# =========================
# LOAD MODEL
# =========================

model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

# =========================
# HEADER
# =========================

st.title("🎓 Prediksi Prestasi Akademik Siswa")
st.markdown("""
Aplikasi ini memprediksi **GPA siswa** berdasarkan faktor akademik
dan aktivitas non-akademik menggunakan metode **Supervised Learning**.
""")

st.divider()

# =========================
# INPUT DATA
# =========================

col1, col2 = st.columns(2)

with col1:
    st.subheader("📚 Faktor Akademik")

    age = st.slider(
        "Usia Siswa",
        min_value=15,
        max_value=25,
        value=17
    )

    study_time = st.slider(
        "Jam Belajar per Minggu",
        min_value=0,
        max_value=40,
        value=10
    )

    absences = st.slider(
        "Jumlah Ketidakhadiran",
        min_value=0,
        max_value=30,
        value=5
    )

with col2:
    st.subheader("⭐ Faktor Pendukung")

    parental_support = st.slider(
        "Dukungan Orang Tua",
        min_value=0,
        max_value=4,
        value=2
    )

    extracurricular = st.checkbox("Mengikuti Ekstrakurikuler")

    sports = st.checkbox("Mengikuti Kegiatan Olahraga")

    music = st.checkbox("Mengikuti Kegiatan Musik")

    volunteering = st.checkbox("Mengikuti Kegiatan Sosial")

# =========================
# FEATURE ENGINEERING
# =========================

activity_score = (
    int(extracurricular)
    + int(sports)
    + int(music)
    + int(volunteering)
)

academic_engagement = (
    study_time / (absences + 1)
)

# =========================
# TAMPILKAN FITUR HASIL
# =========================

st.divider()

st.subheader("📊 Hasil Feature Engineering")

c1, c2 = st.columns(2)

c1.metric(
    "Academic Engagement",
    f"{academic_engagement:.2f}"
)

c2.metric(
    "Activity Score",
    activity_score
)

# =========================
# PREDIKSI
# =========================

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

    gpa = float(prediksi[0])

    st.divider()

    st.subheader("🎯 Hasil Prediksi")

    st.metric(
        "Prediksi GPA",
        f"{gpa:.2f}"
    )

    st.progress(
        min(gpa / 4.0, 1.0)
    )

    if gpa >= 3.5:
        st.success("🏆 Kategori Prestasi: Sangat Baik")
        st.balloons()

    elif gpa >= 3.0:
        st.info("📘 Kategori Prestasi: Baik")

    elif gpa >= 2.0:
        st.warning("📙 Kategori Prestasi: Cukup")

    else:
        st.error("📕 Kategori Prestasi: Perlu Pendampingan")

# =========================
# INFORMASI MODEL
# =========================

st.sidebar.header("📈 Informasi Model")

st.sidebar.write(
    "Model digunakan untuk memprediksi GPA siswa berdasarkan faktor akademik dan aktivitas non-akademik."
)

st.sidebar.success("Model: Random Forest Regression")

# GANTI DENGAN HASIL EVALUASI DARI COLAB
st.sidebar.write("R² Score : isi_hasil_anda")
st.sidebar.write("MAE : isi_hasil_anda")
st.sidebar.write("RMSE : isi_hasil_anda")
