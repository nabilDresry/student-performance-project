import streamlit as st
import joblib
import numpy as np

# =====================================
# KONFIGURASI HALAMAN
# =====================================

st.set_page_config(
    page_title="Prediksi Prestasi Akademik Siswa",
    page_icon="🎓",
    layout="wide"
)

# =====================================
# LOAD MODEL
# =====================================

try:
    model = joblib.load("model.pkl")
    scaler = joblib.load("scaler.pkl")
except Exception as e:
    st.error(f"Gagal memuat model: {e}")
    st.stop()

# =====================================
# HEADER
# =====================================

st.title("🎓 Prediksi Prestasi Akademik Siswa")

st.markdown("""
### Berdasarkan Faktor Akademik dan Aktivitas Non-Akademik Menggunakan Metode Supervised Learning

Aplikasi ini memprediksi GPA siswa menggunakan model Random Forest Regression.
""")

st.divider()

# =====================================
# INPUT
# =====================================

col1, col2 = st.columns(2)

with col1:

    st.subheader("📚 Faktor Akademik")

    age = st.number_input(
        "Usia Siswa",
        min_value=15,
        max_value=25,
        value=17,
        step=1
    )

    study_time = st.number_input(
        "Jam Belajar per Minggu",
        min_value=0,
        max_value=50,
        value=10,
        step=1
    )

    absences = st.number_input(
        "Jumlah Ketidakhadiran",
        min_value=0,
        max_value=50,
        value=5,
        step=1
    )

    parental_support = st.number_input(
        "Dukungan Orang Tua (0-4)",
        min_value=0,
        max_value=4,
        value=2,
        step=1
    )

with col2:

    st.subheader("⭐ Aktivitas Non-Akademik")

    extracurricular = st.checkbox("Ekstrakurikuler")

    sports = st.checkbox("Olahraga")

    music = st.checkbox("Musik")

    volunteering = st.checkbox("Volunteering")

# =====================================
# FEATURE ENGINEERING
# =====================================

activity_score = (
    int(extracurricular)
    + int(sports)
    + int(music)
    + int(volunteering)
)

study_efficiency = (
    study_time / (absences + 1)
)

support_activity = (
    parental_support * activity_score
)

academic_engagement = (
    study_time + parental_support
) / (absences + 1)

# =====================================
# TAMPILKAN FITUR BARU
# =====================================

st.divider()

st.subheader("📊 Hasil Feature Engineering")

c1, c2, c3, c4 = st.columns(4)

c1.metric("Activity Score", activity_score)
c2.metric("Study Efficiency", round(study_efficiency, 2))
c3.metric("Support Activity", support_activity)
c4.metric("Academic Engagement", round(academic_engagement, 2))

# =====================================
# PREDIKSI
# =====================================

if st.button("🔍 Prediksi GPA", use_container_width=True):

    data = np.array([
        age,
        study_time,
        absences,
        parental_support,
        activity_score,
        study_efficiency,
        support_activity,
        academic_engagement
    ]).reshape(1, -1)

    data = scaler.transform(data)

    prediction = model.predict(data)

    gpa = float(prediction[0])

    gpa = max(0.0, min(4.0, gpa))

    st.divider()

    st.subheader("🎯 Hasil Prediksi")

    st.metric(
        "Prediksi GPA",
        f"{gpa:.2f}"
    )

    st.progress(gpa / 4)

    if gpa >= 3.50:
        st.success("🏆 Prestasi Sangat Baik")
        st.balloons()

    elif gpa >= 3.00:
        st.info("📘 Prestasi Baik")

    elif gpa >= 2.00:
        st.warning("📙 Prestasi Cukup")

    else:
        st.error("📕 Perlu Pendampingan Belajar")

# =====================================
# SIDEBAR
# =====================================

st.sidebar.title("📈 Informasi Model")

st.sidebar.success(
    "Random Forest Regression"
)

st.sidebar.write(
    "Target Prediksi: GPA"
)

st.sidebar.write(
    "Jumlah Fitur: 8"
)

st.sidebar.divider()

st.sidebar.subheader("Performa Model")

# GANTI DENGAN HASIL COLAB
st.sidebar.metric("R² Score", "ISI_R2")
st.sidebar.metric("MAE", "ISI_MAE")
st.sidebar.metric("RMSE", "ISI_RMSE")

st.sidebar.divider()

st.sidebar.info(
    "Prediksi Prestasi Akademik Siswa menggunakan Supervised Learning."
)
