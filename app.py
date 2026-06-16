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
# CUSTOM CSS
# =====================================

st.markdown("""
<style>
.main {
    padding-top: 1rem;
}

.big-font {
    font-size:35px !important;
    font-weight:bold;
    text-align:center;
}

.result-box {
    padding:20px;
    border-radius:15px;
    background-color:#1e293b;
    text-align:center;
}

.metric-card {
    padding:15px;
    border-radius:10px;
    background-color:#111827;
}
</style>
""", unsafe_allow_html=True)

# =====================================
# HEADER
# =====================================

st.markdown(
    "<p class='big-font'>🎓 Prediksi Prestasi Akademik Siswa</p>",
    unsafe_allow_html=True
)

st.markdown("""
### Berdasarkan Faktor Akademik dan Aktivitas Non-Akademik Menggunakan Metode Supervised Learning

Aplikasi ini memprediksi **GPA siswa**
berdasarkan faktor akademik serta aktivitas
non-akademik menggunakan model
**Random Forest Regression**.
""")

st.divider()

# =====================================
# INPUT
# =====================================

left, right = st.columns(2)

with left:

    st.subheader("📚 Faktor Akademik")

    age = st.slider(
        "Usia",
        15,
        25,
        17
    )

    study_time = st.slider(
        "Jam Belajar per Minggu",
        0,
        40,
        10
    )

    absences = st.slider(
        "Jumlah Ketidakhadiran",
        0,
        30,
        5
    )

    parental_support = st.slider(
        "Dukungan Orang Tua",
        0,
        4,
        2
    )

with right:

    st.subheader("⭐ Aktivitas Non-Akademik")

    extracurricular = st.checkbox(
        "Ekstrakurikuler"
    )

    sports = st.checkbox(
        "Olahraga"
    )

    music = st.checkbox(
        "Musik"
    )

    volunteering = st.checkbox(
        "Volunteering"
    )

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
    study_time /
    (absences + 1)
)

support_activity = (
    parental_support *
    activity_score
)

academic_engagement = (
    study_time +
    parental_support
) / (absences + 1)

# =====================================
# FEATURE RESULT
# =====================================

st.divider()

st.subheader("📊 Hasil Feature Engineering")

c1, c2, c3, c4 = st.columns(4)

c1.metric(
    "Activity Score",
    activity_score
)

c2.metric(
    "Study Efficiency",
    round(study_efficiency, 2)
)

c3.metric(
    "Support Activity",
    support_activity
)

c4.metric(
    "Academic Engagement",
    round(academic_engagement, 2)
)

# =====================================
# PREDIKSI
# =====================================

if st.button(
    "🔍 Prediksi Prestasi Akademik",
    use_container_width=True
):

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

        st.success(
            "🏆 Prestasi Akademik Sangat Baik"
        )

        st.balloons()

    elif gpa >= 3.00:

        st.info(
            "📘 Prestasi Akademik Baik"
        )

    elif gpa >= 2.00:

        st.warning(
            "📙 Prestasi Akademik Cukup"
        )

    else:

        st.error(
            "📕 Perlu Peningkatan Prestasi Akademik"
        )

# =====================================
# SIDEBAR
# =====================================

st.sidebar.title("📈 Informasi Model")

st.sidebar.success(
    "Random Forest Regression"
)

st.sidebar.write(
    "Target Prediksi : GPA"
)

st.sidebar.write(
    "Jumlah Fitur : 8"
)

st.sidebar.divider()

st.sidebar.subheader(
    "Performa Model"
)

# GANTI DENGAN HASIL DARI COLAB
st.sidebar.metric(
    "R² Score",
    "0.92"
)

st.sidebar.metric(
    "MAE",
    "0.20"
)

st.sidebar.metric(
    "RMSE",
    "0.25"
)

st.sidebar.divider()

st.sidebar.info("""
Judul Proyek:

Prediksi Prestasi Akademik Siswa Berdasarkan Faktor Akademik dan Aktivitas Non-Akademik Menggunakan Metode Supervised Learning
""")
