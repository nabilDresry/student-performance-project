import streamlit as st
import pickle
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
# CSS CUSTOM
# =====================================

st.markdown("""
<style>

.main {
    padding-top: 1rem;
}

.stButton > button {
    width: 100%;
    height: 55px;
    border-radius: 12px;
    font-size: 18px;
    font-weight: bold;
}

div[data-testid="stMetric"] {
    background-color: #1E293B;
    padding: 15px;
    border-radius: 15px;
}

</style>
""", unsafe_allow_html=True)

# =====================================
# HERO BANNER
# =====================================

st.markdown("""
<div style="
background: linear-gradient(135deg,#2563EB,#7C3AED);
padding:30px;
border-radius:20px;
text-align:center;
color:white;
">

<h1>🎓 Prediksi Prestasi Akademik Siswa</h1>

<p>
Berdasarkan Faktor Akademik dan Aktivitas Non-Akademik
Menggunakan Metode Supervised Learning
</p>

</div>
""", unsafe_allow_html=True)

st.write("")

# =====================================
# LOAD MODEL
# =====================================

try:

    with open("model.pkl", "rb") as f:
        model = pickle.load(f)

    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)

except Exception as e:

    st.error(f"Gagal memuat model: {e}")
    st.stop()

# =====================================
# SIDEBAR
# =====================================

st.sidebar.title("📊 Informasi Model")

st.sidebar.success(
    "Random Forest Regressor"
)

# GANTI SESUAI HASIL COLAB
st.sidebar.metric(
    "R² Score",
    "0.92"
)

st.sidebar.metric(
    "MAE",
    "0.11"
)

st.sidebar.metric(
    "RMSE",
    "0.18"
)

st.sidebar.markdown("---")

st.sidebar.info("""
Proyek AI & Big Data

Prediksi Prestasi Akademik Siswa
""")

# =====================================
# INPUT
# =====================================

st.subheader("📚 Input Data Siswa")

col1, col2 = st.columns(2)

with col1:

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

with col2:

    extracurricular = st.checkbox(
        "Mengikuti Ekstrakurikuler"
    )

    sports = st.checkbox(
        "Mengikuti Olahraga"
    )

    music = st.checkbox(
        "Mengikuti Musik"
    )

    volunteering = st.checkbox(
        "Mengikuti Volunteering"
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
# PREVIEW FITUR
# =====================================

st.markdown("---")

st.subheader("📈 Hasil Feature Engineering")

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

if st.button("🔍 Prediksi Prestasi Akademik"):

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

    if gpa < 0:
        gpa = 0

    if gpa > 4:
        gpa = 4

    st.markdown("---")

    st.markdown(f"""
    <div style="
    background:linear-gradient(135deg,#10B981,#059669);
    padding:25px;
    border-radius:20px;
    text-align:center;
    color:white;
    ">

    <h3>🎯 Prediksi GPA</h3>

    <h1>{gpa:.2f}</h1>

    </div>
    """, unsafe_allow_html=True)

    st.progress(gpa / 4)

    if gpa >= 3.5:

        st.success(
            "🏆 Prestasi Akademik Sangat Baik"
        )

        st.balloons()

    elif gpa >= 3.0:

        st.info(
            "📘 Prestasi Akademik Baik"
        )

    elif gpa >= 2.0:

        st.warning(
            "📙 Prestasi Akademik Cukup"
        )

    else:

        st.error(
            "📕 Perlu Peningkatan Prestasi Akademik"
        )

# =====================================
# FOOTER
# =====================================

st.markdown("---")

st.markdown("""
<center>

<b>Universitas Halu Oleo</b><br>
Mata Kuliah AI dan Big Data<br>
Prediksi Prestasi Akademik Siswa

</center>
""", unsafe_allow_html=True)
