import streamlit as st
import pickle
import numpy as np

st.set_page_config(
    page_title="Prediksi Prestasi Akademik Siswa",
    page_icon="🎓",
    layout="wide"
)

try:
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)

    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)

except Exception as e:
    st.error(f"Gagal memuat model: {e}")
    st.stop()

st.markdown("""
# 🎓 Prediksi Prestasi Akademik Siswa

Prediksi GPA berdasarkan faktor akademik dan aktivitas non-akademik menggunakan Random Forest Regression.
""")

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

st.subheader("Hasil Feature Engineering")

st.write(
    {
        "Activity Score": activity_score,
        "Study Efficiency": round(study_efficiency, 2),
        "Support Activity": support_activity,
        "Academic Engagement": round(
            academic_engagement,
            2
        )
    }
)

if st.button("Prediksi GPA"):

    fitur = np.array([
        age,
        study_time,
        absences,
        parental_support,
        activity_score,
        study_efficiency,
        support_activity,
        academic_engagement
    ]).reshape(1, -1)

    fitur = scaler.transform(fitur)

    hasil = model.predict(fitur)

    gpa = float(hasil[0])

    if gpa < 0:
        gpa = 0

    if gpa > 4:
        gpa = 4

    st.success(
        f"Prediksi GPA: {gpa:.2f}"
    )

    if gpa >= 3.5:
        st.success(
            "🏆 Prestasi Sangat Baik"
        )

    elif gpa >= 3:
        st.info(
            "📘 Prestasi Baik"
        )

    elif gpa >= 2:
        st.warning(
            "📙 Prestasi Cukup"
        )

    else:
        st.error(
            "📕 Perlu Pendampingan Belajar"
        )

st.sidebar.title("Informasi Model")

st.sidebar.write(
    "Model: Random Forest Regressor"
)

st.sidebar.write(
    "Jumlah Fitur: 8"
)

st.sidebar.write(
    "Target: GPA"
)
