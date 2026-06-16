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
# LOAD MODEL
# =====================================

model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

# =====================================
# HEADER
# =====================================

st.title("🎓 Prediksi Prestasi Akademik Siswa")

st.markdown("""
### Berdasarkan Faktor Akademik dan Aktivitas Non-Akademik Menggunakan Metode Supervised Learning

Aplikasi ini digunakan untuk memprediksi GPA siswa berdasarkan karakteristik akademik dan aktivitas pendukung.
""")

st.divider()

# =====================================
# INPUT DATA
# =====================================

col1, col2 = st.columns(2)

with col1:

    st.subheader("📚 Faktor Akademik")

    age = st.slider(
        "Usia Siswa",
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

    st.subheader("⭐ Aktivitas Non-Akademik")

    extracurricular = st.checkbox("Mengikuti Ekstrakurikuler")

    sports = st.checkbox("Mengikuti Kegiatan Olahraga")

    music = st.checkbox("Mengikuti Kegiatan Musik")

    volunteering = st.checkbox("Mengikuti Kegiatan Sosial")

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
# HASIL FEATURE ENGINEERING
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
    f"{study_efficiency:.2f}"
)

c3.metric(
    "Support Activity",
    support_activity
)

c4.metric(
    "Academic Engagement",
    f"{academic_engagement:.2f}"
)

# =====================================
# PREDIKSI
# =====================================

if st.button(
    "🔍 Prediksi GPA",
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

    prediksi = model.predict(data)

    gpa = float(prediksi[0])

    # Membatasi GPA
    gpa = max(0.0, min(4.0, gpa))

    st.divider()

    st.subheader("🎯 Hasil Prediksi")

    st.metric(
        label="Prediksi GPA",
        value=f"{gpa:.2f}"
    )

    st.progress(
        min(gpa / 4.0, 1.0)
    )

    st.write(
        f"Persentase terhadap GPA maksimum: {(gpa/4)*100:.1f}%"
    )

    if gpa >= 3.50:

        st.success(
            "🏆 Prestasi Sangat Baik"
        )

        st.balloons()

    elif gpa >= 3.00:

        st.info(
            "📘 Prestasi Baik"
        )

    elif gpa >= 2.00:

        st.warning(
            "📙 Prestasi Cukup"
        )

    else:

        st.error(
            "📕 Perlu Pendampingan Belajar"
        )

# =====================================
# SIDEBAR
# =====================================

st.sidebar.header("📈 Informasi Model")

st.sidebar.success(
    "Model: Random Forest Regression"
)

st.sidebar.write(
    "Target Prediksi: GPA"
)

st.sidebar.write(
    "Jumlah Fitur: 8"
)

st.sidebar.divider()

st.sidebar.subheader(
    "Performa Model"
)

# GANTI DENGAN HASIL DARI COLAB
st.sidebar.metric(
    "R² Score",
    "ISI_R2"
)

st.sidebar.metric(
    "MAE",
    "ISI_MAE"
)

st.sidebar.metric(
    "RMSE",
    "ISI_RMSE"
)

st.sidebar.divider()

st.sidebar.info(
    "Dibangun menggunakan Python, Scikit-Learn, Streamlit, dan Random Forest Regression."
)
