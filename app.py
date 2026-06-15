import streamlit as st
import pickle
import numpy as np

# Load model dan scaler
model = pickle.load(open('model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

st.title("Prediksi Prestasi Akademik Siswa")

st.write("Masukkan data siswa untuk memprediksi GPA")

# INPUT TANPA DESIMAL

age = st.number_input(
    "Usia",
    min_value=15,
    max_value=25,
    value=17,
    step=1,
    format="%d"
)

study_time = st.number_input(
    "Jam Belajar per Minggu",
    min_value=0,
    max_value=50,
    value=10,
    step=1,
    format="%d"
)

absences = st.number_input(
    "Jumlah Ketidakhadiran",
    min_value=0,
    max_value=50,
    value=5,
    step=1,
    format="%d"
)

parental_support = st.number_input(
    "Dukungan Orang Tua (0-4)",
    min_value=0,
    max_value=4,
    value=2,
    step=1,
    format="%d"
)

activity_score = st.number_input(
    "Activity Score",
    min_value=0,
    max_value=10,
    value=2,
    step=1,
    format="%d"
)

academic_engagement = st.number_input(
    "Academic Engagement",
    min_value=0,
    max_value=50,
    value=5,
    step=1,
    format="%d"
)

# Prediksi
if st.button("Prediksi GPA"):

    data = np.array([
        age,
        study_time,
        absences,
        parental_support,
        activity_score,
        academic_engagement
    ]).reshape(1, -1)

    data = scaler.transform(data)

    prediksi = model.predict(data)

    st.success(
        f"Prediksi GPA: {prediksi[0]:.2f}"
    )
