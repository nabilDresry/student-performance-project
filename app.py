import streamlit as st
import pickle
import numpy as np

model = pickle.load(open('model.pkl', 'rb'))

st.title("Prediksi Prestasi Akademik Siswa")

age = st.number_input("Usia")
study_time = st.number_input("Jam Belajar per Minggu")
absences = st.number_input("Jumlah Ketidakhadiran")
parental_support = st.number_input("Dukungan Orang Tua")
activity_score = st.number_input("Activity Score")
academic_engagement = st.number_input("Academic Engagement")

if st.button("Prediksi GPA"):

    data = np.array([
        age,
        study_time,
        absences,
        parental_support,
        activity_score,
        academic_engagement
    ]).reshape(1,-1)

    prediksi = model.predict(data)

    st.success(
        f"Prediksi GPA: {prediksi[0]:.2f}"
    )
