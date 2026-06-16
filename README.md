# Prediksi Prestasi Akademik Siswa Berdasarkan Faktor Akademik dan Aktivitas Non-Akademik Menggunakan Metode Supervised Learning

## Deskripsi Proyek

Proyek ini bertujuan untuk memprediksi prestasi akademik siswa yang direpresentasikan melalui nilai GPA (Grade Point Average) berdasarkan faktor akademik dan aktivitas non-akademik. Prediksi dilakukan menggunakan metode Machine Learning Supervised Learning dengan algoritma Random Forest Regressor.

Aplikasi dikembangkan dalam bentuk website interaktif menggunakan Streamlit sehingga pengguna dapat memasukkan data siswa dan memperoleh hasil prediksi secara langsung.

---

## Latar Belakang

Prestasi akademik siswa dipengaruhi oleh berbagai faktor, baik faktor akademik maupun faktor non-akademik. Faktor seperti waktu belajar, tingkat kehadiran, dukungan orang tua, dan keterlibatan dalam aktivitas sekolah dapat memberikan dampak terhadap hasil belajar siswa.

Dengan memanfaatkan Machine Learning, hubungan antara berbagai faktor tersebut dapat dipelajari untuk menghasilkan model prediksi yang membantu dalam memahami pola prestasi akademik siswa.

---

## Tujuan Proyek

Tujuan dari proyek ini adalah:

1. Menganalisis faktor-faktor yang memengaruhi prestasi akademik siswa.
2. Membangun model prediksi menggunakan metode Supervised Learning.
3. Mengimplementasikan model ke dalam aplikasi web interaktif.
4. Menyediakan sarana prediksi prestasi akademik yang mudah digunakan.

---

## Dataset

Dataset yang digunakan merupakan Student Performance Dataset yang berisi informasi mengenai karakteristik siswa, aktivitas akademik, dan aktivitas non-akademik.

Beberapa atribut yang digunakan dalam penelitian ini antara lain:

- Age
- Study Time Weekly
- Absences
- Parental Support
- Extracurricular Activities
- Sports Participation
- Music Participation
- Volunteering Activities
- GPA

Dataset telah melalui proses pembersihan data dan feature engineering sebelum digunakan dalam proses pelatihan model.

---

## Feature Engineering

Untuk meningkatkan kualitas prediksi, beberapa fitur baru dibuat berdasarkan kombinasi fitur asli.

### Activity Score

Merepresentasikan jumlah aktivitas non-akademik yang diikuti siswa.

### Study Efficiency

Menggambarkan efisiensi belajar berdasarkan perbandingan antara waktu belajar dan jumlah ketidakhadiran.

### Support Activity

Mengukur interaksi antara dukungan orang tua dan tingkat aktivitas siswa.

### Academic Engagement

Menggambarkan tingkat keterlibatan akademik siswa berdasarkan kombinasi waktu belajar, dukungan orang tua, dan absensi.

---

## Metode yang Digunakan

Model Machine Learning yang digunakan dalam proyek ini adalah:

### Random Forest Regressor

Alasan pemilihan algoritma:

- Mampu menangani hubungan non-linear pada data.
- Memiliki performa yang baik untuk tugas regresi.
- Mengurangi risiko overfitting melalui mekanisme ensemble.
- Cocok digunakan pada dataset dengan banyak variabel.

---

## Tahapan Pengembangan

1. Data Collection
2. Data Cleaning
3. Exploratory Data Analysis (EDA)
4. Feature Engineering
5. Data Splitting
6. Data Scaling
7. Model Training
8. Model Evaluation
9. Model Deployment menggunakan Streamlit

---

## Evaluasi Model

Model dievaluasi menggunakan beberapa metrik regresi, yaitu:

### R² Score

Mengukur kemampuan model dalam menjelaskan variasi data target.

### MAE (Mean Absolute Error)

Mengukur rata-rata kesalahan absolut prediksi.

### RMSE (Root Mean Squared Error)

Mengukur besar kesalahan prediksi dengan memberikan penalti lebih besar terhadap error yang tinggi.

### Hasil Evaluasi

| Metrik | Nilai |
|---------|---------|
| R² Score | 0.92 |
| MAE | 0.20 |
| RMSE | 0.25 |

*Sesuaikan nilai di atas dengan hasil evaluasi model yang diperoleh.*

---

## Fitur Aplikasi

Aplikasi yang dikembangkan memiliki fitur sebagai berikut:

- Input data siswa secara interaktif.
- Prediksi GPA secara otomatis.
- Menampilkan hasil prediksi secara real-time.
- Menampilkan informasi performa model.
- Antarmuka berbasis web menggunakan Streamlit.

---

## Struktur Repository

```text
student-performance-project
│
├── app.py
├── model.pkl
├── scaler.pkl
├── Student_performance_data_.csv
├── requirements.txt
├── README.md
└── .streamlit
    └── config.toml
```

---

## Cara Menjalankan Aplikasi

### Clone Repository

```bash
git clone https://github.com/USERNAME/student-performance-project.git
```

### Masuk ke Folder Proyek

```bash
cd student-performance-project
```

### Install Dependency

```bash
pip install -r requirements.txt
```

### Jalankan Streamlit

```bash
streamlit run app.py
```

---

## Teknologi yang Digunakan

- Python
- Pandas
- NumPy
- Scikit-Learn
- Streamlit
- Pickle
- GitHub

---

## Informasi Pengembang

Nama : La Ode Muhammad Nabil Ziyad

NIM : 101012400339

Program Studi : Teknik Telekomunikasi

Mata Kuliah : AI dan Big Data

Universitas : Telkom University

---

## Deployment

GitHub Repository:

[Tambahkan Link Repository GitHub]

Streamlit Application:

[Tambahkan Link Streamlit]

---

## Kesimpulan

Berdasarkan hasil pengujian, model Random Forest Regressor mampu memberikan performa yang baik dalam memprediksi prestasi akademik siswa. Faktor akademik seperti waktu belajar dan absensi, serta faktor non-akademik seperti aktivitas siswa dan dukungan orang tua, terbukti memiliki pengaruh terhadap hasil prediksi.

Implementasi model ke dalam aplikasi berbasis Streamlit memungkinkan proses prediksi dilakukan secara mudah, cepat, dan interaktif sehingga dapat digunakan sebagai media pembelajaran penerapan Machine Learning dalam bidang pendidikan.
