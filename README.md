# <h1 align="center">PREDIKSI KUALITAS UDARA BERDASARKAN DATA POLUTAN DAN LINGKUNGAN</h1>

<div align="center">
  <img src="https://wadr.org/wp-content/uploads/2024/01/gettyimages-635231222-master1bafd88464714d55bff4aeeea02b5440-scaled.jpg" alt="Gambar Utama" width="500" height="300">
  <p>
    <small>
      Sumber Image : <a href="https://www.google.com/url?sa=i&url=https%3A%2F%2Fwww.who.int%2Fnews%2Fitem%2F04-04-2022-billions-of-people-still-breathe-unhealthy-air-new-who-data&psig=AOvVaw0y7Nq7V3oR3uF7j5O7wP3B&ust=1734925003299000&source=images&cd=vfe&opi=89978449&ved=0CBQQjRxqFwoTCJjQ4bO5uooDFQAAAAAdAAAAABAE">Access Here.....</a>
    </small>
  </p>
</div>

# <h1 align="center">TABLE OF CONTENT</h1>


1. [Deskripsi Project](#-deskripsi-project-)
      - [Latar Belakang](#latar-belakang)
      - [Tujuan Pengembangan](#tujuan-pengembangan)
2. [Sumber Dataset](#-sumber-dataset-)
3. [Preprocessing dan Pemodelan](#-preprocessing-dan-pemodelan-)
      - [Pemilihan Kolom/Atribut](#pemilihan-kolomatribut)
      - [Preprocessing Data](#preprocessing-data)
      - [Pemodelan](#pemodelan)
4. [Langkah Instalasi](#-langkah-instalasi-)
      - [Software Utama](#software-utama)
      - [Dependensi](#dependensi)
      - [Menjalankan Sistem Prediksi](#menjalankan-sistem-prediksi)
      - [Pelatihan Model](#pelatihan-model)
5. [Hasil dan Analisis](#-hasil-dan-analisis-)
      - [Evaluasi Model](#evaluasi-model )
6. [Sistem Sederhana Streamlit](#-sistem-sederhana-streamlit-)
      - [Tampilan](#tampilan)
      - [Link Live Demo](#link-live-demo)
7. [Biodata](#-biodata-)  

---


<h1 align="center">📚 Deskripsi Project 📚</h1>
---

# Deskripsi Project

Proyek ini bertujuan untuk **mengembangkan sistem prediksi kualitas udara** berdasarkan data polutan dan faktor lingkungan menggunakan model pembelajaran mesin. Sistem ini memprediksi konsentrasi polutan (seperti CO, NO2, C6H6) dan mengklasifikasikannya ke dalam kategori AQI (Air Quality Index) seperti Good, Moderate, Unhealthy. Prediksi mencakup mode single polutan, multi polutan, dan time-series 24 jam ke depan.
---

### Latar Belakang

Kualitas udara dipengaruhi oleh berbagai faktor, termasuk:
- Konsentrasi polutan seperti CO(GT), NO2(GT), C6H6(GT).
- Kondisi lingkungan seperti suhu (T), kelembaban relatif (RH), kelembaban absolut (AH).
- Faktor waktu seperti jam (Hour), hari (Day), bulan (Month).

Proyek ini menggunakan data tabular dari sensor untuk memprediksi level polutan dan AQI, membantu monitoring lingkungan dan rekomendasi kesehatan.

Tujuan utama dari prediksi ini adalah untuk memprediksi status kualitas udara menjadi salah satu dari empat kategori berikut:

#### Kategori Status AQI:

* **Good** : Kualitas udara baik (AQI <=50).
* **Moderate** : Kualitas udara sedang (AQI 51-100).
* **Unhealthy for Sensitive Groups** : Tidak sehat untuk kelompok sensitif (AQI 101-150).
* **Unhealthy** : Tidak sehat (AQI >150).

---

### Tujuan Pengembangan

1. **Membangun Model Prediksi** untuk konsentrasi polutan dan klasifikasi AQI menggunakan Neural Network base dan pretrained.
2. **Evaluasi Performa Model** : Membandingkan model base (MLP) dengan pretrained (Autoencoder + Regressor, SimpleFTTransformer) melalui metrik regresi (MAE, RMSE) dan klasifikasi (accuracy, precision, recall, F1-score via AQI classes).
3. **Membangun Sistem dengan Streamlit** : Aplikasi web sederhana untuk input data sensor dan tampil hasil prediksi secara lokal.

---

# Sumber Dataset

Dataset yang digunakan dalam proyek ini berasal dari sumber utama:

### 1. **UCI Machine Learning Repository**

- **Judul Dataset** : *Air Quality Dataset (AirQualityUCI.csv)*
- **Link** : [UCI - Air Quality Dataset](https://archive.ics.uci.edu/dataset/360/air+quality)
- **Deskripsi** : Dataset ini berisi 9357 data pengukuran kualitas udara dari sensor di Italia (2004-2005), mencakup polutan seperti CO, NO2, C6H6, dan faktor lingkungan. Total fitur: 15 kolom numerik.

Dataset minimal >5000 data (9357 setelah preprocess), tanpa augmentasi karena sudah cukup.

---

# Preprocessing dan Pemodelan

### Pemilihan Kolom/Atribut

Atribut yang digunakan mencakup polutan dan faktor waktu/lingkungan. Kolom utama:

| Kolom | Tipe | Deskripsi |
|-------|------|-----------|
| **CO(GT)** | Continuous | Konsentrasi CO (mg/m³) - target single/multi. |
| **NO2(GT)** | Continuous | Konsentrasi NO2 (µg/m³) - target multi. |
| **C6H6(GT)** | Continuous | Konsentrasi Benzene (µg/m³) - target multi. |
| **PT08.S1(CO)** | Continuous | Sensor CO. |
| **NMHC(GT)** | Continuous | Non-Methane Hydrocarbons. |
| **PT08.S2(NMHC)** | Continuous | Sensor NMHC. |
| **NOx(GT)** | Continuous | Nitrogen Oxides. |
| **PT08.S3(NOx)** | Continuous | Sensor NOx. |
| **PT08.S4(NO2)** | Continuous | Sensor NO2. |
| **PT08.S5(O3)** | Continuous | Sensor O3. |
| **T** | Continuous | Suhu (°C). |
| **RH** | Continuous | Kelembaban Relatif (%). |
| **AH** | Continuous | Kelembaban Absolut. |
| **Hour** | Integer | Jam pengukuran. |
| **Day** | Integer | Hari. |
| **Month** | Integer | Bulan. |
| **Year** | Integer | Tahun. |

Target: Prediksi numerik polutan, lalu derivasi ke AQI classes (Good, Moderate, Unhealthy).

### Preprocessing Data

1. **Pembersihan Data**:
   - Ganti koma dengan titik untuk numerik.
   - Ganti -200 dengan NaN, isi dengan mean.
   - Parsing datetime dari Date + Time, tambah fitur Hour, Day, Month, Year.
   - Drop kolom Date, Time, Datetime.

2. **Scaling** : StandardScaler untuk normalisasi fitur.
3. **Pembagian Data** : 80% train, 20% test (dengan val 20% dari train). Untuk time-series: Sequence 24 jam.
4. **Fungsi AQI** : Hitung sub-index untuk CO, NO2, C6H6 berdasarkan breakpoints, lalu max sebagai AQI dan kategori.

Analisis awal: Distribusi CO, top 10 worst CO, rata-rata CO per bulan/jam.

---

### Pemodelan

Model yang digunakan (3 wajib + variasi):

1. **MLP (Base Non-Pretrained)**: Feedforward Neural Network (MLP) dengan layer 128-64-1/3, ReLU, Dropout. Dilatih from scratch untuk single/multi polutan.
2. **Autoencoder + Regressor (Pretrained 1)**: Autoencoder dilatih dulu sebagai embedding pretrained, lalu fine-tune regressor (layer 32-1).
3. **SimpleFTTransformer (Pretrained 2)**: Adaptasi FT-Transformer dengan embedding, transformer layers (2 layers, 2 heads), fine-tune untuk regresi.
4. **LSTM (Variasi untuk Time-Series)**: RNN dengan 2 layers LSTM, untuk prediksi 24 jam ke depan.

Parameter: Epoch 50 (early stop patience 5), batch 64, Adam optimizer (lr=0.001, weight decay=1e-5), MSELoss.

---

# Langkah Instalasi

### Software Utama

Proyek ini dapat dijalankan menggunakan Google Colab dan VSCode. Pastikan Python 3.12+ telah terinstal di sistem Anda.

### Dependensi

Dependensi yang diperlukan untuk menjalankan proyek ini telah disediakan dalam file requirements.txt di direktori ini. Anda dapat menginstal seluruh dependensi dengan salah satu cara berikut:

#### Cara 1: Instalasi Langsung

Jalankan perintah berikut di terminal:

```
pip install -r requirements.txt
```

#### Cara 2: Instalasi Manual

Anda juga dapat menginstal dependensi satu per satu menggunakan perintah seperti berikut:

```
pip install torch==2.0.0
```

> Lakukan hal yang sama untuk semua dependensi yang tersedia di requirements.txt

### Menjalankan Sistem Prediksi

Untuk menjalankan sistem prediksi, buka terminal dan jalankan file app.py dengan perintah berikut:

```
streamlit run app.py
```

> Jika anda ingin langsung melihat penggunaan Sistem Prediksi dari project ini, Lihat bagian Link Live Demo

### Pelatihan Model

Model yang telah dilatih tersedia di direktori Model. Namun dalam hal ini, untuk beberapa model tidak dapat diupload ke direktori github karena keterbatasan ukuran file. Berikut disediakan model dalam bentuk link Google Drive jika diperlukan.

> Google Drive - Model Save Pelatihan (klik)

Jika Anda ingin melatih model dari awal, jalankan file training_models.ipynb yang tersedia di direktori ini menggunakan Google Colab atau Jupyter.

---

# Hasil dan Analisis

### Evaluasi Model

Model dievaluasi menggunakan beberapa metrik, termasuk MAE/RMSE (regresi) dan classification report/CM/accuracy via AQI classes (Good, Moderate, Unhealthy). Grafik: Loss train/val (PNG: MLP_Single_plot.png, dll.).

#### Classification Report

Berikut adalah penjelasan tentang metrik yang digunakan dalam classification report:

* **Precision** : Mengukur proporsi prediksi positif yang benar.
* **Recall** : Mengukur proporsi sampel aktual positif yang berhasil diidentifikasi dengan benar.
* **F1-Score** : Rata-rata harmonis dari precision dan recall.
* **Accuracy** : Mengukur keseluruhan performa model.

#### Tabel Perbandingan Classification Report

Berikut adalah perbandingan metrik evaluasi untuk setiap model (adaptasi ke AQI classes):

| Nama Model | Akurasi (AQI Classes) | MAE | RMSE | Hasil Analisis |
|------------|-----------------------|-----|------|---------------|
| MLP_Single (Base) | 0.97 | 0.257 | 0.378 | Performa baik pada kelas Good (precision 0.98), tapi rendah pada minor classes (F1 Unhealthy 0.40); overfit minor, cocok untuk prediksi dasar. |
| AE_Reg (Pretrained 1) | 0.97 | 0.290 | 0.409 | Mirip base, embedding pretrained bantu generalisasi (F1 Moderate 0.78); lebih stabil pada data test. |
| SimpleFTTransformer (Pretrained 2) | 0.97 | 0.276 | 0.406 | Tangkap relasi kompleks (accuracy tinggi), tapi precision rendah pada rare classes; efisien untuk tabular. |
| LSTM_TimeSeries (Variasi) | 0.97 | 0.235 | 0.336 | Bagus untuk prediksi 24 jam (F1 Moderate 0.76); tangkap pola waktu, worst AQI 116.67. |

Contoh Classification Report (MLP_Single AQI Classes):

                                precision    recall  f1-score   support

                          Good       0.98      0.99      0.99      1742
                      Moderate       0.88      0.68      0.77       126
Unhealthy for Sensitive Groups       1.00      0.25      0.40         4

                      accuracy                           0.97      1872
                     macro avg       0.95      0.64      0.72      1872
                  weighted avg       0.97      0.97      0.97      1872

Confusion Matrix: [[1733 9 0] [40 86 0] [0 3 1]]

Analisis: Pretrained model lebih baik dalam generalisasi (RMSE lebih rendah), base MLP overfit pada data train. LSTM untuk time-series: Accuracy 0.97 pada 24 jam pred (max AQI 116.67, rata 24.08).

Grafik Loss: ![MLP_Single Loss](images/MLP_Single_plot.png)

---

# Sistem Sederhana Streamlit

Aplikasi Streamlit untuk prediksi kualitas udara dengan input sensor/lingkungan.

### Tampilan

- **Sidebar**: Pilih mode (single/multi/time-series), info dashboard.
- **Main**: Input data, prediksi AQI, visual (gauge chart, radar, line chart untuk 24 jam), rekomendasi kesehatan.
- UI: CSS custom dengan gradient biru, Plotly charts.

### Link Live Demo

[Demo Dashboard Kualitas Udara](https://your-streamlit-app-link.streamlit.app/)  (Optional: Deploy ke streamlit.io)

---

# Biodata

👤 **Nama** : Muhammad Imam Aditya Ismawan  
📘 **NIM** : 202210370311093
🎓 **Program Studi** : Teknik Informatika  
🏛️ **Universitas** : Universitas Muhammadiyah Malang  

---

## About

Repository ini dibuat untuk memenuhi Tugas Akhir Praktikum (UAP) Mata Kuliah Pembelajaran Mesin. Berisi kode, model, dan dokumentasi pengembangan sistem prediksi kualitas udara. Tidak ada kemiripan dengan praktikan lain.  
**Version: 1.0**  
**December 2025**
