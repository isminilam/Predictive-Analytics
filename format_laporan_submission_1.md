# Laporan Proyek Machine Learning - Ismi Nilam Anggraini

## Domain Proyek

Diabetes melitus merupakan salah satu penyakit tidak menular yang prevalensinya terus meningkat di seluruh dunia. Berdasarkan data dari International Diabetes Federation (IDF) tahun 2021, Indonesia menempati peringkat kelima sebagai negara dengan jumlah penderita diabetes terbanyak di dunia. Pada tahun 2024, tercatat bahwa Indonesia memiliki penderita diabetes sebanyak 20,4 juta jiwa dengan rentang usia 20–79 tahun. Menurut Kementerian Kesehatan Republik Indonesia (Kemenkes RI), jumlah ini diperkirakan akan terus meningkat dan mencapai 28,6 juta jiwa pada tahun 2045.

Tingginya angka ini diperburuk oleh banyaknya kasus yang tidak terdiagnosis sejak dini, terutama akibat gejala awal yang sering tidak disadari dan keterbatasan akses terhadap layanan kesehatan. World Health Organization (WHO) menyatakan bahwa diabetes dapat menyebabkan berbagai komplikasi serius seperti kebutaan, gagal ginjal, serangan jantung, stroke, hingga amputasi anggota tubuh bagian bawah.

Deteksi dini sangat penting untuk mencegah komplikasi lebih lanjut. Teknologi machine learning menjadi solusi prediksi risiko diabetes yang cepat, efisien, serta mudah diakses oleh masyarakat luas. Memanfaatkan data klinis seperti usia, indeks massa tubuh (BMI), tekanan darah, kadar glukosa darah, riwayat merokok, dan beberapa variabel relevan lainnya yang dapat membuat model prediktif berbasis machine learning untuk membantu tenaga medis maupun individu dalam mengenali risiko diabetes secara mandiri.
Berbagai penelitian telah menunjukkan bahwa algoritma machine learning seperti XGBoost dan Random Forest dapat menghasilkan model prediksi diabetes dengan akurasi tinggi.


**Referensi**:
- 

## Business Understanding

### Problem Statements
- Bagaimana memprediksi risiko diabetes pada individu berdasarkan data kesehatan klinis secara tepat dan efisien?
- Algoritma machine learning mana yang memberikan performa terbaik dalam memprediksi risiko diabetes?

### Goals

- Mengembangkan model prediksi risiko diabetes yang tepat dan efisien
- Membandingkan performa algoritma Random Forest dan XGBoost

### Solution statements
- Membangun model prediksi diabetes berdasarkan data klinis menggunakan dua algoritma machine learning, yaitu Random Forest dan XGBoost.
  Pendekatan ini dilakukan untuk menjawab pertanyaan tentang bagaimana memprediksi risiko diabetes secara tepat dan efisien dengan memanfaatkan data seperti usia, tekanan darah, indeks massa tubuh, kadar glukosa darah, riwayat merokok, dan lain-lain.

- Melakukan evaluasi dan perbandingan performa ketiga algoritma menggunakan metrik evaluasi yang relevan. Evaluasi dilakukan menggunakan metrik accuracy, precision, recall, dan F1-score. Precision dan recall digunakan untuk menilai kemampuan model dalam mendeteksi individu berisiko, F1-score sebagai keseimbangan keduanya, serta accuracy untuk mengukur performa keseluruhan. Hasil evaluasi ini akan digunakan untuk menentukan algoritma dengan performa terbaik.


## Data Understanding
Dataset yang digunakan dalam proyek ini diambil dari situs . Dataset ini berisi data kesehatan klinis yang digunakan untuk memprediksi kemungkinan seseorang menderita diabetes. Dataset terdiri dari 100.000 baris dengan berbagai fitur yang merepresentasikan informasi medis dasar dan gaya hidup individu. Data ini sangat relevan karena mencakup fitur-fitur yang umum digunakan dalam proses skrining diabetes seperti usia, tekanan darah, indeks massa tubuh (BMI), kadar glukosa, hingga status merokok. 

### Variabel-variabel pada Diabetes Prediction Dataset adalah sebagai berikut:

| **Nama Variabel**     | **Deskripsi**                                                        | **Tipe Data**     |
| --------------------- | -------------------------------------------------------------------- | ----------------- |
| `gender`              | Jenis kelamin individu                                               | Categorical       |
| `age`                 | Usia individu dalam tahun                                            | Numerical (Float) |
| `hypertension`        | Status hipertensi                                                    | Numerical (Int)   |
| `heart_disease`       | Status penyakit jantung                                              | Numerical (Int)   |
| `smoking_history`     | Riwayat merokok                                                      | Categorical       |
| `bmi`                 | Indeks massa tubuh                                                   | Numerical (Float) |
| `HbA1c_level`         | Tingkat HbA1c (hemoglobin terglikasi)                                | Numerical (Float) |
| `blood_glucose_level` | Kadar glukosa darah                                                  | Numerical (Int)   |
| `diabetes`            | Label target                                                         | Numerical (Int)   |

### Informasi Awal Dataset:
- Terdiri dari 9 kolom dengan 100.000 baris
- Tidak memiliki missing value
- Terdapat outlier pada kolom age = 0.08 dan bmi = 95.69
- Terdapat data duplikat sebanyak 3854 baris

### Exploratory Data Analysis (EDA)

1. Distribusi Kelas Target
   
   ![image](https://github.com/user-attachments/assets/9e35c752-d090-4424-ab94-84908055f849)

2. Distribusi Fitur Numerik

  ![image](https://github.com/user-attachments/assets/ef1abf06-ee26-44f5-b54a-48a5fac4d5d7)

**Insight:**

* `age`: Distribusi cukup merata dengan puncak di sekitar usia 70-80 tahun. Terdapat outlier ekstrem dengan usia di bawah 1 tahun.
* `hypertension` dan `heart_disease`: Datanya tidak seimbang dengan mayoritas 0 (tidak memiliki riwayat).
* `bmi`: Distribusi positively skewed (condong ke kanan), dengan satu puncak tajam.
* `HbA1c_level`: Distribusi cenderung normal dengan puncak di sekitar 6.0–6.5.
* `blood_glucose_level`: Distribusi tidak normal, dengan lonjakan pada kisaran 140–160.
* `diabetes`: Sangat imbalanced, kelas 0 (non-diabetes) jauh lebih banyak dari kelas 1 (diabetes).

3. Distribusi Fitur Kategorikal

   ![image](https://github.com/user-attachments/assets/968b33ce-c974-4ba2-8099-e928ae74bc30)
   ![image](https://github.com/user-attachments/assets/43f2de11-db8d-4e14-bcd8-22a3aad853a1)

   **Insight:**
   Berdasarkan visualisasi tersebut, jumlah female lebih banyak dibanding male pada dataset ini. Selain itu, pada distribusi riwayat merokok yaitu No Info dan never memiliki jumlah tertinggi.

4. Korelasi Antar Fitur

   ![image](https://github.com/user-attachments/assets/a88562e4-585a-48a6-a43a-2674173a5c6e)

   **Insight:**
Berdasarkan heatmap korelasi fitur numerik terhadap target `diabetes`, dapat diamati bahwa fitur `blood_glucose_level` dan `HbA1c_level` memiliki korelasi tertinggi yaitu 0.42 dan 0.40. Selain itu, beberapa fitur lain seperti `age`, `bmi`, `hypertension`, dan `heart_disease` punya kontribusi yang lebih kecil, tapi tetap informatif.



## Data Preparation
Pada bagian ini Anda menerapkan dan menyebutkan teknik data preparation yang dilakukan. Teknik yang digunakan pada notebook dan laporan harus berurutan.

**Rubrik/Kriteria Tambahan (Opsional)**: 
- Menjelaskan proses data preparation yang dilakukan
- Menjelaskan alasan mengapa diperlukan tahapan data preparation tersebut.

## Modeling
Tahapan ini membahas mengenai model machine learning yang digunakan untuk menyelesaikan permasalahan. Anda perlu menjelaskan tahapan dan parameter yang digunakan pada proses pemodelan.

**Rubrik/Kriteria Tambahan (Opsional)**: 
- Menjelaskan kelebihan dan kekurangan dari setiap algoritma yang digunakan.
- Jika menggunakan satu algoritma pada solution statement, lakukan proses improvement terhadap model dengan hyperparameter tuning. **Jelaskan proses improvement yang dilakukan**.
- Jika menggunakan dua atau lebih algoritma pada solution statement, maka pilih model terbaik sebagai solusi. **Jelaskan mengapa memilih model tersebut sebagai model terbaik**.

## Evaluation
Pada bagian ini anda perlu menyebutkan metrik evaluasi yang digunakan. Lalu anda perlu menjelaskan hasil proyek berdasarkan metrik evaluasi yang digunakan.

Sebagai contoh, Anda memiih kasus klasifikasi dan menggunakan metrik **akurasi, precision, recall, dan F1 score**. Jelaskan mengenai beberapa hal berikut:
- Penjelasan mengenai metrik yang digunakan
- Menjelaskan hasil proyek berdasarkan metrik evaluasi

Ingatlah, metrik evaluasi yang digunakan harus sesuai dengan konteks data, problem statement, dan solusi yang diinginkan.

**Rubrik/Kriteria Tambahan (Opsional)**: 
- Menjelaskan formula metrik dan bagaimana metrik tersebut bekerja.

**---Ini adalah bagian akhir laporan---**

_Catatan:_
- _Anda dapat menambahkan gambar, kode, atau tabel ke dalam laporan jika diperlukan. Temukan caranya pada contoh dokumen markdown di situs editor [Dillinger](https://dillinger.io/), [Github Guides: Mastering markdown](https://guides.github.com/features/mastering-markdown/), atau sumber lain di internet. Semangat!_
- Jika terdapat penjelasan yang harus menyertakan code snippet, tuliskan dengan sewajarnya. Tidak perlu menuliskan keseluruhan kode project, cukup bagian yang ingin dijelaskan saja.

