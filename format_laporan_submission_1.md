# Laporan Proyek Machine Learning - Ismi Nilam Anggraini

## Domain Proyek

Diabetes melitus merupakan salah satu penyakit tidak menular yang prevalensinya terus meningkat di seluruh dunia. Berdasarkan data dari International Diabetes Federation (IDF) tahun 2021, Indonesia menempati peringkat kelima sebagai negara dengan jumlah penderita diabetes terbanyak di dunia. Pada tahun 2024, tercatat bahwa Indonesia memiliki penderita diabetes sebanyak 20,4 juta jiwa dengan rentang usia 20–79 tahun. Menurut Kementerian Kesehatan Republik Indonesia (Kemenkes RI), jumlah ini diperkirakan akan terus meningkat dan mencapai 28,6 juta jiwa pada tahun 2045.

Tingginya angka ini diperburuk oleh banyaknya kasus yang tidak terdiagnosis sejak dini, terutama akibat gejala awal yang sering tidak disadari dan keterbatasan akses terhadap layanan kesehatan. World Health Organization (WHO) menyatakan bahwa diabetes dapat menyebabkan berbagai komplikasi serius seperti kebutaan, gagal ginjal, serangan jantung, stroke, hingga amputasi anggota tubuh bagian bawah.

Deteksi dini sangat penting untuk mencegah komplikasi lebih lanjut. Teknologi machine learning menjadi solusi prediksi risiko diabetes yang cepat, efisien, serta mudah diakses oleh masyarakat luas. Memanfaatkan data klinis seperti usia, indeks massa tubuh (BMI), tekanan darah, kadar glukosa darah, riwayat merokok, dan beberapa variabel relevan lainnya yang dapat membuat model prediktif berbasis machine learning untuk membantu tenaga medis maupun individu dalam mengenali risiko diabetes secara mandiri.
Berbagai penelitian telah menunjukkan bahwa algoritma machine learning seperti XGBoost dan Random Forest dapat menghasilkan model prediksi diabetes dengan akurasi tinggi.


**Referensi**:
- International Diabetes Federation. (2025). Indonesia diabetes trends & prevalence. IDF Diabetes Atlas. Retrieved May 26, 2025, from https://diabetesatlas.org/data-by-location/country/indonesia/
- Kementerian Kesehatan Republik Indonesia. (2024, Januari 10). Saatnya mengatur si manis. Sehat Negeriku. https://sehatnegeriku.kemkes.go.id/baca/blog/20240110/5344736/saatnya-mengatur-si-manis/
- World Health Organization. (2024, November 14). Diabetes. https://www.who.int/news-room/fact-sheets/detail/diabetes
- Kusuma, K. D., & Akbar, M. (2022). Diabetes risk prediction using Extreme Gradient Boosting (XGBoost). Jurnal Online Informatika, 7(2), 193–200. https://doi.org/10.15575/join.v7i2.970
- Apriliah, W., Kurniawan, I., Baydhowi, M., & Haryati, T. (2021). Prediksi kemungkinan diabetes pada tahap awal menggunakan algoritma klasifikasi Random Forest. SISTEMASI: Jurnal Sistem Informasi, 10(1), 163–171. https://doi.org/10.32520/stmsi.v10i1.1129
  

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
Dataset yang digunakan dalam proyek ini diambil dari situs [Kaggle - Diabetes Prediction Dataset](https://www.kaggle.com/datasets/iammustafatz/diabetes-prediction-dataset). Dataset ini berisi data kesehatan klinis yang digunakan untuk memprediksi kemungkinan seseorang menderita diabetes. Dataset terdiri dari 100.000 baris dengan berbagai fitur yang merepresentasikan informasi medis dasar dan gaya hidup individu. Data ini sangat relevan karena mencakup fitur-fitur yang umum digunakan dalam proses skrining diabetes seperti usia, tekanan darah, indeks massa tubuh (BMI), kadar glukosa, hingga status merokok. 

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

   **Insight:**
   Jumlah non-penderita diabetes lebih banyak dibandingkan dengan penderita diabetes, sehingga dapat diamati bahwa terjadi ketidakseimbangan data.

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
Sebelum data digunakan untuk pelatihan model machine learning, dilakukan beberapa tahapan data preparation agar memastikan data bersih, konsisten, dan sesuai untuk proses pemodelan. Adapun tahapan-tahapan yang dilakukan adalah sebagai berikut:

1. Pemeriksaan dan Penanganan Data Duplikat

   Data duplikat dapat menyebabkan bias dalam pelatihan model karena informasi yang sama dihitung lebih dari sekali. Hal ini dapat membuat model overfitting atau belajar dari pola yang tidak general. Pada dataset ini, terdapat data duplikat sebanyak 3854 baris.


    ```
    # Mengecek duplikasi dataset menggunakan duplicated().sum()
    print('Jumlah data duplikat:', df.duplicated().sum())
    
    # Menghapus data duplikat menggunakan drop_duplicates().
    df.drop_duplicates(inplace=True)
    
    print('Jumlah data duplikat setelah dihapus:', df.duplicated().sum())
    ```
    ![image](https://github.com/user-attachments/assets/df6a9dd4-0163-44f5-ba24-3eefabda39ab)

2. Pemeriksaan dan Penanganan Outlier

   Nilai yang tidak wajar atau ekstrem dapat mengganggu proses pelatihan, terutama pada algoritma yang sensitif terhadap skala dan distribusi data seperti XGBoost. Outlier diganti dengan rata-rata untuk menjaga distribusi data tetap stabil tanpa membuang informasi terlalu banyak. Pemeriksaan nilai ekstrem dilakukan pada fitur `age`, `bmi`, `HbA1c_level`, dan `blood_glucose_level`. Namun, hasilnya menunjukkan bahwa pada fitur `age` terdapat 910 baris dengan usia di bawah 1 tahun dan fitur `bmi` memiliki 115 baris dengan BMI yang tidak wajar yaitu kurang dari 10 dan lebih dari 60 yang dianggap tidak realistis.

   ```
    # Mengecek data anomali
    # Usia terlalu kecil (misalnya < 1 tahun)
    anomali_usia = df[df['age'] < 1]
    print(f"Jumlah data dengan usia < 1 tahun: {len(anomali_usia)}")
    
    # BMI yang tidak realistis (misalnya < 10 atau > 60)
    anomali_bmi = df[(df['bmi'] < 10) | (df['bmi'] > 60)]
    print(f"Jumlah data dengan BMI tidak wajar: {len(anomali_bmi)}")
    
    # HbA1c level yang sangat ekstrem (misalnya < 3 atau > 15)
    anomali_hba1c = df[(df['HbA1c_level'] < 3) | (df['HbA1c_level'] > 15)]
    print(f"Jumlah data dengan HbA1c_level tidak wajar: {len(anomali_hba1c)}")
    
    # Blood glucose level sangat rendah atau tinggi (misalnya < 50 atau > 300)
    anomali_glukosa = df[(df['blood_glucose_level'] < 50) | (df['blood_glucose_level'] > 300)]
    print(f"Jumlah data dengan blood_glucose_level tidak wajar: {len(anomali_glukosa)}")
  
   ```
   ![image](https://github.com/user-attachments/assets/40f5979e-14a7-4fbb-ac32-a6e9779e97c1)
  
   ```
    # Mengganti usia kurang dari 1 tahun dengan rata-rata usia
    rata_rata_usia = df['age'].mean()
    df.loc[df['age'] < 1, 'age'] = rata_rata_usia
    
    # Menampilkan jumlah data dengan usia < 1 tahun setelah perubahan
    anomali_usia_baru = df[df['age'] < 1]
    print(f"Jumlah data dengan usia < 1 tahun setelah diganti dengan rata-rata: {len(anomali_usia_baru)}")
    
    print("\nDataframe setelah penggantian usia:")
    df.describe()
   ```
   ![image](https://github.com/user-attachments/assets/a2a4dfb6-6c93-4472-bcc6-b0b03eda7434)

   ```
    df_before = df.copy()
    
    # Hitung rata-rata BMI
    rata_rata_bmi = df['bmi'].mean()
    
    # Ganti nilai BMI anomali dengan rata-rata
    df.loc[(df['bmi'] < 10) | (df['bmi'] > 60), 'bmi'] = rata_rata_bmi
    
    # Visualisasi Boxplot BMI Sebelum dan Setelah Perbaikan
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    sns.boxplot(x='diabetes', y='bmi', data=df_before, palette='pastel')
    plt.title('BMI Sebelum Diganti dengan Rata-rata')
    plt.xlabel('Diabetes')
    plt.ylabel('BMI')
    
    plt.subplot(1, 2, 2)
    sns.boxplot(x='diabetes', y='bmi', data=df, palette='pastel')
    plt.title('BMI Setelah Diganti dengan Rata-rata')
    plt.xlabel('Diabetes')
    plt.ylabel('BMI')
    
    plt.tight_layout()
    plt.show()

    # Menampilkan jumlah data dengan BMI anomali setelah perubahan
    anomali_bmi_baru = df[(df['bmi'] < 10) | (df['bmi'] > 60)]
    print(f"Jumlah data dengan BMI tidak wajar setelah diganti dengan rata-rata: {len(anomali_bmi_baru)}")
   ```
   ![image](https://github.com/user-attachments/assets/966f579b-a683-47d2-8cef-0cb97db04f63)
   ![image](https://github.com/user-attachments/assets/c88cae2e-680c-4756-b076-d9077c4e4e99)

3. Penanganan pada Kolom `gender`

   Menghapus 'other' pada kolom gender karena jumlahnya sangat kecil, hanya sebanyak 18 baris.

   ```
    # Menghapus 'other' pada kolom gender
    df = df[df['gender'] != 'Other']
    print(df['gender'].value_counts())
   ```
   ![image](https://github.com/user-attachments/assets/08da1bae-f993-4d0e-8110-a913577ec15f)

4. Encoding

   Variabel kategorikal seperti gender dan smoking_history diubah menjadi numerik menggunakan LabelEncoder. Algoritma machine learning hanya dapat memproses data numerik, sehingga encoding memastikan model dapat membaca dan memproses variabel kategorikal tanpa kehilangan informasi penting. Pada variabel gender, terdapat dua label kategori, yaitu 0 (female) dan 1 (male), sedangkan pada variabel smoking_history, terdapat enam label kategori, yaitu 0 (No Info), 1 (current), 2 (ever), 3 (former), 4 (never), dan 5 (not current).
   
   ```
    categorical_cols = ['gender', 'smoking_history']
    le = LabelEncoder()
    df_encoded = df.copy()
    label_encoders = {}
    
    for col in categorical_cols:
        le = LabelEncoder()
        df_encoded[col] = le.fit_transform(df_encoded[col])
        label_encoders[col] = le
   ```

5. Normalisasi

   Data numerik dinormalisasi menggunakan Min-Max Scaling agar berada dalam rentang 0–1. Normalisasi ini memastikan bahwa tidak ada satu pun fitur numerik yang mendominasi dalam proses pembelajaran model.
   
   ```
    # Initialize MinMaxScaler
    scaler = MinMaxScaler()
    numerical_cols = df_encoded.select_dtypes(include=np.number).columns.tolist()
    numerical_cols.remove('diabetes') # Exclude the target variable
    df_scaled = df_encoded.copy()
    df_scaled[numerical_cols] = scaler.fit_transform(df_scaled[numerical_cols])
   ```

6. Penanganan Ketidakseimbangan Data dengan SMOTE

    Ketidakseimbangan data dapat menyebabkan model bias terhadap kelas mayoritas. Kelas target (diabetes) memiliki data yang tidak seimbang, dengan jumlah penderita diabetes yang lebih sedikit dibandingkan dengan non-penderita diabetes. SMOTE diterapkan untuk menyeimbangkan data pada jumlah penderita dan non-penderita menjadi sama banyak, dengan jumlah penderita dan non-penderita diabetes menjadi 87646.
    
   ```
    # Memisahkan fitur (X) dan target (y)
    X = df_scaled.drop('diabetes', axis=1)
    y = df_scaled['diabetes']
  
    # Menerapkan SMOTE untuk menyeimbangkan data sebelum split
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y) 
    
    print(f"Jumlah penderita diabetes setelah SMOTE: {sum(y_resampled)}") 
    print(f"Jumlah non-penderita diabetes setelah SMOTE: {len(y_resampled) - sum(y_resampled)}") 
   ```

  ![image](https://github.com/user-attachments/assets/613d876b-2365-42a2-95b4-dfa53c00c86d)

7. Train-Test-Split

    Dataset yang telah dibersihkan, dinormalisasi, dan diseimbangkan menggunakan SMOTE dibagi menjadi dua bagian, yaitu training set (80%) untuk melatih model machine learning dan testing set (20%) untuk menguji performa model terhadap data yang belum pernah dilihat sebelumnya. Memisahkan data pelatihan dan pengujian, dapat mengevaluasi sejauh mana model dapat menggeneralisasi ke data baru. Jumlah data training yang digunakan adalah 140233 dan jumlah data test adalah 35059.
    
   ```
    # Membagi data menjadi data latih dan data uji
    X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)
    
    print(f"Jumlah data training: {X_train.shape[0]}")
    print(f"Jumlah data test: {X_test.shape[0]}")
    print(f"Jumlah seluruh data setelah SMOTE: {X_resampled.shape[0]}")
   ```
   ![image](https://github.com/user-attachments/assets/de9a137f-2f62-4b7b-98b3-9e878909cf64)

## Modeling

  Tahapan ini berfokus pada proses pelatihan model machine learning untuk memprediksi risiko diabetes menggunakan dua algoritma berbeda, yaitu Random Forest dan XGBoost.

- Random Forest Classifier

  Deskripsi:
  
  Random Forest adalah algoritma ensemble berbasis pohon keputusan yang membangun banyak pohon secara acak dan menggabungkan hasilnya (melalui voting mayoritas) untuk klasifikasi. Algoritma ini dikenal handal, tahan terhadap overfitting, dan bekerja baik pada data dengan banyak fitur dan korelasi kompleks.

  Parameter yang digunakan adalah random_state = 42 untuk memastikan replikasi hasil.

  Hasil akurasi: 97.67% 

- XGBoost Classifier

  Deskripsi:
  
  XGBoost (Extreme Gradient Boosting) adalah algoritma boosting yang sangat efisien dan powerful. Algoritma ini bekerja dengan membangun model secara bertahap, dengan setiap model baru mencoba memperbaiki kesalahan dari model sebelumnya. XGBoost dikenal karena kecepatannya dan yang performa tinggi.
  
  Parameter yang digunakan adalah random_state=42 untuk memastikan hasil yang konsisten.

  Hasil akurasi: 97.66%
  

| Algoritma         | Kelebihan                                                         | Kekurangan                                                 |
| ----------------- | ----------------------------------------------------------------- | ---------------------------------------------------------- |
| **Random Forest** | - Mudah digunakan, minim tuning                                   | - Kurang optimal untuk dataset besar dengan fitur kompleks |
|                   | - Mampu menangani data yang tidak seimbang                        | - Interpretasi model sulit (black box)                     |
|                   | - Tidak mudah overfitting karena averaging antar pohon            | - Prediksi bisa lambat untuk data besar                    |
| **XGBoost**       | - Performa tinggi dan sangat akurat                               | - Proses training lebih lama dibanding Random Forest       |
|                   | - Memiliki regularisasi (mencegah overfitting)                    | - Butuh tuning parameter lebih kompleks                    |
|                   | - Dapat menangani missing values                                  | - Konsumsi memori lebih besar                              |
|                   | - Mendukung paralelisasi dan efisiensi waktu                      | - Lebih sulit dipahami untuk pemula                        |


Berdasarkan perbandingan hasil akurasi, algoritma **Random Forest** memiliki performa terbaik dibandingkan dengan **XGBoost**, yaitu dengan akurasi sebesar **97,67%**, sedikit lebih tinggi dari akurasi **XGBoost** sebesar **97,66%**. Oleh karena itu, **Random Forest** dipilih sebagai **model terbaik** dalam memprediksi risiko diabetes pada proyek ini.


## Evaluation

Pada proyek klasifikasi ini, digunakan sejumlah metrik evaluasi untuk menilai seberapa baik model dalam mengklasifikasikan apakah seseorang menderita diabetes atau tidak. Metrik yang digunakan adalah akurasi, precision, recall, dan F1-score. Pemilihan metrik ini disesuaikan dengan konteks masalah, yaitu pendeteksian penyakit dan penting untuk mengenali kasus positif (penderita diabetes) dengan akurat.


Metrik evaluasi yang digunakan:
| Metrik        | Penjelasan Singkat                                                                                                                                                                                    | Rumus                                                    |
| ------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------- |
| **Akurasi**   | Proporsi total prediksi yang benar (positif maupun negatif) dari seluruh data. Cocok digunakan saat distribusi data relatif seimbang.                                                                 | **Accuracy** = (TP + TN) / (TP + TN + FP + FN)           |
| **Precision** | Mengukur seberapa banyak prediksi positif yang benar-benar positif. Berguna ketika penting meminimalkan kesalahan dalam prediksi positif (False Positive).                                            | **Precision** = TP / (TP + FP)                           |
| **Recall**    | Mengukur seberapa banyak kasus positif yang berhasil ditemukan oleh model. Sangat penting dalam konteks diagnosis karena ingin menghindari banyak kasus positif yang terlewat (False Negative). | **Recall** = TP / (TP + FN)                              |
| **F1-Score**  | Rata-rata harmonik dari precision dan recall. Berguna ketika ingin menyeimbangkan antara precision dan recall.                                                                                        | **F1** = 2 × (Precision × Recall) / (Precision + Recall) |

Keterangan:

| Simbol | Arti           | Penjelasan                                                                 |
| ------ | -------------- | -------------------------------------------------------------------------- |
| **TP** | True Positive  | Model memprediksi *positif* (diabetes), dan memang benar *positif*.        |
| **TN** | True Negative  | Model memprediksi *negatif* (non-diabetes), dan memang benar *negatif*.    |
| **FP** | False Positive | Model memprediksi *positif*, tetapi sebenarnya *negatif* (salah deteksi).  |
| **FN** | False Negative | Model memprediksi *negatif*, tetapi sebenarnya *positif* (kasus terlewat). |


Hasil Evaluasi
| **Metrik**                       | **Random Forest** | **XGBoost** |
| -------------------------------- | ----------------- | ----------- |
| **Akurasi**                      | 0.9767            | 0.9766      |
| **Precision (0 - Non-Diabetes)** | 0.98              | 0.96        |
| **Precision (1 - Diabetes)**     | 0.98              | 0.99        |
| **Recall (0 - Non-Diabetes)**    | 0.98              | 0.99        |
| **Recall (1 - Diabetes)**        | 0.98              | 0.96        |
| **F1-Score (0 - Non-Diabetes)**  | 0.98              | 0.98        |
| **F1-Score (1 - Diabetes)**      | 0.98              | 0.98        |


Hasil Confusion Matrix

![image](https://github.com/user-attachments/assets/862e569a-1c9f-4339-97ab-32700e58c8af)
![image](https://github.com/user-attachments/assets/2ff50656-09a2-455f-9f40-38f494fae584)


|                     | **Random Forest**     | **XGBoost**           |
| ------------------- | --------------------- | --------------------- |
| True Negative (TN)  | 17.035 (Non-Diabetes) | 17.331 (Non-Diabetes) |
| False Positive (FP) | 430                   | 134                   |
| False Negative (FN) | 386                   | 685                   |
| True Positive (TP)  | 17.208 (Diabetes)     | 16.909 (Diabetes)     |


Random Forest:

Random Forest mampu mengidentifikasi penderita diabetes (TP = 17.208) dan non-diabetes (TN = 17.035) dengan baik. Model ini memiliki False Negative (FN) rendah (386), artinya lebih sedikit penderita diabetes yang terlewat deteksi. Namun, False Positive (FP) cukup tinggi (430), sehingga lebih banyak orang sehat yang salah diprediksi sebagai diabetes.


XGBoost:

XGBoost lebih baik dalam mengenali orang sehat dengan False Positive yang rendah (134) dan True Negative yang sedikit lebih tinggi (17.331). Namun, model ini memiliki False Negative lebih tinggi (685), yang berarti lebih banyak penderita diabetes yang tidak terdeteksi.


Kesimpulan:

Dalam proyek prediksi risiko diabetes ini, kedua algoritma Random Forest dan XGBoost menunjukkan performa yang sangat baik dengan akurasi hampir sama, yaitu Random Forest sekitar 97,67% dan XGBoost sekitar 97,66%. Random Forest memiliki keunggulan dengan False Negative yang lebih rendah, sehingga lebih sedikit penderita diabetes yang terlewatkan, meskipun memiliki False Positive yang lebih tinggi yang menyebabkan lebih banyak orang sehat terdeteksi sebagai penderita diabetes secara salah. Sebaliknya, XGBoost lebih efektif dalam meminimalkan False Positive, sehingga lebih sedikit kesalahan deteksi positif pada orang sehat, tetapi mengorbankan False Negative yang lebih tinggi, berpotensi melewatkan lebih banyak kasus diabetes. Berdasarkan hasil evaluasi dan keseimbangan antara deteksi penderita diabetes dan kesalahan prediksi positif, Random Forest dipilih sebagai model terbaik untuk memprediksi risiko diabetes pada proyek ini. Metrik precision, recall, dan F1-score untuk kedua kelas juga menunjukkan bahwa kedua model bekerja dengan sangat baik dalam mengklasifikasikan diabetes dan non-diabetes.

**---Ini adalah bagian akhir laporan---**
