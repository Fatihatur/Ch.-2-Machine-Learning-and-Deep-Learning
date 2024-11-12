Berikut adalah struktur teks yang telah diubah:

---

# Bab 2: Pembelajaran Mesin dan Pembelajaran Mendalam

## Daftar Isi

1. [Tugas Bab 2 - Kasus 1 (`02_Kelompok_E_1.ipynb`)](#1-tugas-bab-2---kasus-1)
2. [Tugas Bab 2 - Kasus 2 (`02_Kelompok_E_2.ipynb`)](#2-tugas-bab-2---kasus-2)
3. [Tugas Bab 2 - Kasus 3 (`02_Kelompok_E_3.ipynb`)](#3-tugas-bab-2---kasus-3)
4. [Tugas Bab 2 - Kasus 4 (`02_Kelompok_E_4.ipynb`)](#4-tugas-bab-2---kasus-4)

---

## 1. Tugas Bab 2 - Kasus 1

### Deskripsi Tugas
Kasus pertama melibatkan pembuatan model klasifikasi menggunakan algoritma Pembelajaran Mesin untuk memprediksi apakah nasabah akan `Exited` dari bank berdasarkan dataset bank.

### Dataset
Dataset yang digunakan adalah `SC_HW1_bank_data.csv`.

### Library yang Digunakan
- **Pandas**: Untuk manipulasi data.
- **Numpy**: Untuk operasi numerik.
- **Scikit-learn**: Untuk pembuatan model klasifikasi (RandomForestClassifier, SVC, GradientBoostingClassifier).

### Persyaratan
- **Modul dan Versi**: Pastikan semua modul yang diperlukan sudah terinstal.
- **Kolom dan Data yang Digunakan**: Hilangkan kolom yang tidak relevan, dan lakukan preprocessing.

### Langkah Pengerjaan
1. **Preprocessing Data**: Menghapus kolom yang tidak relevan, melakukan *One-Hot Encoding*, dan normalisasi dengan `MinMaxScaler`.
2. **Pemodelan dan Evaluasi**:
   - **Model #1: Random Forest**
   - **Model #2: Support Vector Classifier (SVC)**
   - **Model #3: Gradient Boosting Classifier**
3. **Hyperparameter Tuning**: Melakukan Grid Search untuk parameter optimal.
4. **Evaluasi**:
   - **Metrics**: `accuracy_score`, `classification_report`, dan `confusion_matrix`.
   - Membandingkan hasil akurasi dari ketiga model.

### Hasil dan Kesimpulan
Hasil menunjukkan **Gradient Boosting Classifier** memiliki performa terbaik dalam hal akurasi dan kecepatan pemrosesan.

---

## 2. Tugas Bab 2 - Kasus 2

### Deskripsi Tugas
Kasus kedua melibatkan *clustering* data menggunakan algoritma *KMeans Clustering* untuk segmentasi data.

### Dataset
Dataset yang digunakan adalah `cluster_s1.csv`.

### Library yang Digunakan
- **Pandas**: Untuk manipulasi data.
- **Numpy**: Untuk operasi numerik.
- **Matplotlib & Seaborn**: Untuk visualisasi data.
- **Scikit-learn**: Untuk KMeans Clustering.

### Persyaratan
- **Modul dan Versi**: Pastikan semua modul yang diperlukan sudah terinstal.

### Langkah Pengerjaan
1. **Persiapan Data**: Menghapus kolom yang tidak relevan.
2. **Menentukan Jumlah Cluster**:
   - Menggunakan *Silhouette Score* untuk menentukan nilai *k* terbaik.
3. **Pemodelan KMeans**:
   - Melatih model KMeans dengan jumlah cluster yang optimal.
4. **Evaluasi dan Visualisasi**:
   - Scatter plot hasil clustering dengan warna berbeda untuk setiap cluster.

### Hasil dan Kesimpulan
Nilai *k* terbaik diperoleh dari *Silhouette Score* tertinggi, dengan hasil clustering divisualisasikan dalam bentuk scatter plot.

---

## 3. Tugas Bab 2 - Kasus 3

### Deskripsi Tugas
Kasus ketiga mengharuskan pembuatan model regresi menggunakan TensorFlow-Keras untuk memprediksi harga rumah di California menggunakan arsitektur *Multilayer Perceptron*.

### Dataset
Dataset yang digunakan adalah California House Price dari *Scikit-Learn* dengan variabel target berupa harga rumah.

### Library yang Digunakan
- **Pandas**: Untuk manipulasi data.
- **Numpy**: Untuk operasi array dan matriks.
- **TensorFlow & Keras**: Framework dan API utama untuk deep learning.
- **Scikit-learn**: Untuk preprocessing data.
- **Matplotlib**: Untuk visualisasi hasil pelatihan.

### Langkah Penyelesaian
1. **Persiapan dan Pemisahan Data**:
   - Konversi data ke DataFrame.
   - Pisahkan data menjadi *train*, *validation*, dan *test set*.
   - Standarisasi dan normalisasi data.

2. **Membangun Model Neural Network**:
   - Dua hidden layer dengan 30 neuron dan aktivasi ReLU.
   - Gabungan input ganda (input A dan B) sebelum output layer.

3. **Training dan Evaluasi Model**:
   - Tentukan *learning rate*, *epochs*, dan *batch size*.
   - Lakukan *training* pada model dan visualisasikan *loss*.

4. **Menyimpan Model**:
   - Simpan model yang telah dilatih dan prediksi sampel baru.

---

## 4. Tugas Bab 2 - Kasus 4

### Deskripsi Tugas
Kasus keempat adalah membangun model klasifikasi menggunakan PyTorch untuk mendeteksi transaksi fraud pada dataset *Credit Card Fraud 2023* menggunakan *Multilayer Perceptron*.

### Dataset
Dataset yang digunakan adalah *Credit Card Fraud 2023*, dengan variabel target kolom *Class* (fraud/non-fraud).

### Library yang Digunakan
- **Pandas**: Untuk manipulasi data.
- **cuDF**: Versi GPU-accelerated dari Pandas.
- **cuML**: Untuk pemrosesan data di GPU.
- **Numpy (cuPy)**: Untuk operasi array di GPU.
- **Scikit-learn**: Untuk standardisasi dan pemisahan dataset.
- **PyTorch**: Framework untuk deep learning dan evaluasi.

### Langkah Penyelesaian
1. **Impor Dataset dengan GPU**:
   - Unduh dataset, lalu baca menggunakan cuDF (Pandas versi GPU).
   - Hilangkan kolom ID dan lakukan standarisasi.

2. **Pemisahan Data dan Konversi ke Tensor**:
   - Tentukan fitur X dan target Y.
   - Lakukan pemisahan data *train* dan *test* di GPU.
   - Konversi data ke Tensor untuk *DataLoader* PyTorch.

3. **Membangun Model Neural Network**:
   - *Multilayer Perceptron* dengan 4 hidden layers menggunakan PyTorch.
   - Tentukan parameter seperti *epochs*, *num_layers*, dan *learning rate*.

4. **Training dan Evaluasi Model**:
   - Lakukan *training* dan evaluasi akurasi model.
   - Fine-tuning jika akurasi model belum mencapai 95%.

---

## Petunjuk Menjalankan Kode

1. Buka Google Colab.
2. Unggah file `02_Kelompok_B_1.ipynb` hingga `02_Kelompok_B_4.ipynb`.
3. Jalankan setiap sel sesuai urutan dalam *notebook* dan ikuti instruksi yang disediakan.
