# Proyek Analisis Sentimen Teks Bahasa Indonesia

Proyek ini mengimplementasikan pipeline lengkap untuk analisis sentimen pada teks berbahasa Indonesia. Model dilatih untuk mengklasifikasikan teks ke dalam dua kategori sentimen: **Positif** atau **Negatif**.

## Fitur Utama

* **Preprocessing Data Komprehensif**: Termasuk pembersihan teks (case folding, penghapusan URL, mention, tanda baca), normalisasi slang, penanganan negasi, stemming (Sastrawi), dan penghapusan stopwords (NLTK).
* **Balancing Dataset**: Menggunakan teknik undersampling untuk menyeimbangkan jumlah data antara kelas positif dan negatif, bertujuan untuk mengurangi bias model.
* **Caching Preprocessing**: Hasil preprocessing data disimpan dalam format `.feather` untuk mempercepat pemuatan pada eksekusi berikutnya, sehingga proses pembersihan yang lama hanya perlu dilakukan sekali.
* **Model Custom**: Menggunakan arsitektur Bi-LSTM dengan mekanisme Attention yang dibangun menggunakan TensorFlow (Keras) dan dilatih dari awal pada dataset yang disediakan.
* **Training dan Prediksi Terpisah**:
    * `sentiment_analysis_pipeline.py`: Skrip untuk melakukan seluruh alur preprocessing, training model, penyimpanan artefak (model, tokenizer, label encoder), dan evaluasi model.
    * `predict.py`: Skrip ringan untuk melakukan prediksi interaktif menggunakan model yang sudah dilatih.
* **Evaluasi Model**: Setelah training, skrip akan menampilkan Classification Report dan Confusion Matrix (beserta visualisasinya jika library terinstal) untuk mengevaluasi performa model pada data validasi.
* **Penyimpanan Artefak**: Model terbaik, tokenizer, dan label encoder disimpan secara otomatis setelah training atau ketika performa validasi terbaik tercapai.

## Struktur Proyek

Berikut adalah struktur folder dan file utama dalam proyek ini:

ML-EMOTICA/
├── data/
│   ├── sentiment_dataset.csv                           # File dataset mentah (WAJIB DISEDIAKAN PENGGUNA)
│   └── sentiment_dataset_cleaned_binary_balanced.feather # Cache dataset bersih & seimbang
├── models/
│   ├── best_sentiment_model.keras                      # Model Keras yang sudah dilatih
│   ├── label_encoder.joblib                            # LabelEncoder yang sudah di-fit
│   ├── tokenizer/                                      # Folder berisi file konfigurasi tokenizer
│   │   ├── special_tokens_map.json
│   │   ├── tokenizer_config.json
│   │   └── vocab.txt
│   └── confusion_matrix_final_binary.png               # (Opsional) Gambar Confusion Matrix
├── venv/                                             # Folder virtual environment Python
├── sentiment_analysis_pipeline.py                    # Skrip utama untuk training dan evaluasi model
├── predict.py                                        # Skrip untuk melakukan prediksi interaktif
├── requirements.txt                                  # Daftar semua dependensi Python
└── README.md                                         # File ini


## Persyaratan

* Python 3.11
* `pip` (Python package installer)

## Instalasi & Setup

1.  **Clone atau Unduh Proyek**:
    Jika proyek ini ada di repository Git, clone. Jika tidak, pastikan semua file ada dalam satu folder utama.

2.  **Buat dan Aktifkan Virtual Environment**:
    Sangat direkomendasikan untuk menggunakan virtual environment. Buka terminal di folder utama proyek:
    ```bash
    python -m venv venv
    ```
    Aktifkan environment:
    * Windows:
        ```bash
        .\venv\Scripts\activate
        ```
    * macOS/Linux:
        ```bash
        source venv/bin/activate
        ```

3.  **Install Dependensi**:
    Pastikan Anda sudah mengaktifkan virtual environment, lalu jalankan:
    ```bash
    pip install -r requirements.txt
    ```
    Ini akan menginstal semua library yang dibutuhkan sesuai versi yang tercantum di `requirements.txt`.

4.  **Sediakan Dataset Mentah**:
    * Letakkan file dataset mentah Anda dengan nama `sentiment_dataset.csv` di dalam folder `data/`.
    * Dataset ini **HARUS** memiliki kolom bernama `text` (berisi teks ulasan/kalimat) dan `sentiment` (berisi label sentimen, misal "positif", "negatif", atau "netral". Skrip akan otomatis memfilter dan menyeimbangkan untuk "positif" dan "negatif").

## Cara Penggunaan

### 1. Melatih Model
Skrip `sentiment_analysis_pipeline.py` akan melakukan seluruh proses dari awal hingga akhir: memuat data mentah, membersihkan dan menyeimbangkan data (jika file cache `.feather` belum ada), melatih model, menyimpan artefak, dan menampilkan metrik evaluasi.

* **Perintah**:
    ```bash
    python sentiment_analysis_pipeline.py
    ```
* **Eksekusi Pertama**: Proses pembersihan data akan memakan waktu cukup lama (tergantung ukuran dataset dan spesifikasi komputer Anda). Swifter akan menggunakan semua core CPU untuk mempercepatnya, dan progress bar akan ditampilkan. Hasil pembersihan akan disimpan sebagai `sentiment_dataset_cleaned_binary_balanced.feather` di folder `data/`.
* **Eksekusi Berikutnya**: Jika file `.feather` sudah ada, proses pemuatan data akan sangat cepat.
* **Proses Training**: Training model juga membutuhkan waktu. Callback `ArtifactSaver` akan menyimpan model, tokenizer, dan encoder terbaik secara otomatis ke folder `models/` setiap kali akurasi validasi meningkat. `EarlyStopping` akan menghentikan training jika tidak ada peningkatan performa.
* **Output**:
    * Artefak model (file `.keras`, folder `tokenizer`, file `label_encoder.joblib`) di folder `models/`.
    * Classification Report dan Confusion Matrix (termasuk gambar `.png`-nya jika `matplotlib` & `seaborn` terinstal) ditampilkan di konsol dan disimpan di folder `models/`.

### 2. Melakukan Prediksi Interaktif
Setelah model berhasil dilatih dan semua artefak tersimpan di folder `models/`, Anda bisa menggunakan `predict.py` untuk mencoba model secara interaktif.

* **Perintah**:
    ```bash
    python predict.py
    ```
* **Cara Kerja**: Skrip akan memuat model, tokenizer, dan label encoder dari folder `models/`. Kemudian, Anda akan diminta untuk mengetikkan kalimat. Model akan memprediksi sentimennya (POSITIF atau NEGATIF) beserta tingkat kepercayaannya. Ketik 'exit' atau 'keluar' untuk berhenti.

## Teknologi Utama yang Digunakan

* **Python 3.11**
* **TensorFlow (Keras)**: Untuk membangun dan melatih model deep learning.
* **Pandas & NumPy**: Untuk manipulasi dan pemrosesan data.
* **Scikit-learn**: Untuk `LabelEncoder`, `train_test_split`, dan metrik evaluasi.
* **Transformers (Hugging Face)**: Untuk `BertTokenizerFast` (IndoBERT Tokenizer).
* **Sastrawi**: Untuk stemming Bahasa Indonesia.
* **NLTK**: Untuk stopwords Bahasa Indonesia.
* **Swifter**: Untuk mempercepat operasi `apply` pada Pandas dengan paralelisasi.
* **tqdm**: Untuk menampilkan progress bar.
* **Joblib**: Untuk menyimpan dan memuat objek Python (digunakan untuk `LabelEncoder`).
* **PyArrow**: Untuk mendukung format file `.feather` yang efisien.
* **(Opsional) Matplotlib & Seaborn**: Untuk visualisasi Confusion Matrix.

## Catatan Penting
* Pastikan kelas `AdvancedTextPreprocessor` dalam file `predict.py` **identik** dengan versi yang ada di `sentiment_analysis_pipeline.py` yang digunakan saat training untuk memastikan konsistensi preprocessing.
* Semua path file dalam skrip bersifat relatif terhadap lokasi eksekusi skrip.
* Untuk melatih ulang model dari awal dengan konfigurasi data yang berbeda (misalnya, jika Anda mengubah logika undersampling atau menghapus cache), pastikan untuk menghapus file `.feather` di folder `data/` dan seluruh isi folder `models/` sebelum menjalankan `sentiment_analysis_pipeline.py`.