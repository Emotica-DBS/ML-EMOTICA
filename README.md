# Proyek Analisis Sentimen Teks Bahasa Indonesia

Proyek ini mengimplementasikan pipeline lengkap untuk analisis sentimen pada teks berbahasa Indonesia. Model dilatih untuk mengklasifikasikan teks ke dalam dua kategori sentimen: **Positif** atau **Negatif**.

## 🎯 Fitur Utama

- **Preprocessing Data Komprehensif**: 
  - Case folding dan pembersihan teks
  - Penghapusan URL, mention, dan tanda baca
  - Normalisasi slang dan penanganan negasi
  - Stemming menggunakan Sastrawi
  - Penghapusan stopwords dengan NLTK

- **Balancing Dataset**: Teknik undersampling untuk menyeimbangkan jumlah data antara kelas positif dan negatif, mengurangi bias model

- **Caching Preprocessing**: Hasil preprocessing disimpan dalam format `.feather` untuk mempercepat pemuatan pada eksekusi berikutnya

- **Model Custom**: Arsitektur Bi-LSTM dengan mekanisme Attention menggunakan TensorFlow (Keras)

- **Training dan Prediksi Terpisah**:
  - `sentiment_analysis_pipeline.py`: Pipeline lengkap untuk preprocessing, training, dan evaluasi
  - `predict.py`: Skrip ringan untuk prediksi interaktif

- **Evaluasi Model**: Classification Report dan Confusion Matrix dengan visualisasi untuk evaluasi performa

- **Penyimpanan Artefak**: Otomatis menyimpan model terbaik, tokenizer, dan label encoder

## 📁 Struktur Proyek

```
ML-EMOTICA/
├── data/
│   ├── sentiment_dataset.csv                           # Dataset mentah (WAJIB)
│   └── sentiment_dataset_cleaned_binary_balanced.feather # Cache dataset bersih
├── models/
│   ├── best_sentiment_model.keras                      # Model Keras terlatih
│   ├── label_encoder.joblib                            # LabelEncoder
│   ├── tokenizer/                                      # Konfigurasi tokenizer
│   │   ├── special_tokens_map.json
│   │   ├── tokenizer_config.json
│   │   └── vocab.txt
│   └── confusion_matrix_final_binary.png               # Visualisasi Confusion Matrix
├── venv/                                               # Virtual environment
├── sentiment_analysis_pipeline.py                     # Skrip training utama
├── predict.py                                          # Skrip prediksi interaktif
├── requirements.txt                                    # Dependensi Python
└── README.md                                           # Dokumentasi ini
```

## 🔧 Persyaratan Sistem

- **Python**: 3.11 atau lebih tinggi
- **Package Manager**: pip

## 🚀 Instalasi & Setup

### 1. Clone atau Unduh Proyek
Pastikan semua file proyek berada dalam satu folder utama.

### 2. Buat Virtual Environment
```bash
# Buat virtual environment
python -m venv venv

# Aktivasi environment
# Windows:
.\venv\Scripts\activate

# macOS/Linux:
source venv/bin/activate
```

### 3. Install Dependensi
```bash
pip install -r requirements.txt
```

### 4. Persiapkan Dataset
- Letakkan file dataset dengan nama `sentiment_dataset.csv` di folder `data/`
- Dataset **HARUS** memiliki kolom:
  - `text`: berisi teks ulasan/kalimat
  - `sentiment`: berisi label sentimen ("positif", "negatif", atau "netral")
- Skrip akan otomatis memfilter dan menyeimbangkan data untuk kelas "positif" dan "negatif"

## 📖 Cara Penggunaan

### 1. Melatih Model

Jalankan pipeline lengkap dari preprocessing hingga evaluasi:

```bash
python sentiment_analysis_pipeline.py
```

**Proses yang terjadi:**
- **Eksekusi Pertama**: Pembersihan data memakan waktu (tergantung ukuran dataset). Progress bar akan ditampilkan dan hasil disimpan sebagai cache `.feather`
- **Eksekusi Berikutnya**: Pemuatan data sangat cepat menggunakan cache
- **Training**: Model dilatih dengan callbacks otomatis untuk menyimpan model terbaik
- **Output**: 
  - Artefak model tersimpan di folder `models/`
  - Classification Report dan Confusion Matrix ditampilkan di konsol
  - Visualisasi confusion matrix disimpan sebagai gambar PNG

### 2. Prediksi Interaktif

Setelah model selesai dilatih, gunakan skrip prediksi:

```bash
python predict.py
```

**Cara kerja:**
- Memuat model, tokenizer, dan label encoder dari folder `models/`
- Meminta input kalimat dari pengguna
- Menampilkan prediksi sentimen (POSITIF/NEGATIF) dengan tingkat kepercayaan
- Ketik `exit` atau `keluar` untuk menghentikan program

## 🛠️ Teknologi yang Digunakan

| Kategori | Library | Fungsi |
|----------|---------|---------|
| **Deep Learning** | TensorFlow (Keras) | Membangun dan melatih model Bi-LSTM |
| **Data Processing** | Pandas, NumPy | Manipulasi dan pemrosesan data |
| **Machine Learning** | Scikit-learn | LabelEncoder, train_test_split, evaluasi |
| **NLP** | Transformers (Hugging Face) | BertTokenizerFast (IndoBERT) |
| **Text Processing** | Sastrawi, NLTK | Stemming dan stopwords Bahasa Indonesia |
| **Performance** | Swifter | Paralelisasi operasi pandas |
| **Utilities** | tqdm, Joblib, PyArrow | Progress bar, serialization, format file |
| **Visualization** | Matplotlib, Seaborn | Visualisasi confusion matrix (opsional) |

## ⚠️ Catatan Penting

1. **Konsistensi Preprocessing**: Pastikan kelas `AdvancedTextPreprocessor` di `predict.py` identik dengan versi di `sentiment_analysis_pipeline.py`

2. **Path Relatif**: Semua path file bersifat relatif terhadap lokasi eksekusi skrip

3. **Training Ulang**: Untuk melatih ulang dari awal:
   - Hapus file `.feather` di folder `data/`
   - Hapus seluruh isi folder `models/`
   - Jalankan kembali `sentiment_analysis_pipeline.py`

4. **Performance**: 
   - Swifter menggunakan semua core CPU untuk mempercepat preprocessing
   - EarlyStopping mencegah overfitting selama training
   - Model terbaik disimpan otomatis berdasarkan akurasi validasi

## 📊 Output Model

Setelah training selesai, Anda akan mendapatkan:
- **Classification Report**: Precision, recall, F1-score untuk setiap kelas
- **Confusion Matrix**: Matriks kebingungan dengan visualisasi
- **Model Artifacts**: File model, tokenizer, dan encoder siap pakai untuk produksi
