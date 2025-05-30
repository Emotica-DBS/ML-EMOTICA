# Pipeline Analisis Sentimen Teks Bahasa Indonesia

![Python Version](https://img.shields.io/badge/python-3.9%2B-blue.svg)
![TensorFlow Version](https://img.shields.io/badge/tensorflow-2.16-orange.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

Proyek ini berisi pipeline lengkap untuk melatih dan menjalankan model analisis sentimen untuk teks berbahasa Indonesia. Pipeline ini menggunakan arsitektur model deep learning (Bi-LSTM dengan Attention) dan divalidasi menggunakan K-Fold Cross-Validation untuk memastikan performa yang kuat dan andal.

## Fitur Utama

- **Pipeline End-to-End**: Mulai dari pemrosesan data mentah hingga prediksi interaktif.
- **Preprocessing Tingkat Lanjut**: Termasuk normalisasi kata slang, stemming (Sastrawi), dan penanganan negasi khusus untuk Bahasa Indonesia.
- **Arsitektur Model Modern**: Menggunakan `Bidirectional LSTM` untuk menangkap konteks dari dua arah dan `Multi-Head Attention` untuk fokus pada bagian teks yang paling relevan.
- **Validasi K-Fold**: Melatih model dengan _5-Fold Cross-Validation_ untuk evaluasi yang lebih robust dan memilih model terbaik secara otomatis.
- **Mekanisme Cerdas**: Script akan **melatih model hanya sekali**. Pada eksekusi berikutnya, ia akan otomatis memuat model yang sudah disimpan untuk inferensi cepat.
- **Mode Prediksi Interaktif**: Setelah model siap, Anda bisa langsung mencoba memprediksi sentimen dari kalimat apa pun melalui terminal.

## Struktur Proyek

```
.
├── sentiment_analysis_pipeline.py    # Script utama yang menjalankan seluruh pipeline
├── requirements.txt                  # Daftar dependensi Python
├── README.md                         # Dokumentasi proyek (file ini)
├── combined_sentiment_datasets.csv   # (Opsional) File dataset yang akan digunakan
└── best_sentiment_model.keras        # (Output) File model terbaik yang disimpan setelah training
```

## Instalasi & Setup

Ikuti langkah-langkah berikut untuk menyiapkan dan menjalankan proyek ini di lingkungan lokal Anda.

### Prasyarat

- Python 3.9 atau yang lebih baru.

### Langkah-langkah Instalasi

1.  **Clone Repositori (jika ada di Git)**

    ```bash
    git clone [URL-repositori-anda]
    cd [nama-folder-repositori]
    ```

2.  **Buat dan Aktifkan Virtual Environment (Sangat Direkomendasikan)**
    Ini akan mengisolasi dependensi proyek Anda dari sistem utama.

    - **Untuk macOS/Linux:**
      ```bash
      python3 -m venv venv
      source venv/bin/activate
      ```
    - **Untuk Windows:**
      ```bash
      python -m venv venv
      .\venv\Scripts\activate
      ```

3.  **Instal Semua Dependensi**
    Gunakan file `requirements.txt` untuk menginstal semua library yang dibutuhkan dengan satu perintah.

    ```bash
    pip install -r requirements.txt
    ```

4.  **Siapkan Dataset**
    - Letakkan file dataset Anda (misalnya, `combined_sentiment_datasets.csv`) di direktori yang sama dengan script.
    - Jika file dataset tidak ditemukan, script akan membuat file _dummy_ untuk tujuan demonstrasi pada saat pertama kali dijalankan.

## Cara Penggunaan

Setelah instalasi selesai, Anda dapat menjalankan pipeline dari terminal.

### 1. Menjalankan Training (Eksekusi Pertama Kali)

Pada saat pertama kali Anda menjalankan script, ia akan secara otomatis mendeteksi bahwa tidak ada file model yang tersimpan (`best_sentiment_model.keras`) dan akan memulai proses training K-Fold.

```bash
python sentiment_analysis_pipeline.py
```

**Output yang Diharapkan:**

- Anda akan melihat log proses pembersihan data.
- Proses training 5-Fold akan dimulai, menampilkan progress bar untuk setiap fold.
- Di akhir, akan ada ringkasan hasil K-Fold dan pemberitahuan bahwa model terbaik telah disimpan.
- Script akan langsung masuk ke mode prediksi interaktif.

### 2. Menjalankan Mode Prediksi Interaktif (Eksekusi Selanjutnya)

Setelah file `best_sentiment_model.keras` dibuat, setiap kali Anda menjalankan script, proses training akan dilewati.

```bash
python sentiment_analysis_pipeline.py
```

**Output yang Diharapkan:**

- Pesan bahwa model sedang dimuat dari file yang ada.
- Script akan langsung masuk ke mode prediksi interaktif.
- Contoh interaksi:

  ```
  🚀 MEMULAI MODE PREDIKSI SENTIMEN INTERAKTIF
     Ketik 'exit' atau 'keluar' untuk berhenti.
  ==================================================

  Ketik sebuah kalimat: pelayanannya sangat memuaskan, saya pasti kembali lagi
  ➡️  Prediksi Sentimen: **POSITIF** (Kepercayaan: 99.89%)

  Ketik sebuah kalimat: produknya tidak sebagus yang saya kira
  ➡️  Prediksi Sentimen: **NEGATIF** (Kepercayaan: 98.75%)

  Ketik sebuah kalimat: exit
  👋 Terima kasih telah mencoba. Sampai jumpa!
  ```

## Lisensi

Proyek ini dilisensikan di bawah Lisensi MIT. Lihat file `LICENSE` untuk detailnya.
