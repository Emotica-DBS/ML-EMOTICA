# ==============================================================================
# SENTIMENT ANALYSIS PIPELINE
#
# Deskripsi:
# Script ini mencakup seluruh pipeline untuk analisis sentimen teks Bahasa Indonesia:
# 1. Memuat dan membersihkan dataset.
# 2. Melatih model klasifikasi sentimen menggunakan TensorFlow (LSTM dengan Attention)
#    melalui K-Fold Cross-Validation untuk mendapatkan model yang robust.
# 3. Menyimpan model terbaik yang dihasilkan dari proses training.
# 4. Jika model sudah ada, script akan melewatkan training dan langsung memuatnya.
# 5. Menyediakan antarmuka interaktif di terminal untuk memprediksi sentimen
#    dari kalimat baru yang diinput oleh pengguna.
#
# Cara Menjalankan:
# 1. Pastikan semua library yang dibutuhkan terinstal (lihat bagian Imports).
#    Script ini akan mencoba menginstal 'transformers' dan 'sastrawi' jika belum ada.
# 2. Jalankan dari terminal: python sentiment_analysis_pipeline.py
#
# Catatan:
# Pada saat pertama kali dijalankan, script akan melakukan training K-Fold yang
# mungkin memakan waktu cukup lama. Pada eksekusi selanjutnya, ia akan
# otomatis menggunakan model yang sudah disimpan.
# ==============================================================================

# ==============================================================================
# BAGIAN 1: IMPORTS LIBRARY
# ==============================================================================
import os
import re
import sys
import subprocess
import warnings
from io import StringIO
from pathlib import Path
import joblib # <<< TAMBAHKAN IMPORT INI

# Scientific & Data Handling
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import requests

# Machine Learning - Scikit-Learn
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.utils.class_weight import compute_class_weight

# Machine Learning - TensorFlow & Keras
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Embedding, LSTM, Dense, Dropout, Bidirectional,
    GlobalMaxPooling1D, GlobalAveragePooling1D, Concatenate,
    Input, MultiHeadAttention, LayerNormalization, Add
)
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.utils import to_categorical

# NLP - Transformers (untuk Tokenizer)
def setup_transformers():
    """Setup transformers library dengan auto-install jika diperlukan."""
    try:
        import transformers
        print(f"Transformers version: {transformers.__version__}")
        return transformers
    except ImportError:
        print("⏳ Installing transformers library...")
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-q", "transformers[sentencepiece]"
        ])
        import transformers
        print(f"✅ Transformers version: {transformers.__version__} installed and loaded.")
        return transformers

transformers = setup_transformers()
from transformers import BertTokenizerFast

# Mengabaikan warning yang tidak kritikal
warnings.filterwarnings('ignore')

print(f"TensorFlow version: {tf.__version__}")
print("✅ Semua library berhasil dimuat!")


# ==============================================================================
# BAGIAN 2: DEFINISI KELAS-KELAS HELPER
# ==============================================================================

class DatasetManager:
    """
    Kelas ini bertanggung jawab untuk mengelola dan memuat dataset dari berbagai sumber.
    Saat ini, ia dikonfigurasi untuk menarik dataset dari repositori GitHub.
    """
    
    def __init__(self):
        self.available_datasets = {
            'komentar_instagram_cyberbullying': {
                'url': 'https://raw.githubusercontent.com/rizalespe/Dataset-Sentimen-Analisis-Bahasa-Indonesia/master/dataset_komentar_instagram_cyberbullying.csv'
            },
            'tweet_sentimen_tayangan_tv': {
                'url': 'https://raw.githubusercontent.com/rizalespe/Dataset-Sentimen-Analisis-Bahasa-Indonesia/master/dataset_tweet_sentimen_tayangan_tv.csv'
            },
        }

    def load_dataset_from_github(self, dataset_name: str) -> pd.DataFrame | None:
        """Memuat dataset dari URL GitHub dan menstandarkan nama kolom."""
        if dataset_name not in self.available_datasets:
            print(f"❌ Dataset {dataset_name} tidak tersedia.")
            return None
            
        try:
            url = self.available_datasets[dataset_name]['url']
            print(f"🔄 Mencoba memuat dataset: {dataset_name} dari {url}")
            
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            
            df = pd.read_csv(StringIO(response.text))
            print(f"✅ Berhasil memuat data mentah untuk {dataset_name}.")

            # Standarisasi nama kolom
            df = self._standardize_columns(df, dataset_name)
            
            if df is not None and 'text' in df.columns and 'sentiment' in df.columns:
                return df[['text', 'sentiment']]
            else:
                print(f"⚠️ Kolom 'text' atau 'sentiment' tidak ditemukan untuk {dataset_name}")
                return None

        except requests.exceptions.RequestException as e:
            print(f"❌ Error jaringan saat memuat {dataset_name}: {e}")
            return None
        except Exception as e:
            print(f"❌ Error tidak terduga saat memuat {dataset_name}: {e}")
            return None

    def _standardize_columns(self, df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
        """Standarisasi nama kolom berdasarkan dataset."""
        column_mapping = {
            'komentar_instagram_cyberbullying': {
                'Instagram Comment Text': 'text', 
                'Sentiment': 'sentiment'
            },
            'tweet_sentimen_tayangan_tv': {
                'Text Tweet': 'text', 
                'Sentiment': 'sentiment'
            }
        }
        
        if dataset_name in column_mapping:
            return df.rename(columns=column_mapping[dataset_name])
        return df


class AdvancedTextPreprocessor:
    """
    Kelas untuk melakukan semua langkah preprocessing teks tingkat lanjut,
    termasuk normalisasi slang, stemming, stopword removal, dan penanganan negasi.
    """
    
    def __init__(self):
        self.stemmer = None
        self.stop_words = set()
        self.negation_words = {
            'tidak', 'bukan', 'jangan', 'belum', 'tanpa', 'anti', 
            'ga', 'gak', 'nggak'
        }
        self.slang_dict = self._load_slang_dictionary()
        self.setup_indonesian_tools()

    def setup_indonesian_tools(self):
        """Menginstal dan memuat library NLP Bahasa Indonesia (Sastrawi & NLTK)."""
        try:
            self._setup_sastrawi()
            self._setup_nltk()
            self._configure_stopwords()
            print("✅ Peralatan NLP Bahasa Indonesia berhasil dimuat.")
        except Exception as e:
            print(f"⚠️ Peringatan: Gagal memuat peralatan NLP: {e}. "
                  "Stemming/Stopword mungkin terpengaruh.")

    def _setup_sastrawi(self):
        """Setup Sastrawi stemmer."""
        try:
            __import__('Sastrawi')
        except ImportError:
            print("⏳ Menginstal Sastrawi...")
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", "-q", "Sastrawi"
            ])
        
        from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
        self.stemmer = StemmerFactory().create_stemmer()

    def _setup_nltk(self):
        """Setup NLTK stopwords."""
        import nltk
        try:
            nltk.data.find('corpora/stopwords.zip')
        except LookupError:
            print("⏳ Mengunduh korpus NLTK 'stopwords'...")
            nltk.download('stopwords', quiet=True)
        
        from nltk.corpus import stopwords
        self.stop_words = set(stopwords.words('indonesian'))

    def _configure_stopwords(self):
        """Konfigurasi stopwords dengan menghapus kata negasi."""
        for neg_word in self.negation_words:
            self.stop_words.discard(neg_word)

    def _load_slang_dictionary(self) -> dict:
        """Kamus untuk normalisasi kata-kata slang."""
        return {
            'gue': 'saya', 'gw': 'saya', 'lu': 'kamu', 'lo': 'kamu',
            'yg': 'yang', 'dgn': 'dengan', 'ga': 'tidak', 'gak': 'tidak',
            'nggak': 'tidak', 'bgt': 'banget', 'gmn': 'bagaimana'
        }

    def clean_text(self, text: str) -> str:
        """Pipeline pembersihan teks lengkap."""
        if not isinstance(text, str):
            return ""
        
        # Normalisasi dasar
        text = text.lower()
        text = self._remove_urls_and_mentions(text)
        text = self._normalize_slang(text)
        text = self._handle_negation(text)
        text = self._remove_punctuation(text)
        text = self._remove_stopwords_and_stem(text)
        
        return text.strip()

    def _remove_urls_and_mentions(self, text: str) -> str:
        """Menghapus URL, email, dan mention."""
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        text = re.sub(r'\S*@\S*\s?', '', text)
        text = re.sub(r'@\w+|#\w+', '', text)
        return text

    def _normalize_slang(self, text: str) -> str:
        """Normalisasi kata slang."""
        words = text.split()
        normalized_words = [self.slang_dict.get(word, word) for word in words]
        return ' '.join(normalized_words)

    def _handle_negation(self, text: str) -> str:
        """Penanganan negasi (misal: 'tidak bagus' -> 'NOT_bagus')."""
        words = text.split()
        processed_words = []
        i = 0
        
        while i < len(words):
            if words[i] in self.negation_words and i + 1 < len(words):
                processed_words.append(f"NOT_{words[i+1]}")
                i += 2
            else:
                processed_words.append(words[i])
                i += 1
                
        return ' '.join(processed_words)

    def _remove_punctuation(self, text: str) -> str:
        """Menghapus tanda baca."""
        return re.sub(r'[^\w\s_]', '', text) # Membiarkan underscore untuk NOT_

    def _remove_stopwords_and_stem(self, text: str) -> str:
        """Menghapus stopwords dan melakukan stemming."""
        words = text.split()
        final_words = []
        
        for word in words:
            if word.startswith("NOT_"):
                # Jangan hapus kata negasi dari stopword removal
                original_word = word[4:]
                stemmed_word = (self.stemmer.stem(original_word) 
                                if self.stemmer else original_word)
                final_words.append(f"NOT_{stemmed_word}")
            elif word not in self.stop_words:
                stemmed_word = (self.stemmer.stem(word) 
                                if self.stemmer else word)
                final_words.append(stemmed_word)
        
        return ' '.join(final_words)


class ModelArchitectures:
    """
    Kelas statis yang berisi berbagai arsitektur model.
    Memudahkan untuk mencoba model yang berbeda di masa depan.
    """
    
    @staticmethod
    def create_lstm_attention_model(vocab_size: int, embedding_dim: int, 
                                    input_length: int, num_classes: int,
                                    l2_reg: float, dropout_val: float) -> Model:
        """Membuat model Bi-LSTM dengan Mekanisme Attention."""
        inputs = Input(shape=(input_length,))
        
        # Embedding layer
        embedding_layer = Embedding(
        vocab_size, embedding_dim,
        input_length=input_length,
        mask_zero=True,
        embeddings_regularizer=l2(l2_reg) # <<< TAMBAHKAN INI
        )(inputs)
        
        # Bidirectional LSTM
        lstm_out = Bidirectional(LSTM(
            64, return_sequences=True, 
            kernel_regularizer=l2(l2_reg)
        ))(embedding_layer)
        
        # Multi-head attention
        # Pastikan dimensi key_dim * num_heads adalah kelipatan dari dimensi lstm_out (64*2=128)
        # Jika key_dim = 32 dan num_heads = 4, maka 32*4 = 128. Ini cocok.
        attention = MultiHeadAttention(
            num_heads=4, key_dim=32 
        )(lstm_out, lstm_out) # Query, Value, Key (Q, V, K)
        
        # Add & Norm (Residual connection)
        attention_norm = Add()([lstm_out, attention])
        attention_norm = LayerNormalization()(attention_norm)
        
        # Global pooling
        global_avg_pool = GlobalAveragePooling1D()(attention_norm)
        global_max_pool = GlobalMaxPooling1D()(attention_norm)
        concat_pool = Concatenate()([global_avg_pool, global_max_pool])
        
        # Dense layers
        dense1 = Dense(
            128, activation='relu', 
            kernel_regularizer=l2(l2_reg)
        )(concat_pool)
        dropout1 = Dropout(dropout_val)(dense1)
        outputs = Dense(num_classes, activation='softmax')(dropout1)
        
        model = Model(
            inputs=inputs, outputs=outputs,
            name=f"LSTM_Attention_L2_{l2_reg}_Drop_{dropout_val}"
        )
        return model


class SentimentAnalysisTrainer:
    """
    Kelas utama yang mengatur seluruh pipeline, mulai dari data hingga training dan evaluasi.
    """
    
    def __init__(self, max_len: int = 150, embedding_dim: int = 128, 
                 bert_tokenizer_name: str = 'indobenchmark/indobert-base-p1',
                 model_dir: str = "model"): # <<< TAMBAHKAN model_dir
        self.max_len = max_len
        self.embedding_dim = embedding_dim
        self.bert_tokenizer_name = bert_tokenizer_name
        self.preprocessor = AdvancedTextPreprocessor()
        self.dataset_manager = DatasetManager()
        self.tokenizer: BertTokenizerFast | None = None
        self.label_encoder: LabelEncoder | None = None
        self.model: Model | None = None
        
        # Path untuk menyimpan/memuat tokenizer dan label encoder
        self.model_dir = model_dir
        self.tokenizer_path = os.path.join(self.model_dir, "tokenizer")
        self.label_encoder_path = os.path.join(self.model_dir, "label_encoder.joblib")

    def init_tokenizer_and_encoder(self, load_existing: bool = False): # <<< TAMBAHKAN parameter
        """
        Menginisialisasi tokenizer dan label encoder.
        Jika load_existing True, coba muat dari file.
        """
        if load_existing and os.path.exists(self.tokenizer_path) and os.path.exists(self.label_encoder_path):
            print(f"🔄 Memuat tokenizer dari: {self.tokenizer_path}")
            self.tokenizer = BertTokenizerFast.from_pretrained(self.tokenizer_path)
            print(f"✅ Tokenizer berhasil dimuat.")
            
            print(f"🔄 Memuat label encoder dari: {self.label_encoder_path}")
            self.label_encoder = joblib.load(self.label_encoder_path)
            print(f"✅ Label encoder berhasil dimuat.")
        else:
            try:
                print(f"🔄 Menginisialisasi BERT tokenizer baru: {self.bert_tokenizer_name}")
                self.tokenizer = BertTokenizerFast.from_pretrained(self.bert_tokenizer_name)
                print(f"✅ BERT tokenizer '{self.bert_tokenizer_name}' berhasil dimuat.")
            except Exception as e:
                print(f"❌ Gagal memuat BERT tokenizer: {e}")
                raise
            
            print("🔄 Menginisialisasi LabelEncoder baru.")
            self.label_encoder = LabelEncoder()
            print("✅ LabelEncoder baru berhasil diinisialisasi.")

    def save_tokenizer_and_encoder(self):
        """Menyimpan tokenizer dan label encoder ke file."""
        if self.tokenizer:
            print(f"💾 Menyimpan tokenizer ke: {self.tokenizer_path}")
            self.tokenizer.save_pretrained(self.tokenizer_path)
            print("✅ Tokenizer berhasil disimpan.")
        else:
            print("⚠️ Tokenizer belum diinisialisasi, tidak dapat menyimpan.")

        if self.label_encoder:
            # Pastikan label encoder sudah di-fit sebelum disimpan
            if not hasattr(self.label_encoder, 'classes_') or self.label_encoder.classes_ is None:
                 print("⚠️ LabelEncoder belum di-fit. Harap fit terlebih dahulu sebelum menyimpan.")
            else:
                print(f"💾 Menyimpan label encoder ke: {self.label_encoder_path}")
                joblib.dump(self.label_encoder, self.label_encoder_path)
                print("✅ Label encoder berhasil disimpan.")
        else:
            print("⚠️ LabelEncoder belum diinisialisasi, tidak dapat menyimpan.")


    def preprocess_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Melakukan preprocessing pada seluruh dataframe."""
        print("🔄 Melakukan preprocessing pada DataFrame...")
        
        # Pembersihan data dasar
        df = self._clean_dataframe(df)
        
        # Normalisasi label sentimen
        df = self._normalize_sentiment_labels(df)
        
        print(f"📊 Distribusi sentimen final: {df['sentiment'].value_counts().to_dict()}")
        
        # Preprocessing teks
        print("   Menerapkan pembersihan teks tingkat lanjut...")
        df['cleaned_text'] = df['text'].apply(self.preprocessor.clean_text)
        
        # Filter teks yang terlalu pendek
        df = df[df['cleaned_text'].str.strip().str.len() >= 3]
        
        return df[['cleaned_text', 'sentiment']]

    def _clean_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Pembersihan dasar dataframe."""
        df.dropna(subset=['text', 'sentiment'], inplace=True)
        df['text'] = df['text'].astype(str)
        df = df[df['text'].str.strip() != '']
        return df

    def _normalize_sentiment_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalisasi label sentimen."""
        sentiment_mapping = {
            'positive': 'positif', 'positif': 'positif',
            'negative': 'negatif', 'negatif': 'negatif'
        }
        df['sentiment'] = (df['sentiment'].astype(str).str.lower()
                           .map(sentiment_mapping))
        df.dropna(subset=['sentiment'], inplace=True)
        return df

    def create_model(self, vocab_size: int, num_classes: int, 
                     l2_reg_val: float, dropout_val: float, 
                     learning_rate: float) -> Model:
        """Membuat dan meng-compile model."""
        print(f"🔄 Membuat model dengan L2: {l2_reg_val}, "
              f"Dropout: {dropout_val}, LR: {learning_rate}...")
        
        model = ModelArchitectures.create_lstm_attention_model(
            vocab_size, self.embedding_dim, self.max_len, num_classes,
            l2_reg=l2_reg_val, dropout_val=dropout_val
        )
        
        optimizer = Adam(learning_rate=learning_rate)
        model.compile(
            optimizer=optimizer, 
            loss='categorical_crossentropy', 
            metrics=['accuracy']
        )
        
        print(f"✅ Model '{model.name}' berhasil dibuat dan di-compile.")
        return model


# ==============================================================================
# BAGIAN 3: FUNGSI-FUNGSI UTAMA
# ==============================================================================

def run_kfold_cross_validation(trainer: SentimentAnalysisTrainer, 
                               dataset_path_or_key: str, n_splits: int, 
                               epochs: int, batch_size: int, learning_rate: float, 
                               l2_reg: float, dropout: float) -> str:
    """
    Menjalankan pipeline training lengkap menggunakan K-Fold Cross-Validation.
    Mengembalikan path dari model dengan performa terbaik di antara semua fold.
    """
    print("=" * 70)
    print(f"🚀 MEMULAI {n_splits}-FOLD CROSS-VALIDATION PIPELINE")
    print("=" * 70)

    # Load dan preprocess data
    df = _load_dataset(trainer, dataset_path_or_key)
    if df is None or df.empty:
        print("❌ Pipeline dihentikan: Gagal memuat data.")
        return ""

    df_processed = trainer.preprocess_dataframe(df)
    if df_processed.empty:
        print("❌ Pipeline dihentikan: Tidak ada data setelah preprocessing.")
        return ""
        
    # Pastikan tokenizer dan encoder diinisialisasi SEBELUM digunakan untuk fit/transform
    # Jika belum ada (misalnya, training pertama kali), ini akan menginisialisasi yang baru.
    if trainer.tokenizer is None or trainer.label_encoder is None:
         trainer.init_tokenizer_and_encoder(load_existing=False) # Selalu buat baru untuk training K-Fold awal

    # Tokenisasi dan encoding
    X_sequences, vocab_size = _prepare_sequences(trainer, df_processed)
    y_categorical, y_indices_for_split, num_classes = _prepare_labels(
        trainer, df_processed
    )
    
    # K-Fold training
    best_model_path_fold = _run_kfold_training(
        trainer, X_sequences, y_categorical, y_indices_for_split,
        vocab_size, num_classes, n_splits, epochs, batch_size,
        learning_rate, l2_reg, dropout
    )

    return best_model_path_fold


def _load_dataset(trainer: SentimentAnalysisTrainer, 
                  dataset_path_or_key: str) -> pd.DataFrame | None:
    """Load dataset dari file atau GitHub."""
    if os.path.exists(dataset_path_or_key):
        try:
            return pd.read_csv(dataset_path_or_key)
        except Exception as e:
            print(f"❌ Gagal memuat dataset lokal dari '{dataset_path_or_key}': {e}")
            # Coba dari GitHub sebagai fallback jika key cocok
            if dataset_path_or_key in trainer.dataset_manager.available_datasets:
                 print(f"🔄 Mencoba memuat '{dataset_path_or_key}' dari GitHub sebagai gantinya...")
                 return trainer.dataset_manager.load_dataset_from_github(dataset_path_or_key)
            return None
    else:
        return trainer.dataset_manager.load_dataset_from_github(dataset_path_or_key)


def _prepare_sequences(trainer: SentimentAnalysisTrainer, 
                       df_processed: pd.DataFrame) -> tuple[np.ndarray, int]:
    """Prepare tokenized sequences."""
    if trainer.tokenizer is None:
        raise ValueError("Tokenizer belum diinisialisasi.")
        
    X_sequences = trainer.tokenizer.batch_encode_plus(
        df_processed['cleaned_text'].tolist(),
        add_special_tokens=True, 
        max_length=trainer.max_len, 
        padding='max_length', 
        truncation=True
    )['input_ids']
    
    X_sequences = np.array(X_sequences)
    vocab_size = trainer.tokenizer.vocab_size
    
    return X_sequences, vocab_size


def _prepare_labels(trainer: SentimentAnalysisTrainer, 
                    df_processed: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, int]:
    """Prepare categorical labels."""
    if trainer.label_encoder is None:
        raise ValueError("LabelEncoder belum diinisialisasi.")
        
    # Fit LabelEncoder jika belum di-fit atau jika classes_ tidak ada
    # Ini penting karena LabelEncoder perlu tahu semua kelas unik dari data training
    if not hasattr(trainer.label_encoder, 'classes_') or trainer.label_encoder.classes_ is None:
        print("ℹ️ Melakukan fit pada LabelEncoder dengan data sentimen...")
        trainer.label_encoder.fit(df_processed['sentiment'].values)
        print(f"📚 Kelas yang terdeteksi oleh LabelEncoder: {trainer.label_encoder.classes_}")

    y_transformed = trainer.label_encoder.transform(df_processed['sentiment'].values)
    y_categorical = to_categorical(y_transformed)
    
    y_indices_for_split = np.argmax(y_categorical, axis=1) # Untuk StratifiedKFold
    num_classes = y_categorical.shape[1]
    
    if num_classes <= 1:
        raise ValueError(f"Hanya ada {num_classes} kelas unik yang terdeteksi. Perlu minimal 2 kelas untuk klasifikasi.")

    return y_categorical, y_indices_for_split, num_classes


def _run_kfold_training(trainer: SentimentAnalysisTrainer, 
                        X_sequences: np.ndarray, y_categorical: np.ndarray,
                        y_indices_for_split: np.ndarray, vocab_size: int,
                        num_classes: int, n_splits: int, epochs: int,
                        batch_size: int, learning_rate: float,
                        l2_reg: float, dropout: float) -> str:
    """Run K-Fold training loop."""
    kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_scores = []
    best_fold_accuracy = -1
    best_model_path_fold = "" # Path model dari fold terbaik

    # Pastikan direktori model ada untuk menyimpan model fold sementara
    os.makedirs(trainer.model_dir, exist_ok=True)

    for fold_num, (train_indices, val_indices) in enumerate(
        kfold.split(X_sequences, y_indices_for_split), 1
    ):
        print(f"\n{'=' * 30} FOLD {fold_num}/{n_splits} {'=' * 30}")
        
        model = trainer.create_model(
            vocab_size, num_classes, l2_reg, dropout, learning_rate
        )
        
        X_train, X_val = X_sequences[train_indices], X_sequences[val_indices]
        y_train, y_val = y_categorical[train_indices], y_categorical[val_indices]
        
        # Simpan model fold di dalam direktori model
        model_save_path_fold = os.path.join(trainer.model_dir, f"sentiment_model_fold_{fold_num}.keras")
        
        callbacks = [
            EarlyStopping(
                monitor='val_accuracy', patience=5, # Atau monitor 'val_loss', mode='min'
                restore_best_weights=True, verbose=1
            ),
            ModelCheckpoint(
                model_save_path_fold, monitor='val_accuracy', # Atau monitor 'val_loss', mode='min'
                save_best_only=True, verbose=1
            ),
            ReduceLROnPlateau( # <<< TAMBAHKAN ATAU PASTIKAN INI ADA
                monitor='val_loss', # Lebih sensitif daripada val_accuracy
                factor=0.2,         # Kurangi LR sebesar 80% (1 - 0.2)
                patience=2,         # Jika val_loss tidak membaik selama 2 epoch
                min_lr=1e-6,        # Batas bawah learning rate
                verbose=1
            )
        ]

        model.fit(
            X_train, y_train, 
            validation_data=(X_val, y_val),
            epochs=epochs, 
            batch_size=batch_size, 
            callbacks=callbacks, 
            verbose=1
        )

        # Evaluasi
        # Model terbaik fold sudah otomatis dimuat oleh ModelCheckpoint dan EarlyStopping (restore_best_weights=True)
        # Jadi, kita bisa langsung evaluasi
        loss, accuracy = model.evaluate(X_val, y_val, verbose=0)
        print(f"--- Evaluasi Fold {fold_num}: Akurasi = {accuracy:.4f} ---")
        fold_scores.append(accuracy)

        if accuracy > best_fold_accuracy:
            best_fold_accuracy = accuracy
            best_model_path_fold = model_save_path_fold # Simpan path dari model .keras fold ini
            print(f"✨ Akurasi terbaik baru di fold {fold_num}: {accuracy:.4f}. Model disimpan di {best_model_path_fold}")


    _print_kfold_summary(fold_scores, best_model_path_fold, best_fold_accuracy)
    return best_model_path_fold # Kembalikan path dari file .keras terbaik


def _print_kfold_summary(fold_scores: list, best_model_path_fold: str, 
                         best_fold_accuracy: float):
    """Print K-Fold training summary."""
    print("\n" + "=" * 70)
    print("📊 RINGKASAN K-FOLD CROSS-VALIDATION")
    print(f"Skor setiap fold: {[f'{s:.4f}' for s in fold_scores]}")
    print(f"Akurasi Rata-rata: {np.mean(fold_scores):.4f} "
          f"(± {np.std(fold_scores):.4f})")
    if best_model_path_fold:
        print(f"🏆 Model terbaik dari K-Fold disimpan sementara di: {best_model_path_fold} "
              f"dengan akurasi {best_fold_accuracy:.4f}")
    else:
        print("⚠️ Tidak ada model yang mencapai performa memuaskan selama K-Fold.")
    print("=" * 70)


def predict_sentiment(text_input: str, model: Model, 
                      trainer: SentimentAnalysisTrainer) -> tuple[str, float]:
    """
    Fungsi untuk memprediksi sentimen dari sebuah teks input.
    Mengembalikan label prediksi dan tingkat kepercayaannya.
    """
    if not isinstance(text_input, str) or not text_input.strip():
        return "Input tidak valid", 0.0

    if trainer.tokenizer is None or trainer.label_encoder is None:
        print("❌ Tokenizer atau LabelEncoder belum dimuat/diinisialisasi. Prediksi tidak bisa dilakukan.")
        return "Error: Komponen model tidak siap", 0.0
    if not hasattr(trainer.label_encoder, 'classes_') or trainer.label_encoder.classes_ is None:
        print("❌ LabelEncoder belum di-fit. Tidak bisa melakukan inverse_transform.")
        return "Error: LabelEncoder tidak siap", 0.0

    # Preprocessing dan tokenisasi
    cleaned_text = trainer.preprocessor.clean_text(text_input)
    if not cleaned_text: # Jika teks menjadi kosong setelah preprocessing
        print("⚠️ Teks input menjadi kosong setelah preprocessing.")
        return "Teks tidak bermakna", 0.0
        
    encoded_input = trainer.tokenizer.batch_encode_plus(
        [cleaned_text], 
        add_special_tokens=True, 
        max_length=trainer.max_len,
        padding='max_length', 
        truncation=True,
        return_tensors='tf' # <<< Kembalikan sebagai TensorFlow tensor
    )['input_ids']
    
    # Prediksi
    # padded_sequence = np.array(encoded_input) # Tidak perlu jika sudah return_tensors='tf'
    prediction_probs = model.predict(encoded_input, verbose=0)[0]
    
    predicted_index = np.argmax(prediction_probs)
    
    try:
        predicted_label = trainer.label_encoder.inverse_transform([predicted_index])[0]
    except ValueError as e:
        print(f"❌ Error saat inverse_transform: {e}. Indeks prediksi: {predicted_index}, Kelas LabelEncoder: {trainer.label_encoder.classes_}")
        return "Error decoding label", 0.0

    confidence = prediction_probs[predicted_index]
    
    return predicted_label, confidence


# ==============================================================================
# BAGIAN 4: BLOK EKSEKUSI UTAMA
# ==============================================================================

def main():
    """Fungsi utama yang menjalankan seluruh logika script."""
    
    # Konfigurasi
    DATASET_PATH_OR_KEY = "data/sentiment_dataset.csv" # Atau nama dataset dari DatasetManager
    # DATASET_PATH_OR_KEY = 'komentar_instagram_cyberbullying' # Contoh jika pakai dari GitHub
    
    MODEL_DIR = "model_artefacts" # <<< Ganti nama direktori agar lebih jelas
    BEST_MODEL_KERAS_PATH = os.path.join(MODEL_DIR, "best_sentiment_model.keras")
    # Path untuk tokenizer dan label encoder akan dikelola oleh SentimentAnalysisTrainer

    # Buat direktori model jika belum ada
    os.makedirs(MODEL_DIR, exist_ok=True)

    # Inisialisasi
    trainer = SentimentAnalysisTrainer(model_dir=MODEL_DIR) # <<< Pass MODEL_DIR
    
    # Load atau train model
    # init_tokenizer_and_encoder akan dipanggil di dalam _load_or_train_model
    loaded_model = _load_or_train_model(trainer, DATASET_PATH_OR_KEY, BEST_MODEL_KERAS_PATH)
    
    if loaded_model is None:
        print("❌ Gagal memuat atau melatih model. Program berhenti.")
        return
    
    # Mode inferensi interaktif
    _run_interactive_mode(loaded_model, trainer)


def _load_or_train_model(trainer: SentimentAnalysisTrainer, 
                         dataset_path_or_key: str, 
                         best_model_keras_path: str) -> Model | None:
    """Load existing model beserta tokenizer dan label encoder, atau train model baru."""
    
    # Cek apakah model, tokenizer, dan label encoder SUDAH ADA
    keras_model_exists = os.path.exists(best_model_keras_path)
    tokenizer_exists = os.path.isdir(trainer.tokenizer_path) # save_pretrained membuat direktori
    label_encoder_exists = os.path.exists(trainer.label_encoder_path)

    if keras_model_exists and tokenizer_exists and label_encoder_exists:
        print(f"\n✅ Model, Tokenizer, dan LabelEncoder sudah ada. Memuat...")
        try:
            # Muat tokenizer dan label encoder DULU
            trainer.init_tokenizer_and_encoder(load_existing=True)
            
            # Kemudian muat model Keras
            print(f"🔄 Memuat model Keras dari '{best_model_keras_path}'...")
            loaded_model = tf.keras.models.load_model(best_model_keras_path)
            print("✅ Model Keras berhasil dimuat.")
            
            # Verifikasi LabelEncoder setelah dimuat (opsional tapi bagus untuk debug)
            if trainer.label_encoder and hasattr(trainer.label_encoder, 'classes_'):
                print(f"ℹ️ Kelas yang diketahui LabelEncoder (setelah dimuat): {trainer.label_encoder.classes_}")
            else:
                 print("⚠️ LabelEncoder tidak dimuat dengan benar atau tidak memiliki atribut 'classes_'.")
                 return None # Gagal jika komponen penting tidak ada
            return loaded_model
        except Exception as e:
            print(f"❌ Gagal memuat model atau komponen yang sudah ada: {e}")
            print("🤔 Mencoba melatih model baru sebagai gantinya...")
            # Lanjutkan ke blok training jika ada error saat memuat
            pass # Jatuh ke blok else untuk training
    
    # Jika salah satu tidak ada, atau ada error saat memuat, lakukan training
    print(f"\n❌ Model lengkap (model, tokenizer, atau encoder) tidak ditemukan atau gagal dimuat. Memulai proses training...")
    # Pastikan tokenizer dan encoder diinisialisasi SEGAR untuk training baru
    trainer.init_tokenizer_and_encoder(load_existing=False) 
    return _train_new_model(trainer, dataset_path_or_key, best_model_keras_path)


def _train_new_model(trainer: SentimentAnalysisTrainer, 
                     dataset_path_or_key: str, 
                     best_model_keras_path: str) -> Model | None:
    """Train model baru dengan K-Fold CV dan simpan semua artefak."""
    
    # Jalankan K-Fold CV, ini akan mengembalikan path ke model .keras TERBAIK dari semua fold
    path_keras_from_kfold = run_kfold_cross_validation(
        trainer=trainer,
        dataset_path_or_key=dataset_path_or_key,
        n_splits=5, # Kurangi untuk testing cepat, idealnya 5 atau 10
        epochs=20,  # Kurangi untuk testing cepat, idealnya >= 20
        batch_size=32,
        learning_rate=0.0001,
        l2_reg=0.005, # Nilai L2 yang lebih kecil mungkin lebih baik
        dropout=0.5,   # Nilai dropout yang lebih kecil mungkin lebih baik
    )
    
    if path_keras_from_kfold and os.path.exists(path_keras_from_kfold):
        # Pindahkan/rename model .keras terbaik dari K-Fold ke path final
        # Ini adalah model .keras yang sebenarnya, bukan path dari ModelCheckpoint
        # Jika path_keras_from_kfold sudah merupakan path akhir, rename tidak diperlukan
        # atau bisa juga langsung load dari path_keras_from_kfold
        
        # Muat model terbaik dari path yang dikembalikan K-Fold untuk memastikan kita punya objek modelnya
        print(f"🔄 Memuat model terbaik dari K-Fold: {path_keras_from_kfold}")
        final_model = tf.keras.models.load_model(path_keras_from_kfold)

        # Simpan model ini ke lokasi BEST_MODEL_KERAS_PATH jika berbeda
        if path_keras_from_kfold != best_model_keras_path:
             print(f"💾 Menyimpan model Keras final ke: {best_model_keras_path}")
             final_model.save(best_model_keras_path)
        else:
            print(f"✅ Model Keras terbaik sudah berada di: {best_model_keras_path}")
        
        # Sekarang simpan tokenizer dan label encoder yang sudah di-fit
        # LabelEncoder seharusnya sudah di-fit di dalam _prepare_labels
        # Tokenizer diinisialisasi di awal.
        trainer.save_tokenizer_and_encoder()
        
        print(f"✅ Model, Tokenizer, dan LabelEncoder telah disimpan di direktori '{trainer.model_dir}'")
        return final_model
    else:
        print("❌ Training K-Fold gagal atau tidak menghasilkan model terbaik. Tidak ada model yang bisa dimuat.")
        return None


def _run_interactive_mode(loaded_model: Model, trainer: SentimentAnalysisTrainer):
    """Menjalankan mode prediksi interaktif."""
    print("\n" + "=" * 50)
    print("🚀 MEMULAI MODE PREDIKSI SENTIMEN INTERAKTIF")
    print("   Ketik 'exit' atau 'keluar' untuk berhenti.")
    print("=" * 50)

    while True:
        try:
            user_input = input("\nKetik sebuah kalimat: ")
            if user_input.lower() in ['exit', 'keluar']:
                print("👋 Terima kasih telah mencoba. Sampai jumpa!")
                break
            
            prediction, confidence = predict_sentiment(
                user_input, loaded_model, trainer
            )
            if prediction == "Error: Komponen model tidak siap" or prediction == "Error: LabelEncoder tidak siap":
                print(f"❌ {prediction}")
                print("Program berhenti karena komponen model tidak siap.")
                break
            
            print(f"➡️  Prediksi Sentimen: **{str(prediction).upper()}** "
                  f"(Kepercayaan: {confidence:.2%})")

        except KeyboardInterrupt:
            print("\n👋 Program dihentikan oleh pengguna. Sampai jumpa!")
            break
        except Exception as e:
            print(f"❌ Terjadi error saat prediksi interaktif: {e}")


# Menjalankan fungsi main() jika script dieksekusi secara langsung
if __name__ == "__main__":
    main()