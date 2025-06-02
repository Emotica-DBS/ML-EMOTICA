# ==============================================================================
# SENTIMENT ANALYSIS PIPELINE (VERSI FINAL - BINARY SEIMBANG DENGAN EVALUASI)
#
# Deskripsi:
# Script ini melatih model sentimen binary (positif/negatif) dengan dataset
# yang diseimbangkan melalui undersampling.
# Termasuk caching preprocessing, training, penyimpanan artefak, dan evaluasi.
# Untuk prediksi interaktif, gunakan predict.py.
# ==============================================================================

# ==============================================================================
# BAGIAN 1: IMPORTS LIBRARY
# ==============================================================================
import os
import re
import sys
import subprocess
import warnings
import joblib
import time
from functools import wraps

import numpy as np
import pandas as pd

try:
    from tqdm.auto import tqdm
except ImportError:
    print("⏳ Menginstal tqdm untuk progress bar...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "tqdm"])
    from tqdm.auto import tqdm
try:
    import swifter
except ImportError:
    print("⏳ Menginstal Swifter...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "swifter"])
    import swifter
    
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
# class_weight tidak digunakan karena kita menyeimbangkan dataset dengan undersampling
from sklearn.metrics import classification_report, confusion_matrix

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Embedding, LSTM, Dense, Dropout, Bidirectional,
    GlobalMaxPooling1D, GlobalAveragePooling1D, Concatenate,
    Input, MultiHeadAttention, LayerNormalization, Add
)
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, Callback
from tensorflow.keras.utils import to_categorical
from transformers import BertTokenizerFast

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    PLOT_AVAILABLE = True
except ImportError:
    PLOT_AVAILABLE = False
    print("⚠️ matplotlib atau seaborn tidak terinstal. Visualisasi Confusion Matrix akan dilewati.")
    print("   Anda bisa menginstalnya dengan: pip install matplotlib seaborn")

warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
print(f"TensorFlow version: {tf.__version__}")
print("✅ Semua library berhasil dimuat!")

# ==============================================================================
# DECORATOR UNTUK MENGHITUNG WAKTU EKSEKUSI
# ==============================================================================
def time_it(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        print(f"\n⏳ Memulai '{func.__name__}'...")
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"✅ Selesai '{func.__name__}' dalam {end_time - start_time:.2f} detik.")
        return result
    return wrapper

# ==============================================================================
# BAGIAN 2: DEFINISI KELAS-KELAS HELPER
# ==============================================================================
class AdvancedTextPreprocessor:
    def __init__(self):
        self.stemmer = None
        self.stop_words = set()
        self.negation_words = {'tidak', 'bukan', 'jangan', 'belum', 'tanpa', 'anti', 'ga', 'gak', 'nggak'}
        self.slang_dict = self._load_slang_dictionary()
        self.setup_indonesian_tools()

    def setup_indonesian_tools(self):
        try:
            self._setup_sastrawi()
            self._setup_nltk()
            self._configure_stopwords()
            print("✅ Peralatan NLP Bahasa Indonesia berhasil dimuat.")
        except Exception as e:
            print(f"⚠️ Peringatan: Gagal memuat peralatan NLP: {e}. Stemming/Stopword mungkin terpengaruh.")

    def _setup_sastrawi(self):
        try:
            from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
            self.stemmer = StemmerFactory().create_stemmer()
        except ImportError:
            print("⏳ Menginstal Sastrawi...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "Sastrawi"])
            from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
            self.stemmer = StemmerFactory().create_stemmer()

    def _setup_nltk(self):
        try:
            import nltk
            from nltk.corpus import stopwords
            self.stop_words = set(stopwords.words('indonesian'))
        except LookupError:
            print("⏳ Mengunduh korpus NLTK 'stopwords'...")
            import nltk
            nltk.download('stopwords', quiet=True)
            from nltk.corpus import stopwords
            self.stop_words = set(stopwords.words('indonesian'))
        except ImportError:
            print("⏳ Menginstal NLTK...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "nltk"])
            self._setup_nltk()

    def _configure_stopwords(self):
        for neg_word in self.negation_words:
            self.stop_words.discard(neg_word)

    def _load_slang_dictionary(self) -> dict:
        return {'gue': 'saya', 'gw': 'saya', 'lu': 'kamu', 'lo': 'kamu', 'yg': 'yang', 'dgn': 'dengan', 'ga': 'tidak', 'gak': 'tidak', 'nggak': 'tidak', 'bgt': 'banget', 'gmn': 'bagaimana'}

    def clean_text(self, text: str) -> str:
        if not isinstance(text, str): return ""
        text = text.lower()
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        text = re.sub(r'\S*@\S*\s?', '', text)
        text = re.sub(r'@\w+|#\w+', '', text)
        words = text.split()
        normalized_words = [self.slang_dict.get(word, word) for word in words]
        text = ' '.join(normalized_words)
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
        text = ' '.join(processed_words)
        text = re.sub(r'[^\w\s_]', '', text) # Membiarkan underscore untuk NOT_
        words = text.split()
        final_words = []
        for word in words:
            if word.startswith("NOT_"):
                original_word = word[4:]
                stemmed_word = self.stemmer.stem(original_word) if self.stemmer else original_word
                final_words.append(f"NOT_{stemmed_word}")
            elif word not in self.stop_words:
                stemmed_word = self.stemmer.stem(word) if self.stemmer else word
                final_words.append(stemmed_word)
        return ' '.join(final_words).strip()

class ModelArchitectures:
    @staticmethod
    def create_lstm_attention_model(vocab_size: int, embedding_dim: int, input_length: int, num_classes: int, l2_reg: float, dropout_val: float) -> Model:
        inputs = Input(shape=(input_length,))
        embedding_layer = Embedding(vocab_size, embedding_dim, input_length=input_length, mask_zero=True, embeddings_regularizer=l2(l2_reg))(inputs)
        lstm_out = Bidirectional(LSTM(64, return_sequences=True, kernel_regularizer=l2(l2_reg)))(embedding_layer)
        attention = MultiHeadAttention(num_heads=4, key_dim=32)(lstm_out, lstm_out)
        attention_norm = Add()([lstm_out, attention])
        attention_norm = LayerNormalization()(attention_norm)
        global_avg_pool = GlobalAveragePooling1D()(attention_norm)
        global_max_pool = GlobalMaxPooling1D()(attention_norm)
        concat_pool = Concatenate()([global_avg_pool, global_max_pool])
        dense1 = Dense(128, activation='relu', kernel_regularizer=l2(l2_reg))(concat_pool)
        dropout1 = Dropout(dropout_val)(dense1)
        outputs = Dense(num_classes, activation='softmax')(dropout1) # Softmax untuk >1 output neuron
        model = Model(inputs=inputs, outputs=outputs, name=f"LSTM_Attention_L2_{l2_reg}_Drop_{dropout_val}")
        return model

class SentimentAnalysisTrainer:
    def __init__(self, max_len: int = 150, embedding_dim: int = 128, bert_tokenizer_name: str = 'indobenchmark/indobert-base-p1', model_dir: str = "models"): # Model dir default ke "models"
        self.max_len=max_len
        self.embedding_dim=embedding_dim
        self.bert_tokenizer_name=bert_tokenizer_name
        self.preprocessor=AdvancedTextPreprocessor()
        self.tokenizer: BertTokenizerFast | None = None
        self.label_encoder: LabelEncoder | None = None
        self.model_dir=model_dir
        self.tokenizer_path=os.path.join(self.model_dir, "tokenizer")
        self.label_encoder_path=os.path.join(self.model_dir, "label_encoder.joblib")
    
    @time_it
    def init_tokenizer_and_encoder(self, load_existing: bool = False):
        if load_existing and os.path.isdir(self.tokenizer_path) and os.path.exists(self.label_encoder_path):
            try:
                print(f"🔄 Memuat tokenizer dari: {self.tokenizer_path}")
                self.tokenizer = BertTokenizerFast.from_pretrained(self.tokenizer_path)
                print(f"🔄 Memuat label encoder dari: {self.label_encoder_path}")
                self.label_encoder = joblib.load(self.label_encoder_path)
                print(f"✅ Tokenizer dan Label encoder berhasil dimuat.")
                return
            except Exception as e:
                print(f"⚠️ Gagal memuat artefak yang ada: {e}. Akan menginisialisasi yang baru.")
        
        print(f"🔄 Menginisialisasi BERT tokenizer baru dari '{self.bert_tokenizer_name}'...")
        self.tokenizer = BertTokenizerFast.from_pretrained(self.bert_tokenizer_name)
        print("🔄 Menginisialisasi LabelEncoder baru.")
        self.label_encoder = LabelEncoder()
        print("✅ Tokenizer dan LabelEncoder baru berhasil diinisialisasi.")
    
    def save_tokenizer_and_encoder(self):
        if self.tokenizer and self.label_encoder and hasattr(self.label_encoder, 'classes_') and self.label_encoder.classes_ is not None:
            os.makedirs(self.model_dir, exist_ok=True)
            print(f"💾 Menyimpan tokenizer ke: {self.tokenizer_path}")
            self.tokenizer.save_pretrained(self.tokenizer_path)
            print(f"💾 Menyimpan label encoder ke: {self.label_encoder_path}")
            joblib.dump(self.label_encoder, self.label_encoder_path)
            print("✅ Tokenizer dan label encoder berhasil disimpan.")
        else:
            print("⚠️ Tokenizer atau LabelEncoder belum siap/di-fit, tidak dapat menyimpan.")
    
    @time_it
    def preprocess_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        print("🔄 Melakukan preprocessing pada DataFrame (yang mungkin sudah difilter/diseimbangkan)...")
        df.dropna(subset=['text', 'sentiment'], inplace=True)
        df['text'] = df['text'].astype(str)
        df = df[df['text'].str.strip() != '']
        # _normalize_sentiment_labels sekarang dipanggil sebelum undersampling di _load_or_create_cleaned_dataset
        # print(f"📊 Distribusi sentimen yang akan diproses: \n{df['sentiment'].value_counts()}") # Akan dicetak setelah undersampling
        print("   Menerapkan pembersihan teks tingkat lanjut (menggunakan Swifter untuk paralelisasi)...")
        df['cleaned_text'] = df['text'].swifter.apply(self.preprocessor.clean_text)
        df = df[df['cleaned_text'].str.strip().str.len() >= 3].copy()
        return df[['cleaned_text', 'sentiment']]
    
    def _normalize_sentiment_labels(self, df: pd.DataFrame) -> pd.DataFrame: # Untuk memfilter label
        df['sentiment'] = df['sentiment'].astype(str).str.lower()
        valid_labels = ['positif', 'negatif'] # ### HANYA POSITIF DAN NEGATIF ###
        df = df[df['sentiment'].isin(valid_labels)]
        df.dropna(subset=['sentiment'], inplace=True)
        return df
    
    def create_model(self, vocab_size: int, num_classes: int, l2_reg_val: float, dropout_val: float, learning_rate: float) -> Model:
        model = ModelArchitectures.create_lstm_attention_model(vocab_size, self.embedding_dim, self.max_len, num_classes, l2_reg=l2_reg_val, dropout_val=dropout_val)
        optimizer = Adam(learning_rate=learning_rate)
        # Untuk binary classification, jika num_classes=1 (output sigmoid), gunakan binary_crossentropy.
        # Jika num_classes=2 (output softmax), categorical_crossentropy masih valid.
        loss_function = 'binary_crossentropy' if num_classes == 1 else 'categorical_crossentropy'
        model.compile(optimizer=optimizer, loss=loss_function, metrics=['accuracy'])
        print(f"✅ Model '{model.name}' dibuat dengan loss: {loss_function} dan di-compile.")
        return model

class ArtifactSaver(Callback):
    def __init__(self, trainer, model_save_path):
        super().__init__(); self.trainer = trainer; self.model_save_path = model_save_path; self.best_val_accuracy = -1
    def on_epoch_end(self, epoch, logs=None):
        current_val_accuracy = logs.get('val_accuracy')
        if current_val_accuracy is None: return
        if current_val_accuracy > self.best_val_accuracy:
            print(f"\n✨ Akurasi validasi meningkat dari {self.best_val_accuracy:.4f} ke {current_val_accuracy:.4f}.")
            print(f"   Menyimpan semua artefak ke direktori '{self.trainer.model_dir}'...")
            self.best_val_accuracy = current_val_accuracy
            self.model.save(self.model_save_path) # self.model adalah model yang sedang dilatih oleh Keras
            self.trainer.save_tokenizer_and_encoder()
            print("   ✅ Semua artefak (Model, Tokenizer, Encoder) berhasil disimpan.")

# ==============================================================================
# BAGIAN 3: FUNGSI-FUNGSI UTAMA
# ==============================================================================

@time_it
def _load_or_create_cleaned_dataset(raw_dataset_path: str, cleaned_dataset_path: str, trainer: SentimentAnalysisTrainer) -> pd.DataFrame | None:
    df_cleaned = None
    if os.path.exists(cleaned_dataset_path):
        print(f"✅ File dataset yang sudah bersih ditemukan. Memuat dari '{cleaned_dataset_path}'...")
        try: df_cleaned = pd.read_feather(cleaned_dataset_path)
        except Exception as e_feather:
            print(f"⚠️ Gagal memuat file .feather: {e_feather}. Mencoba menghapus dan membuat ulang.")
            try: os.remove(cleaned_dataset_path)
            except OSError as e_remove: print(f" Gagal menghapus file .feather yang korup: {e_remove}")
    else:
        print(f"ℹ️ File cache '{cleaned_dataset_path}' tidak ditemukan.")

    if df_cleaned is None:
        print("   Memulai proses pembuatan file bersih dari data mentah...")
        try: df_raw = pd.read_csv(raw_dataset_path)
        except FileNotFoundError: print(f"❌ KESALAHAN: File dataset mentah tidak ditemukan di '{raw_dataset_path}'"); return None
        if df_raw.empty: print("Dataset mentah kosong."); return None
        print(f"   Berhasil memuat {len(df_raw)} baris data mentah dari '{raw_dataset_path}'.")

        print("🔄 Memfilter data untuk sentimen positif dan negatif saja (menghapus 'netral')...")
        df_raw['sentiment'] = df_raw['sentiment'].astype(str).str.lower()
        # Gunakan fungsi _normalize_sentiment_labels dari trainer untuk konsistensi
        df_raw = trainer._normalize_sentiment_labels(df_raw) 
        df_raw.dropna(subset=['sentiment'], inplace=True) 

        if df_raw.empty: print("❌ Tidak ada data positif atau negatif setelah filtering awal."); return None
        print(f"📊 Distribusi setelah filter awal (hanya positif/negatif): \n{df_raw['sentiment'].value_counts()}")

        print("🔄 Melakukan Undersampling kelas 'positif' agar seimbang dengan 'negatif'...")
        sentiment_counts = df_raw['sentiment'].value_counts()
        
        if 'negatif' not in sentiment_counts or sentiment_counts['negatif'] == 0: print("❌ Kelas 'negatif' tidak ditemukan/kosong. Tidak bisa undersampling."); return None
        if 'positif' not in sentiment_counts or sentiment_counts['positif'] == 0: print("❌ Kelas 'positif' tidak ditemukan/kosong. Tidak bisa undersampling."); return None
            
        count_negatif = sentiment_counts['negatif']
        df_positif = df_raw[df_raw['sentiment'] == 'positif']
        df_negatif = df_raw[df_raw['sentiment'] == 'negatif']

        if len(df_positif) > count_negatif: df_positif_undersampled = df_positif.sample(n=count_negatif, random_state=42)
        else: df_positif_undersampled = df_positif
        
        df_balanced = pd.concat([df_positif_undersampled, df_negatif]).sample(frac=1, random_state=42).reset_index(drop=True)
        
        print(f"   Distribusi setelah undersampling 'positif': \n{df_balanced['sentiment'].value_counts()}")
        print(f"   Total data setelah undersampling: {len(df_balanced)}")
        
        df_cleaned = trainer.preprocess_dataframe(df_balanced)
        
        if df_cleaned is not None and not df_cleaned.empty:
            print(f"💾 Menyimpan dataset yang sudah bersih ke '{cleaned_dataset_path}'...")
            try: df_cleaned.to_feather(cleaned_dataset_path); print("   ✅ Berhasil disimpan dalam format Feather.")
            except Exception as e_save: print(f"⚠️ Gagal menyimpan sebagai .feather: {e_save}.")
    
    if df_cleaned is not None: print(f"✅ Berhasil memuat atau membuat {len(df_cleaned)} baris data yang sudah diproses."); print(f"📊 Distribusi sentimen akhir yang akan digunakan: \n{df_cleaned['sentiment'].value_counts()}")
    return df_cleaned

@time_it
def train_model_and_evaluate(trainer: SentimentAnalysisTrainer, raw_dataset_path: str, cleaned_dataset_path: str, model_save_path: str, epochs: int, batch_size: int, learning_rate: float, l2_reg: float, dropout: float):
    print("=" * 70); print(f"🚀 MEMULAI PIPELINE TRAINING (BINARY POSITIF/NEGATIF SEIMBANG)"); print("=" * 70)
    df_processed = _load_or_create_cleaned_dataset(raw_dataset_path, cleaned_dataset_path, trainer)
    if df_processed is None or df_processed.empty: print("❌ Pipeline dihentikan: Gagal memuat data."); return

    # Untuk tes cepat, aktifkan baris di bawah ini.
    # df_processed = df_processed.sample(n=10000, random_state=42)
    # print(f"⚠️ MENGGUNAKAN SAMPEL SEBANYAK {len(df_processed)} DATA UNTUK DEVELOPMENT CEPAT! ⚠️")

    trainer.init_tokenizer_and_encoder()
    print("🔄 Tokenisasi data dalam potongan (chunks)..."); all_input_ids = []
    chunk_size = 10000
    for i in tqdm(range(0, len(df_processed), chunk_size), desc="Tokenizing data"):
        batch_texts = df_processed['cleaned_text'].iloc[i:i + chunk_size].tolist()
        encoded_batch = trainer.tokenizer.batch_encode_plus(batch_texts, add_special_tokens=True, max_length=trainer.max_len, padding='max_length', truncation=True)['input_ids']
        all_input_ids.extend(encoded_batch)
    X_sequences = np.array(all_input_ids); print(f"✅ Tokenisasi selesai. Shape data X: {X_sequences.shape}")
    vocab_size = trainer.tokenizer.vocab_size

    print("🔄 Encoding label..."); trainer.label_encoder.fit(df_processed['sentiment'].values)
    print(f"📚 Kelas terdeteksi: {trainer.label_encoder.classes_}")
    y_transformed = trainer.label_encoder.transform(df_processed['sentiment'].values)
    # Untuk binary, num_classes akan jadi 2
    num_classes = len(trainer.label_encoder.classes_) 
    y_categorical = to_categorical(y_transformed, num_classes=num_classes)
    print(f"✅ Encoding label selesai. Shape data Y: {y_categorical.shape}. Num classes: {num_classes}")
    
    class_weights = None # Tidak menggunakan class_weight karena data sudah di-balance
    print("ℹ️ Dataset sudah diseimbangkan (binary), class_weight tidak digunakan.")

    print("🔄 Membagi data menjadi set training (80%) dan validasi (20%)...")
    if len(np.unique(y_transformed)) < 2: print("⚠️ Kurang dari 2 kelas unik, stratify tidak bisa digunakan. Melakukan split biasa."); X_train, X_val, y_train, y_val = train_test_split(X_sequences, y_categorical, test_size=0.2, random_state=42, shuffle=True)
    else: X_train, X_val, y_train, y_val = train_test_split(X_sequences, y_categorical, test_size=0.2, random_state=42, stratify=y_transformed)
    print(f"   Data training: {len(X_train)} baris | Data validasi: {len(X_val)} baris")

    if num_classes <= 1: print("❌ Tidak cukup kelas untuk training. Minimal 2 kelas diperlukan."); return

    model = trainer.create_model(vocab_size, num_classes, l2_reg, dropout, learning_rate)
    callbacks = [
        ArtifactSaver(trainer, model_save_path),
        EarlyStopping(monitor='val_accuracy', patience=5, verbose=1, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, min_lr=1e-6, verbose=1)
    ]
    print(f"🚀 Memulai training model untuk {epochs} epochs...")
    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=epochs, batch_size=batch_size, callbacks=callbacks, class_weight=class_weights, verbose=1)

    # Setelah training selesai, langsung evaluasi model terbaik yang disimpan
    if os.path.exists(model_save_path):
        print("\n🔄 Memuat model terbaik yang disimpan untuk evaluasi akhir...")
        loaded_model_for_eval = tf.keras.models.load_model(model_save_path)
        # y_val sudah dalam bentuk categorical, kita butuh label 1D untuk classification_report
        y_val_true_labels = np.argmax(y_val, axis=1)
        
        print("\n📊 Melakukan prediksi pada data validasi untuk evaluasi akhir...")
        y_pred_probs_eval = loaded_model_for_eval.predict(X_val, batch_size=batch_size, verbose=1)
        y_pred_labels_eval = np.argmax(y_pred_probs_eval, axis=1)
        
        print("\n" + "="*30 + " HASIL EVALUASI MODEL AKHIR " + "="*30)
        target_names_eval = trainer.label_encoder.classes_
        print(classification_report(y_val_true_labels, y_pred_labels_eval, target_names=target_names_eval))
        print("\n" + "="*30 + " CONFUSION MATRIX AKHIR " + "="*30)
        cm = confusion_matrix(y_val_true_labels, y_pred_labels_eval)
        print(cm)
        if PLOT_AVAILABLE:
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=target_names_eval, yticklabels=target_names_eval)
            plt.xlabel('Predicted Label'); plt.ylabel('True Label'); plt.title('Confusion Matrix (Binary - Final Model)')
            plt.savefig(os.path.join(trainer.model_dir, "confusion_matrix_final_binary.png"))
            print(f"   Visualisasi Confusion Matrix disimpan sebagai 'confusion_matrix_final_binary.png' di folder '{trainer.model_dir}'.")

# ==============================================================================
# BAGIAN 4: BLOK EKSEKUSI UTAMA
# ==============================================================================
@time_it
def main():
    RAW_DATASET_PATH = "data/sentiment_dataset.csv"
    DATA_DIR = "data"
    # Menggunakan nama file dan folder yang diminta pengguna
    CLEANED_DATASET_PATH = os.path.join(DATA_DIR, "sentiment_dataset_cleaned.feather")
    MODEL_DIR = "models" 
    BEST_MODEL_KERAS_PATH = os.path.join(MODEL_DIR, "best_sentiment_model.keras")

    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(DATA_DIR, exist_ok=True)
    if not os.path.exists(RAW_DATASET_PATH):
        print(f"❌ FATAL: File dataset mentah '{RAW_DATASET_PATH}' tidak ditemukan. Harap sediakan file tersebut."); return

    trainer = SentimentAnalysisTrainer(model_dir=MODEL_DIR) # Menggunakan MODEL_DIR yang sudah ditetapkan
    
    tokenizer_exists = os.path.isdir(trainer.tokenizer_path)
    encoder_exists = os.path.exists(trainer.label_encoder_path)
    model_exists = os.path.exists(BEST_MODEL_KERAS_PATH)

    if not model_exists or not tokenizer_exists or not encoder_exists:
        print(f"⚠️ Model atau artefak pendukung (tokenizer/encoder) tidak ditemukan lengkap. Memulai proses training baru...")
        # Fungsi training sekarang juga menangani evaluasi di akhir
        train_model_and_evaluate(
            trainer=trainer, raw_dataset_path=RAW_DATASET_PATH, cleaned_dataset_path=CLEANED_DATASET_PATH,
            model_save_path=BEST_MODEL_KERAS_PATH, epochs=20, batch_size=32, 
            learning_rate=0.0001, l2_reg=0.005, dropout=0.5
        )
    else: # Jika model dan artefak sudah ada, kita tetap bisa jalankan evaluasi
        print(f"\n✅ Model dan artefak pendukung sudah ada di '{MODEL_DIR}'.")
        print("   Menjalankan evaluasi pada model yang ada...")
        # Panggil fungsi training & evaluasi, tapi ia akan load model dari ArtifactSaver
        # atau jika ingin evaluasi saja:
        try:
            trainer.init_tokenizer_and_encoder(load_existing=True) 
            loaded_model = tf.keras.models.load_model(BEST_MODEL_KERAS_PATH)
            print("   Model berhasil dimuat untuk evaluasi.")

            # Mempersiapkan data validasi untuk evaluasi
            df_for_eval = _load_or_create_cleaned_dataset(RAW_DATASET_PATH, CLEANED_DATASET_PATH, trainer)
            if df_for_eval is not None and not df_for_eval.empty:
                all_input_ids_eval = []
                chunk_size_eval = 10000
                for i in tqdm(range(0, len(df_for_eval), chunk_size_eval), desc="Tokenizing data untuk evaluasi"):
                    batch_texts_eval = df_for_eval['cleaned_text'].iloc[i:i + chunk_size_eval].tolist()
                    encoded_batch_eval = trainer.tokenizer.batch_encode_plus(batch_texts_eval, add_special_tokens=True, max_length=trainer.max_len, padding='max_length', truncation=True)['input_ids']
                    all_input_ids_eval.extend(encoded_batch_eval)
                X_sequences_eval = np.array(all_input_ids_eval)
                
                if not hasattr(trainer.label_encoder, 'classes_') or trainer.label_encoder.classes_ is None:
                    trainer.label_encoder.fit(df_for_eval['sentiment'].values)

                y_transformed_eval = trainer.label_encoder.transform(df_for_eval['sentiment'].values)
                num_classes_eval = len(trainer.label_encoder.classes_)
                y_categorical_eval = to_categorical(y_transformed_eval, num_classes=num_classes_eval)
                
                stratify_labels_eval = y_transformed_eval
                if len(np.unique(stratify_labels_eval)) < 2:
                    _, X_val_eval, _, y_val_eval_cat = train_test_split(X_sequences_eval, y_categorical_eval, test_size=0.2, random_state=42, shuffle=True)
                else: 
                    _, X_val_eval, _, y_val_eval_cat = train_test_split(X_sequences_eval, y_categorical_eval, test_size=0.2, random_state=42, stratify=stratify_labels_eval)
                
                y_val_eval_true_labels = np.argmax(y_val_eval_cat, axis=1)
                
                print("\n📊 Melakukan prediksi pada data validasi untuk evaluasi...")
                y_pred_probs_eval = loaded_model.predict(X_val_eval, batch_size=32, verbose=1)
                y_pred_labels_eval = np.argmax(y_pred_probs_eval, axis=1)
                
                print("\n" + "="*30 + " HASIL EVALUASI MODEL (DARI MODEL YANG ADA) " + "="*30)
                target_names_eval = trainer.label_encoder.classes_
                print(classification_report(y_val_eval_true_labels, y_pred_labels_eval, target_names=target_names_eval))
                print("\n" + "="*30 + " CONFUSION MATRIX (DARI MODEL YANG ADA) " + "="*30)
                cm = confusion_matrix(y_val_eval_true_labels, y_pred_labels_eval)
                print(cm)
                if PLOT_AVAILABLE: 
                    plt.figure(figsize=(8, 6))
                    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=target_names_eval, yticklabels=target_names_eval)
                    plt.xlabel('Predicted Label'); plt.ylabel('True Label'); plt.title('Confusion Matrix (Binary - Existing Model)')
                    plt.savefig(os.path.join(MODEL_DIR, "confusion_matrix_existing_binary.png"))
                    print(f"   Visualisasi Confusion Matrix disimpan sebagai 'confusion_matrix_existing_binary.png' di folder '{MODEL_DIR}'.")
            else: print("⚠️ Tidak bisa melakukan evaluasi karena data bersih tidak tersedia atau kosong.")
        except Exception as e: print(f"❌ Gagal melakukan evaluasi pada model yang ada: {e}")

    print(f"\n✅ Skrip selesai. Artefak tersimpan di '{MODEL_DIR}'. Gunakan 'predict.py' untuk prediksi.")

if __name__ == "__main__":
    main()