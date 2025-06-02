# ==============================================================================
# PREDICTION SCRIPT (VERSI FINAL - BINARY)
#
# Deskripsi:
# Script ini HANYA untuk melakukan prediksi menggunakan model binary
# (positif/negatif) yang sudah dilatih dan diseimbangkan.
# Ia akan memuat model, tokenizer, dan label encoder dari folder 'models'.
# ==============================================================================

import os
import re
import joblib
import numpy as np
import tensorflow as tf
from transformers import BertTokenizerFast
import warnings
import sys
import subprocess # Diperlukan oleh AdvancedTextPreprocessor

warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

# ==============================================================================
# KELAS PREPROCESSOR (DISALIN PERSIS DARI VERSI TRAINING)
# ==============================================================================
class AdvancedTextPreprocessor:
    def __init__(self):
        self.stemmer = None
        self.stop_words = set()
        self.negation_words = {'tidak', 'bukan', 'jangan', 'belum', 'tanpa', 'anti', 'ga', 'gak', 'nggak'}
        self.slang_dict = self._load_slang_dictionary()
        self.setup_indonesian_tools() # Panggil setup tools

    def setup_indonesian_tools(self):
        # Untuk prediktor, kita asumsikan library sudah ada atau bisa auto-install
        # jika memang diperlukan untuk konsistensi dengan training.
        try:
            self._setup_sastrawi()
            self._setup_nltk()
            self._configure_stopwords()
            print("✅ Peralatan NLP Bahasa Indonesia (untuk preprocessor) berhasil dimuat/dicek.")
        except Exception as e:
            print(f"⚠️ Peringatan: Gagal memuat peralatan NLP untuk preprocessor: {e}.")

    def _setup_sastrawi(self):
        try:
            from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
            self.stemmer = StemmerFactory().create_stemmer()
        except ImportError:
            print("⏳ (Predictor) Sastrawi tidak ditemukan, mencoba menginstal jika diperlukan untuk stemming...")
            # Umumnya untuk prediksi, jika training menggunakan stemming, Sastrawi harus ada.
            # Namun, jika clean_text di bawah tidak memaksa stemming, ini bisa opsional.
            # Untuk konsistensi penuh, kita coba setup.
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "Sastrawi"])
                from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
                self.stemmer = StemmerFactory().create_stemmer()
                print("   Sastrawi berhasil diinstal.")
            except Exception as e:
                print(f"   Gagal menginstal Sastrawi: {e}. Stemming tidak akan berfungsi.")
                self.stemmer = None


    def _setup_nltk(self):
        try:
            import nltk
            from nltk.corpus import stopwords
            # Cek apakah 'stopwords' sudah diunduh
            try:
                self.stop_words = set(stopwords.words('indonesian'))
            except LookupError:
                print("⏳ (Predictor) Korpus NLTK 'stopwords' belum diunduh, mencoba mengunduh...")
                nltk.download('stopwords', quiet=True)
                self.stop_words = set(stopwords.words('indonesian'))
                print("   Korpus 'stopwords' NLTK berhasil diunduh.")
        except ImportError:
            print("⏳ (Predictor) NLTK tidak ditemukan, mencoba menginstal jika diperlukan untuk stopword removal...")
            # Sama seperti Sastrawi, untuk konsistensi.
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "nltk"])
                import nltk
                nltk.download('stopwords', quiet=True) # Download stopwords setelah instalasi
                from nltk.corpus import stopwords
                self.stop_words = set(stopwords.words('indonesian'))
                print("   NLTK dan korpus 'stopwords' berhasil diinstal/dimuat.")
            except Exception as e:
                print(f"   Gagal menginstal NLTK/stopwords: {e}. Stopword removal tidak akan berfungsi optimal.")
                self.stop_words = set()


    def _configure_stopwords(self):
        if self.stop_words: # Hanya konfigurasi jika stop_words berhasil dimuat
            for neg_word in self.negation_words:
                self.stop_words.discard(neg_word)

    def _load_slang_dictionary(self) -> dict:
        return {'gue': 'saya', 'gw': 'saya', 'lu': 'kamu', 'lo': 'kamu', 'yg': 'yang', 'dgn': 'dengan', 'ga': 'tidak', 'gak': 'tidak', 'nggak': 'tidak', 'bgt': 'banget', 'gmn': 'bagaimana'}

    def clean_text(self, text: str) -> str:
        # Fungsi ini HARUS IDENTIK dengan yang digunakan saat training
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
                # Lakukan stemming jika stemmer ada (konsisten dengan training)
                stemmed_word = self.stemmer.stem(original_word) if self.stemmer else original_word
                final_words.append(f"NOT_{stemmed_word}")
            # Lakukan stopword removal jika stop_words ada (konsisten dengan training)
            elif self.stop_words and word not in self.stop_words: 
                stemmed_word = self.stemmer.stem(word) if self.stemmer else word
                final_words.append(stemmed_word)
            elif not self.stop_words: # Jika stop_words tidak di-set (misal, karena NLTK gagal load)
                stemmed_word = self.stemmer.stem(word) if self.stemmer else word
                final_words.append(stemmed_word)

        return ' '.join(final_words).strip()


# ==============================================================================
# KELAS UTAMA UNTUK PREDIKSI
# ==============================================================================
class Predictor:
    def __init__(self, model_dir="models", model_filename="best_sentiment_model.keras", max_len=150):
        print("Inisialisasi Predictor...")
        self.model_dir = model_dir
        self.model_path = os.path.join(self.model_dir, model_filename)
        self.tokenizer_path = os.path.join(self.model_dir, "tokenizer")
        self.encoder_path = os.path.join(self.model_dir, "label_encoder.joblib")
        self.max_len = max_len

        self.model = None
        self.tokenizer = None
        self.label_encoder = None
        self.preprocessor = AdvancedTextPreprocessor() # Preprocessor dipanggil di sini

    def load_artifacts(self) -> bool:
        print("Mencari artefak yang diperlukan...")
        if not os.path.exists(self.model_path): print(f"❌ KESALAHAN: File model tidak ditemukan di '{self.model_path}'"); return False
        if not os.path.isdir(self.tokenizer_path): print(f"❌ KESALAHAN: Folder tokenizer tidak ditemukan di '{self.tokenizer_path}'"); return False
        if not os.path.exists(self.encoder_path): print(f"❌ KESALAHAN: File label encoder tidak ditemukan di '{self.encoder_path}'"); return False
        
        print("✅ Semua artefak ditemukan. Memuat...")
        try:
            self.model = tf.keras.models.load_model(self.model_path)
            self.tokenizer = BertTokenizerFast.from_pretrained(self.tokenizer_path)
            self.label_encoder = joblib.load(self.encoder_path)
            print("✅ Model, Tokenizer, dan Label Encoder berhasil dimuat.")
            
            # Validasi untuk model binary
            if len(self.label_encoder.classes_) != 2:
                print(f"⚠️ PERINGATAN: Label encoder memuat {len(self.label_encoder.classes_)} kelas ({self.label_encoder.classes_}), "
                      f"namun model ini diharapkan binary (positif/negatif).")
                print(f"   Pastikan model di '{self.model_path}' adalah model binary.")
            else:
                print(f"   Label encoder dimuat untuk kelas: {self.label_encoder.classes_} (Binary).")
            return True
        except Exception as e: print(f"❌ Terjadi kesalahan saat memuat artefak: {e}"); return False

    def predict(self, text: str) -> tuple[str, float] | tuple[str, None]:
        if not isinstance(text, str) or not text.strip(): return "Input tidak valid", None
        
        print(f"   Teks asli: '{text}'")
        cleaned_text = self.preprocessor.clean_text(text)
        print(f"   Teks setelah dibersihkan: '{cleaned_text}'")
        if not cleaned_text: return "Teks tidak bermakna setelah dibersihkan", None
        
        encoded_input = self.tokenizer.batch_encode_plus(
            [cleaned_text], add_special_tokens=True, max_length=self.max_len,
            padding='max_length', truncation=True, return_tensors='tf'
        )['input_ids']
        
        prediction_probs = self.model.predict(encoded_input, verbose=0)[0]
        
        predicted_index = np.argmax(prediction_probs)
        confidence = prediction_probs[predicted_index]
        
        # Pastikan label_encoder sudah dimuat dan memiliki atribut classes_
        if self.label_encoder and hasattr(self.label_encoder, 'classes_'):
            predicted_label = self.label_encoder.inverse_transform([predicted_index])[0]
        else:
            return "Error: Label encoder tidak siap", None
            
        return predicted_label, confidence

# ==============================================================================
# BLOK EKSEKUSI UTAMA
# ==============================================================================
if __name__ == "__main__":
    # Menggunakan nama folder dan file sesuai permintaan Anda
    MODEL_DIRECTORY = "models"
    MODEL_FILENAME = "best_sentiment_model.keras"

    print(f"TensorFlow version: {tf.__version__}") # Tambahkan ini untuk info
    
    predictor = Predictor(model_dir=MODEL_DIRECTORY, model_filename=MODEL_FILENAME)
    
    if predictor.load_artifacts():
        print("\n" + "=" * 50)
        print("🚀 MODEL PREDIKSI SENTIMEN (BINARY) SIAP DIGUNAKAN")
        print("   Ketik 'exit' atau 'keluar' untuk berhenti.")
        print("=" * 50)
        while True:
            try:
                user_input = input("\nKetik sebuah kalimat: ")
                if user_input.lower() in ['exit', 'keluar']: print("👋 Terima kasih! Sampai jumpa."); break
                label, confidence = predictor.predict(user_input)
                if confidence is not None: print(f"➡️  Prediksi Sentimen: **{str(label).upper()}** (Kepercayaan: {confidence:.2%})")
                else: print(f"⚠️  {label}")
            except KeyboardInterrupt: print("\n👋 Program dihentikan. Sampai jumpa!"); break
            except Exception as e: print(f"❌ Terjadi error tak terduga: {e}")
    else:
        print("\nProgram berhenti karena gagal memuat semua komponen yang diperlukan.")
        print(f"Pastikan Anda sudah menjalankan skrip training dan folder '{MODEL_DIRECTORY}' berisi:")
        print(f"   - '{MODEL_FILENAME}'")
        print(f"   - folder 'tokenizer/'")
        print(f"   - file 'label_encoder.joblib'")