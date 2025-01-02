from flask import Flask, request, jsonify
import requests
import pandas as pd
import re
import nltk
from nltk.tokenize import word_tokenize
from flask_cors import CORS
from dotenv import load_dotenv
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import os

# Load environment variables
load_dotenv()

app = Flask(__name__)
CORS(app)

# Pastikan data NLTK tersedia
nltk.data.path.append('/root/nltk_data')

# Konfigurasi Supabase
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_API_KEY = os.getenv("SUPABASE_API_KEY")

# Inisialisasi stemmer Sastrawi
factory = StemmerFactory()
stemmer = factory.create_stemmer()

# Fungsi untuk mendapatkan dataset dari Supabase
def fetch_data_from_supabase():
    headers = {
        "apikey": SUPABASE_API_KEY,
        "Authorization": f"Bearer {SUPABASE_API_KEY}",
    }
    response = requests.get(SUPABASE_URL, headers=headers)
    if response.status_code == 200:
        data = response.json()
        return pd.DataFrame(data)
    else:
        raise Exception(f"Error fetching data: {response.status_code}, {response.text}")

# Fungsi normalisasi teks
def normalize_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)  # Hapus karakter non-huruf
    stemmed_text = stemmer.stem(text)  # Stem teks menggunakan Sastrawi
    return stemmed_text

# Fungsi preprocessing teks
def preprocess_text(text):
    normalized_text = normalize_text(text)  # Normalisasi teks
    tokens = word_tokenize(normalized_text)  # Tokenisasi teks
    return tokens

# Translator: Membangun mapping teks ke path gesture
def build_translator(dataframe):
    translator = {}
    for _, row in dataframe.iterrows():
        translator[row['text'].lower()] = row['path_gesture']
    return translator

# Translator: Menerjemahkan teks ke path gesture
def translate_to_sign_language(text, translator, debug=False):
    tokens = preprocess_text(text)  # Preprocessing teks input
    result = []
    for token in tokens:
        if token in translator:
            result.append(translator[token])  # Ambil path gesture
        else:
            result.append(f"Token '{token}' not found in translator.")
    if debug:
        return tokens, result  # Kembalikan tokens dan hasil translasi
    return result

@app.route('/api/v1/text-to-gesture', methods=['POST'])
def text_to_gesture():
    try:
        # Ambil teks dari JSON atau form input
        input_text = request.form.get('text') or request.json.get('text')
        if not input_text:
            return jsonify({"error": "No text provided"}), 400

        # Ambil data dari Supabase
        df = fetch_data_from_supabase()

        # Pastikan dataset memiliki kolom yang diperlukan
        if "text" not in df.columns or "path_gesture" not in df.columns:
            return jsonify({"error": "Dataset is missing required columns"}), 400

        # Bangun translator dan terjemahkan
        translator = build_translator(df)
        tokens, sign_language_paths = translate_to_sign_language(input_text, translator, debug=True)

        # Kembalikan tokens dan hasil translasi
        return jsonify({
            "tokens": tokens,
            "paths": sign_language_paths
        }), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
