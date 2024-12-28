from flask import Flask, request, jsonify
import requests
import pandas as pd
import re
import nltk
from nltk.tokenize import word_tokenize
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

nltk.download('punkt')
nltk.download('punkt_tab')


# Lokasi data NLTK
nltk.data.path.append('/root/nltk_data')

# API Key dan URL Supabase
SUPABASE_URL = "https://bbmgbfgcmwippuwnutkk.supabase.co/rest/v1/dataset?select=*"
SUPABASE_API_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImJibWdiZmdjbXdpcHB1d251dGtrIiwicm9sZSI6ImFub24iLCJpYXQiOjE3MzUzMDA3ODYsImV4cCI6MjA1MDg3Njc4Nn0.X0Duw7PxkJyhbyfabvghHHhr4-WaheiRT4dYUE6PKYY"

app = Flask(__name__)

# Fungsi untuk mendapatkan dataset dari Supabase
def fetch_data_from_supabase():
    headers = {
        "apikey": SUPABASE_API_KEY,
        "Authorization": f"Bearer {SUPABASE_API_KEY}",
    }
    response = requests.get(SUPABASE_URL, headers=headers)
    if response.status_code == 200:
        data = response.json()
        if all(key in data[0] for key in ['text', 'path_gesture']):
            return data
        else:
            raise ValueError("Dataset missing 'text' or 'path_gesture' columns")
    else:
        raise Exception(f"Error fetching data: {response.status_code}, {response.text}")

# Fungsi preprocessing teks
def preprocess_text(text):
    text = text.lower()  # Lowercase
    text = re.sub(r"[^a-z\s]", "", text)  # Hapus angka dan simbol
    tokens = word_tokenize(text)  # Tokenisasi
    return tokens

# Translator: Membangun Mapping Huruf ke Path Gesture
def build_translator(dataframe):
    translator = {}
    for _, row in dataframe.iterrows():
        char = row['text'].lower()
        path = row['path_gesture']
        translator[char] = path
    return translator

# Translator: Menerjemahkan Teks ke Path Gesture
def translate_to_sign_language(text, translator):
    tokens = preprocess_text(text)  # Preprocessing teks input
    result = []
    for token in tokens:
        for char in token:  # Per karakter
            if char in translator:
                result.append(translator[char])  # Tambahkan path gesture
            else:
                result.append(f"Character '{char}' not found in translator.")
    return result

@app.route('/api/v1/text-to-gesture', methods=['POST'])
def text_to_gesture():
    try:
        # Ambil teks dari JSON atau form input
        input_text = request.form.get('text') or request.json.get('text')
        if not input_text:
            return jsonify({"error": "No text provided"}), 400

        # Ambil data dari Supabase
        data = fetch_data_from_supabase()
        df = pd.DataFrame(data)

        # Pastikan ada kolom teks untuk mapping
        if "text" in df.columns and "path_gesture" in df.columns:
            # Membangun translator
            translator = build_translator(df)

            # Terjemahkan ke bahasa isyarat
            sign_language_paths = translate_to_sign_language(input_text, translator)
            return jsonify({"paths": sign_language_paths}), 200
        else:
            return jsonify({"error": "Dataset is missing required columns"}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
