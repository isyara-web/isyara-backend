from flask import Flask, request, jsonify
import os
from flask_cors import CORS
import requests
import re
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from collections import Counter
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from dotenv import load_dotenv
from moviepy.editor import VideoFileClip
from pydub import AudioSegment
from pydub.effects import low_pass_filter, high_pass_filter
import speech_recognition as sr
import yt_dlp

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Constants
UPLOAD_FOLDER = 'uploads'
SUPPORTED_FORMATS = ['mp4', 'avi', 'mkv', 'mov']

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# NLTK and Sastrawi setup
nltk.data.path.append('/root/nltk_data')
factory = StemmerFactory()
stemmer = factory.create_stemmer()

# Supabase Configuration
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_API_KEY = os.getenv("SUPABASE_API_KEY")

def preprocess_text(text):
    text = text.lower()
    tokens = re.findall(r'\b\w+(?:-\w+)?\b', text)
    stemmed_tokens = [stemmer.stem(token) if '-' not in token else token for token in tokens]
    return stemmed_tokens

def count_word_frequency(tokens):
    return Counter(tokens)

def process_video(video_path):
    audio_path = os.path.splitext(video_path)[0] + '.wav'
    filtered_audio_path = os.path.splitext(video_path)[0] + '_filtered.wav'
    try:
        video_clip = VideoFileClip(video_path)
        video_clip.audio.write_audiofile(audio_path, logger=None)
        video_clip.close()

        audio = AudioSegment.from_file(audio_path)
        audio = high_pass_filter(audio, cutoff=50)
        audio = low_pass_filter(audio, cutoff=300)
        audio.export(filtered_audio_path, format="wav")

        recognizer = sr.Recognizer()
        with sr.AudioFile(filtered_audio_path) as source:
            audio_data = recognizer.record(source)
            transcription = recognizer.recognize_google(audio_data, language='id-ID')

        os.remove(video_path)
        os.remove(audio_path)
        os.remove(filtered_audio_path)

        return transcription
    except Exception as e:
        raise Exception(f"Proses video gagal: {str(e)}")

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

def normalize_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)
    stemmed_text = stemmer.stem(text)
    return stemmed_text

def preprocess_text_for_gesture(text, translator):
    normalized_text = normalize_text(text)
    tokens = word_tokenize(normalized_text)  # Tokenisasi awal
    return tokens  # Kembalikan token


def build_translator(dataframe):
    translator = {}
    for _, row in dataframe.iterrows():
        translator[row['text'].lower()] = row['path_gesture']
    return translator

def translate_to_sign_language(text, translator, debug=False):
    tokens = preprocess_text_for_gesture(text, translator)
    result = []

    # Sliding window untuk memeriksa frasa
    i = 0
    while i < len(tokens):
        matched = False
        # Cek dari frasa terpanjang (gabungan token)
        for j in range(len(tokens), i, -1):
            phrase = " ".join(tokens[i:j])  # Gabungkan token menjadi frasa
            if phrase in translator:
                result.append(translator[phrase])  # Tambahkan path jika cocok
                i = j  # Lanjutkan setelah frasa yang cocok
                matched = True
                break
        
        # Jika tidak ada frasa panjang yang cocok, lanjutkan dengan token individu
        if not matched:
            token = tokens[i]
            if token in translator:
                result.append(translator[token])
            else:
                result.append(f"Token '{token}' not found in translator.")
            i += 1

    if debug:
        return tokens, result
    return result

@app.route('/api/v1/upload-file', methods=['POST'])
def upload_file():
    try:
        video_file = request.files.get('video')
        if not video_file:
            return jsonify({'status': 'error', 'message': 'Harap unggah file video.'}), 400

        file_extension = video_file.filename.split('.')[-1].lower()
        if file_extension not in SUPPORTED_FORMATS:
            return jsonify({'status': 'error', 'message': 'Format file tidak didukung.'}), 400

        video_path = os.path.join(UPLOAD_FOLDER, video_file.filename)
        video_file.save(video_path)

        transcription = process_video(video_path)
        preprocessed_text = preprocess_text(transcription)
        word_frequencies = count_word_frequency(preprocessed_text)

        return jsonify({
            'status': 'success',
            'message': 'Video berhasil diproses.',
            'transcription': transcription,
            'preprocessed_text': preprocessed_text,
            'word_frequencies': word_frequencies
        })

    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/v1/upload-link', methods=['POST'])
def upload_link():
    try:
        video_link = request.json.get('videoLink')
        if not video_link:
            return jsonify({'status': 'error', 'message': 'Harap masukkan link video.'}), 400

        video_path = os.path.join(UPLOAD_FOLDER, 'downloaded_video.mp4')
        if 'youtube.com' in video_link or 'youtu.be' in video_link:
            ydl_opts = {
                'format': 'best',
                'outtmpl': video_path,
                'quiet': True,
            }
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([video_link])
        else:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36'
            }
            response = requests.get(video_link, headers=headers, stream=True)
            if response.status_code == 200:
                with open(video_path, 'wb') as video_file:
                    for chunk in response.iter_content(chunk_size=1024):
                        video_file.write(chunk)
            else:
                return jsonify({'status': 'error', 'message': 'Gagal mengunduh video. Periksa link Anda.'}), 400

        transcription = process_video(video_path)
        preprocessed_text = preprocess_text(transcription)
        word_frequencies = count_word_frequency(preprocessed_text)

        return jsonify({
            'status': 'success',
            'message': 'Video berhasil diproses.',
            'transcription': transcription,
            'preprocessed_text': preprocessed_text,
            'word_frequencies': word_frequencies
        })

    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/v1/text-to-gesture', methods=['POST'])
def text_to_gesture():
    try:
        input_text = request.form.get('text') or request.json.get('text')
        if not input_text:
            return jsonify({"error": "No text provided"}), 400

        df = fetch_data_from_supabase()

        if "text" not in df.columns or "path_gesture" not in df.columns:
            return jsonify({"error": "Dataset is missing required columns"}), 400

        translator = build_translator(df)
        tokens, sign_language_paths = translate_to_sign_language(input_text, translator, debug=True)

        return jsonify({
            "tokens": tokens,
            "paths": sign_language_paths
        }), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
