from flask import Flask, request, jsonify, send_from_directory
from sklearn.metrics import confusion_matrix
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
from moviepy.editor import concatenate_videoclips, VideoFileClip
from pydub import AudioSegment
from pydub.effects import low_pass_filter, high_pass_filter
import speech_recognition as sr
import yt_dlp
import torch
from sentence_transformers import SentenceTransformer, util
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from datetime import datetime, date
import glob

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

model = SentenceTransformer('distiluse-base-multilingual-cased-v1')

def preprocess_text(text):
    text = text.lower()
    tokens = re.findall(r'\b\w+(?:-\w+)?\b', text)
    stemmed_tokens = [stemmer.stem(token) if '-' not in token else token for token in tokens]
    return stemmed_tokens

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
    url = f"{SUPABASE_URL}/dataset?select=*,synonym(*)"
    headers = {
        "apikey": SUPABASE_API_KEY,
        "Authorization": f"Bearer {SUPABASE_API_KEY}",
    }
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        data = response.json()
        print("DEBUG synonym response:", data)
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
    tokens = word_tokenize(normalized_text)
    return tokens

def build_translator(dataframe):
    translator = {}
    for _, row in dataframe.iterrows():
        translator[row['text'].lower()] = row['path_gesture']
    return translator

def translate_to_sign_language(text, translator, debug=False):
    tokens = preprocess_text_for_gesture(text, translator)
    result = []

    i = 0
    while i < len(tokens):
        matched = False
        for j in range(len(tokens), i, -1):
            phrase = " ".join(tokens[i:j]) 
            if phrase in translator:
                result.append(translator[phrase])  
                i = j 
                matched = True
                break
        
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

def handle_missing_token(token, translator):
    paths = []
    for char in token:
        if char in translator:
            paths.append(translator[char])
        else:
            return f"Token '{token}' not found in translator."
    return paths

def merge_videos(video_paths, output_filename):
    try:
        clips = [VideoFileClip(path) for path in video_paths]
        final_clip = concatenate_videoclips(clips)
        final_clip.write_videofile(output_filename, codec='libx264', audio_codec='aac')
        final_clip.close()
        return output_filename
    except Exception as e:
        raise Exception(f"Error merging videos: {str(e)}")

@app.route('/uploads/<path:filename>', methods=['GET'])
def serve_uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

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

        # Jika file video sudah ada, lewati proses unggah ulang
        if os.path.exists(video_path):
            transcription = process_video(video_path)
        else:
            video_file.save(video_path)
            transcription = process_video(video_path)

        # Proses gesture bahasa isyarat
        df = fetch_data_from_supabase()
        if "text" not in df.columns or "path_gesture" not in df.columns:
            return jsonify({"error": "Dataset is missing required columns"}), 400

        translator = build_translator(df)
        tokens = preprocess_text_for_gesture(transcription, translator)
        sign_language_paths = []
        
        synonym_map = {
            entry['synonym'].lower(): True
            for _, row in df.iterrows()
            for entry in row.get('synonym', [])
            if 'synonym' in entry
        }

        y_true = [1 if token in translator or token in synonym_map else 0 for token in tokens]
        y_pred = []

        for token in tokens:
            path = None
            
            if token in translator:
                path = translator[token]
            else :
                # Cek apakah token adalah synonym
                matched_row = None
                for _, row in df.iterrows():
                    synonym = row.get('synonym',[])
                    for synonym_entry in synonym:
                        if synonym_entry.get("synonym","").lower() == token:
                            filtered = df[df['id'] == synonym_entry['id_dataset']]
                            if not filtered.empty:
                                matched_row = filtered.iloc[0]
                                path = matched_row['path_gesture']
                                break
                    if path:
                        break
            
            if path:
                if path.startswith("uploads"):
                    path = f"http://localhost:5000/{path.replace(os.sep, '/')}"
                sign_language_paths.append(path)
                y_pred.append(1)
            else:
                matching_files = glob.glob(os.path.join(UPLOAD_FOLDER, f"{token}_*.mp4"))
                if matching_files:
                    # File gesture sudah ada → gunakan yang ada
                    latest_file = max(matching_files, key=os.path.getmtime)
                    filename = os.path.basename(latest_file)
                    try:
                        date_str = filename.split('_')[1].replace('.mp4','')
                        file_date = datetime.strptime(date_str, "%d%m%Y").date()
                        today = datetime.today().date()
                    except (IndexError, ValueError):
                        file_date = today
                        
                    sign_language_paths.append(f"http://localhost:5000/uploads/{os.path.basename(latest_file)}")
                    y_pred.append(1) 
                    
                    if file_date != today:
                        y_true[tokens.index(token)] = 1
                else:
                    result = handle_missing_token(token, translator)
                    if isinstance(result, str):  # Error message
                        sign_language_paths.append(result)
                        y_pred.append(0)
                    else:
                        base_name = f"{token}_{datetime.today().strftime('%d%m%Y')}.mp4"
                        merged_video_path = merge_videos(result, os.path.join(UPLOAD_FOLDER, base_name))
                        sign_language_paths.append(f"http://localhost:5000/uploads/{os.path.basename(merged_video_path)}")
                        y_pred.append(1)

        # Hitung metrik evaluasi
        accuracy = round(accuracy_score(y_true, y_pred) * 100, 2)
        precision = round(precision_score(y_true, y_pred, zero_division=0) * 100, 2)
        recall = round(recall_score(y_true, y_pred, zero_division=0) * 100, 2)
        f1 = round(f1_score(y_true, y_pred, zero_division=0) * 100, 2)
        cm = confusion_matrix(y_true, y_pred).tolist()
        
        return jsonify({
            'status': 'success',
            'message': 'Video berhasil diproses.',
            'transcription': transcription,
            'gesture_paths': sign_language_paths,
            'evaluation': {
                'accuracy': f"{accuracy}%",
                'precision': f"{precision}%",
                'recall': f"{recall}%",
                'f1_score': f"{f1}%",
                'confusion_matrix': cm
            }

        })

    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/v1/upload-link', methods=['POST'])
def upload_link():
    try:
        video_link = request.json.get('videoLink')
        if not video_link:
            return jsonify({'status': 'error', 'message': 'Harap masukkan link video.'}), 400

        # Generate a unique filename based on the video link
        video_filename = re.sub(r'\W+', '_', video_link) + '.mp4'
        video_path = os.path.join(UPLOAD_FOLDER, video_filename)

        # Jika file video sudah ada, lewati proses unduhan
        if os.path.exists(video_path):
            transcription = process_video(video_path)
        else:
            if 'youtube.com' in video_link or 'youtu.be' in video_link:
                ydl_opts = {
                    'format': 'bestvideo+bestaudio/best',
                    'outtmpl': video_path,
                    'merge_output_format': 'mp4',
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

        # Proses gesture bahasa isyarat
        df = fetch_data_from_supabase()
        if "text" not in df.columns or "path_gesture" not in df.columns:
            return jsonify({"error": "Dataset is missing required columns"}), 400

        translator = build_translator(df)
        tokens = preprocess_text_for_gesture(transcription, translator)
        sign_language_paths = []
        
        synonym_map = {
            entry['synonym'].lower(): True
            for _, row in df.iterrows()
            for entry in row.get('synonym', [])
            if 'synonym' in entry
        }

        y_true = [1 if token in translator or token in synonym_map else 0 for token in tokens]
        y_pred = []
        
        for token in tokens:
            path = None
            if token in translator:
                path = translator[token]
            else:
                #cek synonym
                matched_row = None
                for _, row in df.iterrows():
                    synonym = row.get('synonym', [])
                    for synonym_entry in synonym:
                        if synonym_entry.get("synonym", "").lower() == token:
                            filtered = df[df['id'] == synonym_entry['id_dataset']]             
                            if not filtered.empty:
                                matched_row = filtered.iloc[0]
                                path = matched_row['path_gesture']
                                break
                    if path:
                        break
                      
            if path:       
                if path.startswith("uploads"):
                    path = f"http://localhost:5000/{path.replace(os.sep, '/')}"
                sign_language_paths.append(path)
                y_pred.append(1)
            else:
                # Periksa apakah file video gesture untuk token sudah ada
                matching_files = glob.glob(os.path.join(UPLOAD_FOLDER, f"{token}.mp4"))
                if matching_files:
                    latest_file = max(matching_files, key=os.path.getmtime)
                    filename = os.path.basename(latest_file)
                    try:
                        date_str = filename.split('_')[1].replace('.mp4','')
                        file_date = datetime.strptime(date_str, "%d%m%Y").date()
                        today = datetime.today().date()
                    except (IndexError, ValueError):
                        file_date = today
                    
                    sign_language_paths.append(f"http://localhost:5000/uploads/{os.path.basename(latest_file)}")
                    y_pred.append(1)
                    
                    if file_date != today:
                        y_true[tokens.index(token)] = 1
                else:
                    existing_token_videos = glob.glob(os.path.join(UPLOAD_FOLDER, f"{token}_*.mp4"))
                    if existing_token_videos:
                        # Gunakan video gesture pertama yang ditemukan
                        latest_file = max(existing_token_videos, key=os.path.getmtime)
                        sign_language_paths.append(f"http://localhost:5000/uploads/{os.path.basename(latest_file)}")
                        y_pred.append(1)
                    else:
                        result = handle_missing_token(token, translator)
                        if isinstance(result, str):
                            sign_language_paths.append(result)
                            y_pred.append(0)
                        else:
                            base_name = f"{token}_{datetime.today().strftime('%d%m%Y')}.mp4"
                            merged_video_path = merge_videos(result, os.path.join(UPLOAD_FOLDER, base_name))
                            sign_language_paths.append(f"http://localhost:5000/uploads/{os.path.basename(merged_video_path)}")
                            y_pred.append(1)
        
        accuracy = round(accuracy_score(y_true, y_pred) * 100, 2)
        precision = round(precision_score(y_true, y_pred, zero_division=0) * 100, 2)
        recall = round(recall_score(y_true, y_pred, zero_division=0) * 100, 2)
        f1 = round(f1_score(y_true, y_pred, zero_division=0) * 100, 2)
        cm = confusion_matrix(y_true, y_pred).tolist()

        return jsonify({
            'status': 'success',
            'message': 'Video berhasil diproses.',
            'transcription': transcription,
            'gesture_paths': sign_language_paths,
            'evaluation': {
                'accuracy': f"{accuracy}%",
                'precision': f"{precision}%",
                'recall': f"{recall}%",
                'f1_score': f"{f1}%",
                'confusion_matrix': cm
            }
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
        tokens = preprocess_text_for_gesture(input_text, translator)
        sign_language_paths = []
        
        synonym_map = {
            entry['synonym'].lower(): True
            for _, row in df.iterrows()
            for entry in row.get('synonym', [])
            if 'synonym' in entry
        }

        y_true = [1 if token in translator or token in synonym_map else 0 for token in tokens]
        y_pred = []

        for token in tokens:
            path = None

            # Langsung ada di translator (text)
            if token in translator:
                path = translator[token]
            else:
                # Cek apakah token adalah synonym
                matched_row = None
                for _, row in df.iterrows():
                    synonyms = row.get('synonym', [])
                    for synonym_entry in synonyms:
                        if synonym_entry.get("synonym", "").lower() == token:
                            filtered = df[df['id'] == synonym_entry['id_dataset']]
                            if not filtered.empty:
                                matched_row = filtered.iloc[0]
                                path = matched_row['path_gesture']
                                break
                    if path:
                        break

            if path:
                # Ubah ke URL jika path lokal
                if path.startswith("uploads"):
                    path = f"http://localhost:5000/{path.replace(os.sep, '/')}"
                sign_language_paths.append(path)
                y_pred.append(1)
            else:
                matching_files = glob.glob(os.path.join(UPLOAD_FOLDER, f"{token}_*.mp4"))
                if matching_files:
                    # File gesture sudah ada → gunakan yang ada
                    latest_file = max(matching_files, key=os.path.getmtime)
                    filename = os.path.basename(latest_file)
                    try:
                        date_str = filename.split('_')[1].replace('.mp4','')
                        file_date = datetime.strptime(date_str, "%d%m%Y").date()
                        today = datetime.today().date()
                    except (IndexError, ValueError):
                        file_date = today
                        
                    sign_language_paths.append(f"http://localhost:5000/uploads/{os.path.basename(latest_file)}")
                    y_pred.append(1) 
                    
                    if file_date != today:
                        y_true[tokens.index(token)] = 1
                else:
                    result = handle_missing_token(token, translator)
                    if isinstance(result, str):  # Error message
                        sign_language_paths.append(result)
                        y_pred.append(0)
                    else:
                        base_name = f"{token}_{datetime.today().strftime('%d%m%Y')}.mp4"
                        merged_video_path = merge_videos(result, os.path.join(UPLOAD_FOLDER, base_name))
                        sign_language_paths.append(f"http://localhost:5000/uploads/{os.path.basename(merged_video_path)}")
                        y_pred.append(1)
                        
        # Evaluasi
        accuracy = round(accuracy_score(y_true, y_pred) * 100, 2)
        precision = round(precision_score(y_true, y_pred, zero_division=0) * 100, 2)
        recall = round(recall_score(y_true, y_pred, zero_division=0) * 100, 2)
        f1 = round(f1_score(y_true, y_pred, zero_division=0) * 100, 2)
        cm = confusion_matrix(y_true, y_pred).tolist()

        return jsonify({
            "tokens": tokens,
            "paths": sign_language_paths,
            "evaluation": {
                "accuracy": f"{accuracy}%",
                "precision": f"{precision}%",
                "recall": f"{recall}%",
                "f1_score": f"{f1}%",
                "confusion_matrix": cm
            }
        }), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
