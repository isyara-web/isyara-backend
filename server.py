from flask import Flask, request, jsonify
import os
from flask_cors import CORS
from moviepy.editor import VideoFileClip
from pydub import AudioSegment
from pydub.effects import low_pass_filter, high_pass_filter
import speech_recognition as sr
import yt_dlp  # Gantikan pytube dengan yt-dlp
import requests  # Untuk link langsung

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

SUPPORTED_FORMATS = ['mp4', 'avi', 'mkv', 'mov']


def process_video(video_path):
    """ Proses video: ekstrak audio, filter, dan transkripsi. """
    audio_path = os.path.splitext(video_path)[0] + '.wav'
    filtered_audio_path = os.path.splitext(video_path)[0] + '_filtered.wav'
    try:
        # Ekstrak audio dari video
        video_clip = VideoFileClip(video_path)
        video_clip.audio.write_audiofile(audio_path, logger=None)
        video_clip.close()

        # Filter audio frekuensi suara manusia
        audio = AudioSegment.from_file(audio_path)
        audio = high_pass_filter(audio, cutoff=50)
        audio = low_pass_filter(audio, cutoff=300)
        audio.export(filtered_audio_path, format="wav")

        # Transkripsi audio ke teks
        recognizer = sr.Recognizer()
        with sr.AudioFile(filtered_audio_path) as source:
            audio_data = recognizer.record(source)
            transcription = recognizer.recognize_google(audio_data, language='id-ID')

        # Hapus file sementara
        os.remove(video_path)
        os.remove(audio_path)
        os.remove(filtered_audio_path)

        return transcription
    except Exception as e:
        raise Exception(f"Proses video gagal: {str(e)}")


@app.route('/upload-file', methods=['POST'])
def upload_file():
    try:
        video_file = request.files.get('video')
        if not video_file:
            return jsonify({'status': 'error', 'message': 'Harap unggah file video.'}), 400

        # Periksa format file
        file_extension = video_file.filename.split('.')[-1].lower()
        if file_extension not in SUPPORTED_FORMATS:
            return jsonify({'status': 'error', 'message': 'Format file tidak didukung.'}), 400

        # Simpan file sementara
        video_path = os.path.join(UPLOAD_FOLDER, video_file.filename)
        video_file.save(video_path)

        transcription = process_video(video_path)
        return jsonify({'status': 'success', 'message': 'Video berhasil diproses.', 'transcription': transcription})

    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/upload-link', methods=['POST'])
def upload_link():
    try:
        video_link = request.json.get('videoLink')
        if not video_link:
            return jsonify({'status': 'error', 'message': 'Harap masukkan link video.'}), 400

        # Unduh video dari link YouTube atau URL lainnya
        video_path = os.path.join(UPLOAD_FOLDER, 'downloaded_video.mp4')
        if 'youtube.com' in video_link or 'youtu.be' in video_link:
            # Menggunakan yt-dlp untuk download video YouTube
            ydl_opts = {
                'format': 'best',
                'outtmpl': video_path,
                'quiet': True,
            }
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([video_link])
        else:
            # Unduh video dari URL langsung
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
        return jsonify({'status': 'success', 'message': 'Video berhasil diproses.', 'transcription': transcription})

    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
