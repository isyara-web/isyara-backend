from flask import Flask, request, jsonify
import os
from flask_cors import CORS
from moviepy.editor import VideoFileClip
from pydub import AudioSegment
from pydub.effects import low_pass_filter, high_pass_filter
import speech_recognition as sr

app = Flask(__name__)
CORS(app)

# Folder untuk menyimpan file sementara
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

SUPPORTED_FORMATS = ['mp4', 'avi', 'mkv', 'mov']  # Format video yang didukung

@app.route('/upload-and-transcribe', methods=['POST'])
def upload_and_transcribe():
    if 'video' not in request.files:
        return jsonify({'error': 'Tidak ada file video yang diunggah.'}), 400

    video_file = request.files['video']

    if video_file.filename == '':
        return jsonify({'error': 'Nama file tidak valid.'}), 400

    # Periksa format file
    file_extension = video_file.filename.split('.')[-1].lower()
    if file_extension not in SUPPORTED_FORMATS:
        return jsonify({'error': f'Format file {file_extension} tidak didukung. Hanya mendukung format: {", ".join(SUPPORTED_FORMATS)}'}), 400

    try:
        # Simpan file video ke folder sementara
        video_path = os.path.join(UPLOAD_FOLDER, video_file.filename)
        video_file.save(video_path)

        # Ekstrak audio dari video
        audio_path = os.path.splitext(video_path)[0] + '.wav'
        filtered_audio_path = os.path.splitext(video_path)[0] + '_filtered.wav'

        try:
            video_clip = VideoFileClip(video_path)
            video_clip.audio.write_audiofile(audio_path, logger=None)
            video_clip.close()  # Pastikan file dilepaskan
        except Exception as e:
            return jsonify({'error': f'Gagal mengekstrak audio: {str(e)}'}), 500

        # Filter frekuensi suara manusia (80 Hz - 300 Hz)
        try:
            audio = AudioSegment.from_file(audio_path)
            audio = high_pass_filter(audio, cutoff=80)  # Hilangkan frekuensi rendah (noise)
            audio = low_pass_filter(audio, cutoff=300)  # Hilangkan frekuensi tinggi (musik, efek suara)
            audio.export(filtered_audio_path, format="wav")
        except Exception as e:
            return jsonify({'error': f'Gagal memproses audio: {str(e)}'}), 500

        # Gunakan SpeechRecognition untuk transkripsi
        try:
            recognizer = sr.Recognizer()
            with sr.AudioFile(filtered_audio_path) as source:
                audio_data = recognizer.record(source)
                transcription = recognizer.recognize_google(audio_data, language='id-ID')
        except Exception as e:
            return jsonify({'error': f'Gagal mentranskripsi audio: {str(e)}'}), 500

        # Hapus file sementara
        os.remove(video_path)
        os.remove(audio_path)
        os.remove(filtered_audio_path)

        # Berikan respons dengan hasil transkripsi
        return jsonify({
            'status': 'success',
            'message': 'Transkripsi berhasil dilakukan.',
            'transcription': transcription
        })

    except Exception as e:
        # Log error untuk debugging
        print(f'Error: {e}')
        return jsonify({'error': f'Terjadi kesalahan saat memproses video: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
