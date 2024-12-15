from flask import Flask, request, jsonify
from moviepy.editor import VideoFileClip
import whisper
import os

app = Flask(__name__)
model = whisper.load_model("base")

@app.route('/upload-and-transcribe', methods=['POST'])
def upload_and_transcribe():
    if 'video' not in request.files:
        return jsonify({'error': 'No video file provided'}), 400

    # Simpan file video yang diunggah
    video_file = request.files['video']
    video_path = 'uploaded_video.mp4'
    audio_path = 'audio.wav'

    video_file.save(video_path)

    # Ekstrak audio dari video menggunakan MoviePy
    try:
        video = VideoFileClip(video_path)
        video.audio.write_audiofile(audio_path)
    except Exception as e:
        return jsonify({'error': f'Failed to extract audio: {str(e)}'}), 500

    # Gunakan Whisper untuk transkripsi audio
    try:
        result = model.transcribe(audio_path)
        transcription = result['text']
    except Exception as e:
        return jsonify({'error': f'Failed to transcribe audio: {str(e)}'}), 500
    finally:
        # Hapus file sementara
        if os.path.exists(video_path):
            os.remove(video_path)
        if os.path.exists(audio_path):
            os.remove(audio_path)

    return jsonify({'transcription': transcription})

if __name__ == '__main__':
    app.run(debug=True)
