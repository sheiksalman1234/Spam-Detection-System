import os

# Add FFmpeg to PATH
os.environ["PATH"] = r"C:\ffmpeg\bin;" + os.environ.get("PATH", "")

import whisper

print("Loading Whisper AI...")

model = whisper.load_model("base")

print("Whisper Loaded Successfully!")

result = model.transcribe(r"C:\SpamCallDetection\backend\spam.mp3.mp3")

print(result["text"])