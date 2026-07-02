#testing whisper

import whisper

model = whisper.load_model("medium")
result = model.transcribe(audio="sample_audio.wav", task="translate")
print(result["text"])