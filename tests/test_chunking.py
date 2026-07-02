#testing speech-to-text and saving it in a .json file

import whisper
import json

model = whisper.load_model("medium")
result = model.transcribe(audio="test_chuncking.mp3", task="translate", word_timestamps=False)

chunks = []
for segment in result["segments"]:
    chunks.append({"start":segment["start"],
                    "end":segment["end"],
                    "text":segment["text"]})
    
with open("test_output.json", "w") as f:
    json.dump(chunks, f)