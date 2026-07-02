#2 : Convert the .mp3 files into text that is in .json files

import whisper
import os
import json

audios = os.listdir("audios")
model = whisper.load_model("medium")

for audio in audios:
    num = audio.split("_")[0]
    title = os.path.splitext(audio)[0][2:]

    audio_path = os.path.join("audios", audio)

    result = model.transcribe(
        audio=audio_path,
        task="translate",
        word_timestamps=False
    )

    chunks = []

    for segment in result["segments"]:
        chunks.append({
            "number": num,
            "title": title,
            "start": segment["start"],
            "end": segment["end"],
            "text": segment["text"]
        })

    chunks_with_metadata = {
        "chunks": chunks,
        "text": result["text"]
    }

    json_name = os.path.splitext(audio)[0] + ".json"

    with open(os.path.join("json_chunks", json_name), "w") as f:
        json.dump(chunks_with_metadata, f, indent=4)