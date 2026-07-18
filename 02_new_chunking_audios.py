#Import Libraries
import os
import json
import whisper
import math

#Make Variable with directories
AUDIO_DIR = os.environ.get("AUDIO_DIR", "audios")
OUTPUT_DIR = os.environ.get("NEW_JSON_DIR", "new_json_chunks")

#Acces the directort
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if not os.path.exists(AUDIO_DIR):
        print(f"Audio directory '{AUDIO_DIR}' not found.")
        return
    
    audios = [f for f in os.listdir(AUDIO_DIR) if not f.startswith('.')]
    if not audios:
        print(f"No audios found in '{AUDIO_DIR}' directory.")
        return
    
    #Load Model
    model = whisper.load_model("medium")

    #Number of Chunks
    n = 5

    for audio in audios:
        num = audio.split("_")[0]
        title = os.path.splitext(audio)[0][2:]
        audio_path = os.path.join(AUDIO_DIR, audio)

        result = model.transcribe(
            audio = audio_path,
            task = "translate",
            word_timestamps = False
        )

        raw_segments = result["segments"]
        num_chunks = len(raw_segments)
        num_new_chunks = math.ceil(num_chunks/n)
        new_chunks = []

        for i in range(num_new_chunks):
            start_index = i*n
            end_index = min((i+1)*n, num_chunks)
            chunk_group = raw_segments[start_index:end_index]

            if not chunk_group:
                continue

            new_chunks.append({
                "title" : title,
                "number": num,
                "start": chunk_group[0]["start"],
                "end": chunk_group[-1]["end"],
                "text": " ".join(c["text"] for c in chunk_group)
            })
    
        chunks_w_metadata = {
            "chunks" : new_chunks,
            "text" : result["text"]
        }

        json_name = os.path.splitext(audio)[0] + ".json"
        output_path = os.path.join(OUTPUT_DIR, json_name)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(chunks_w_metadata, f, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    main()