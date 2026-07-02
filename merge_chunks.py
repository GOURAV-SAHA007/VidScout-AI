#merging chunks to improve runtime and reduce the number of chunks to be processed 

import os
import json
import math

n = 5

for filename in os.listdir("json_chunks"):
    if filename.endswith(".json"):
        file_path = os.path.join("json_chunks", filename)
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            new_chunks = []
            num_chunks = len(data["chunks"])
            num_new_chunks = math.ceil(num_chunks/n)

            for i in range(num_new_chunks):
                start_index = i*n
                end_index = min((i+1)*n, num_chunks)

                chunk_group = data["chunks"][start_index:end_index]

                new_chunks.append({
                    "title": chunk_group[0]["title"],
                    "number": data["chunks"][0]["number"],
                    "start": chunk_group[0]["start"],
                    "end": chunk_group[-1]["end"],
                    "text": " ".join(c["text"] for c in chunk_group)
                })

            os.makedirs("new_json_chunks", exist_ok=True)
            with open(os.path.join("new_json_chunks", filename), "w", encoding="utf-8") as f:
                json.dump({"chunks": new_chunks,
                           "text": data["text"]}, f, indent=4)
