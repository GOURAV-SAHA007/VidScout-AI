import os
import json
import requests
import pandas as pd
import joblib

INPUT_DIR = os.environ.get("NEW_JSON_CHUNKS_DIR", "new_json_chunks")
EMBEDDINGS_PATH = os.environ.get("EMBEDDINGS_PATH", "embeddings.joblib")
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")

def create_embedding(text_list):
    """Generates text embeddings using the local Ollama API layer."""
    endpoint = f"{OLLAMA_HOST.rstrip('/')}/api/embed"
    response = requests.post(endpoint, json={
        "model": "bge-m3",
        "input": text_list
    })
    response.raise_for_status()
    return response.json()['embeddings']

def main():
    if not os.path.exists(INPUT_DIR):
        print(f"⚠️ Reference chunk directory '{INPUT_DIR}' not found.")
        return

    jsons = [f for f in os.listdir(INPUT_DIR) if f.endswith(".json")]
    if not jsons:
        print(f"⚠️ No processed JSON chunk manifests located in '{INPUT_DIR}'.")
        return

    my_dicts = []
    chunk_id = 0

    for json_file in jsons:
        file_path = os.path.join(INPUT_DIR, json_file)
        with open(file_path, "r", encoding="utf-8") as f:
            content = json.load(f)
            print(f"Generating BGE-M3 Embeddings for {json_file}")
            
            # Isolate text strings for batch vectorization
            text_batch = [c["text"] for c in content["chunks"]]
            if not text_batch:
                continue
                
            embeddings = create_embedding(text_batch)  

            for i, chunk in enumerate(content["chunks"]):
                chunk["chunk_id"] = chunk_id
                chunk["embedding"] = embeddings[i]
                chunk_id += 1
                my_dicts.append(chunk)

    if my_dicts:
        df = pd.DataFrame.from_records(my_dicts)
        
        # Ensure the target folder location for the vector database exist
        output_directory = os.path.dirname(EMBEDDINGS_PATH)
        if output_directory:
            os.makedirs(output_directory, exist_ok=True)
            
        joblib.dump(df, EMBEDDINGS_PATH)
        print(f"Local vector space successfully compiled at: {EMBEDDINGS_PATH}")
    else:
        print("Core operational failure: No context data vectorized.")

if __name__ == "__main__":
    main()