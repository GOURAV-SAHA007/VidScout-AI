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
        print(f"Reference chunk directory '{INPUT_DIR}' not found.")
        return

    jsons = [f for f in os.listdir(INPUT_DIR) if f.endswith(".json")]
    if not jsons:
        print(f"No processed JSON chunk manifests located in '{INPUT_DIR}'.")
        return
    
    existing_df = None
    processed_files = set()
    start_chunk_id = 0

    if os.path.exists(EMBEDDINGS_PATH):
        try:
            existing_df = joblib.load(EMBEDDINGS_PATH)
            if not existing_df.empty and "source_file" in existing_df.columns:
                processed_files = set(existing_df["source_file"].unique())
                start_chunk_id = int(existing_df["chunk_id"].max())+1
            elif not existing_df.empty:
                start_chunk_id = len(existing_df)
        except Exception as e:
            existing_df = None

    pending_jsons = [f for f in jsons if f not in processed_files]

    if not pending_jsons:
        print("All JSON File are already transcribed")
        return

    my_dicts = []
    chunk_id = start_chunk_id

    for json_file in pending_jsons:
        file_path = os.path.join(INPUT_DIR, json_file)
        with open(file_path, "r", encoding="utf-8") as f:
            content = json.load(f)
            print(f"Generating BGE-M3 Embeddings for {json_file}")
            
            # Isolate text strings for batch vectorization
            chunks = content.get("chunks", [])
            text_batch = [c["text"] for c in content["chunks"]]
            if not text_batch:
                continue
                
            embeddings = create_embedding(text_batch)  

            for i, chunk in enumerate(content["chunks"]):
                chunk_record = chunk.copy()
                chunk_record["chunk_id"] = chunk_id
                chunk_record["source_file"] = json_file  
                chunk_record["embedding"] = embeddings[i]
                chunk_id += 1
                my_dicts.append(chunk_record)

    if my_dicts:
        new_df = pd.DataFrame.from_records(my_dicts)
        
        # Combine existing and new dataframes if applicable
        if existing_df is not None and not existing_df.empty:
            final_df = pd.concat([existing_df, new_df], ignore_index=True)
        else:
            final_df = new_df

        # Ensure the target folder location for the vector database exists
        output_directory = os.path.dirname(EMBEDDINGS_PATH)
        if output_directory:
            os.makedirs(output_directory, exist_ok=True)
            
        joblib.dump(final_df, EMBEDDINGS_PATH)
        print(f"Local vector space updated successfully! Total chunks in DB: {len(final_df)} at '{EMBEDDINGS_PATH}'")
    else:
        print("No new context data was vectorized.")

if __name__ == "__main__":
    main()