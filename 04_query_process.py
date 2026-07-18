import os
import requests
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import joblib

EMBEDDINGS_PATH = os.environ.get("EMBEDDINGS_PATH", "embeddings.joblib")
PROMPT_PATH = os.environ.get("PROMPT_PATH", "prompt.txt")
RESPONSE_PATH = os.environ.get("RESPONSE_PATH", "response.txt")
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
LLM_MODEL = os.environ.get("LLM_MODEL", "deepseek-r1:1.5b")

def create_embedding(text_list):
    """Generates query vector embedding using the local Ollama backend."""
    endpoint = f"{OLLAMA_HOST.rstrip('/')}/api/embed"
    r = requests.post(endpoint, json={
        "model": "bge-m3",
        "input": text_list
    })
    r.raise_for_status()
    return r.json()['embeddings']

def inference(prompt_text):
    """Routes the final context-stuffed prompt payload to the chosen local LLM."""
    endpoint = f"{OLLAMA_HOST.rstrip('/')}/api/generate"
    r = requests.post(endpoint, json={
        "model": LLM_MODEL,
        "prompt": prompt_text,
        "stream": False
    })
    r.raise_for_status()
    return r.json()

def main():
    if not os.path.exists(EMBEDDINGS_PATH):
        print(f"Vector database index missing at '{EMBEDDINGS_PATH}'. Index videos first.")
        return

    if not os.path.exists(PROMPT_PATH):
        print(f"Target prompt file missing at '{PROMPT_PATH}'.")
        return

    # Read the raw string query dropped into the project workspace by the UI
    with open(PROMPT_PATH, "r", encoding="utf-8") as f:
        query = f.read().strip()

    if not query:
        print("Received empty prompt payload.")
        return

    print(f"Accessing vector space: {EMBEDDINGS_PATH}")
    df = joblib.load(EMBEDDINGS_PATH)

    print(f"Vectorizing user query...")
    query_embed = create_embedding([query])[0]

    print("Calculating semantic similarity rankings...")
    similarity = cosine_similarity(np.vstack(df["embedding"]), [query_embed]).flatten()

    # Slice the top 30 most contextually matching chunks
    max_indx = similarity.argsort()[::-1][0:30]
    new_df = df.loc[max_indx]

    # Reconstruct the system prompt using the project-specific matching chunks
    full_prompt = f'''I am giving you some TEDx videos as context, here in video subtitle 
containing video title, video number, start time, end time and text.

{new_df[["title", "number", "start", "end", "text"]].to_json(orient="records")}
---------------------------------
{query}
User asked this question related to the video chunks, you have to answer in a 
human way (dont mention the above format, its just for you) where and how much content 
is taught in which video (in which video and at what timestamp) and guide the user to go to that 
particular video. If user asks unrelated question, tell him that you can only answer questions 
related to the course
'''

    # Overwrite the prompt file with the system-wrapped query payload
    with open(PROMPT_PATH, "w", encoding="utf-8") as f:
        f.write(full_prompt)

    print(f"🤖 Deploying context payload to model: '{LLM_MODEL}'")
    response_json = inference(full_prompt)
    response_text = response_json["response"]

    # Ensure output workspace folder structure exists before saving
    output_dir = os.path.dirname(RESPONSE_PATH)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(RESPONSE_PATH, "w", encoding="utf-8") as f:
        f.write(response_text)

    print(f"Response recorded successfully inside workspace path: {RESPONSE_PATH}")

if __name__ == "__main__":
    main()