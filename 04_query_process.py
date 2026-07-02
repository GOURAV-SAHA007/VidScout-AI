import requests
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import joblib

df = joblib.load("embeddings.joblib")

def create_embedding(text_list):
    r = requests.post("http://localhost:11434/api/embed", json={
        "model": "bge-m3",
        "input": text_list
    })
    embeddings = r.json()['embeddings']
    return embeddings

def inference(prompt):
    r = requests.post("http://localhost:11434/api/generate", json={
        "model": "deepseek-r1:1.5b",
        "prompt": prompt,
        "stream": False
    })

    response = r.json()
    return response

query = input("Ask a Question:")
query_embed = create_embedding([query])[0]

similarity = cosine_similarity(np.vstack(df["embedding"]), [query_embed]).flatten()

max_indx = similarity.argsort()[::-1][0:30]
new_df = df.loc[max_indx]

prompt = f'''I am giving you some TEDx videos as context, here in video subtitle 
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

with open("prompt.txt", "w") as f:
    f.write(prompt)

response = inference(prompt)["response"]
print(response)

with open("response.txt", "w") as f:
    f.write(response)