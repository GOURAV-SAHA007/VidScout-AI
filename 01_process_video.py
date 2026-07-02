#1 : Process the Video Files and Convert it into .mp3(audio) format

import os
import subprocess

os.makedirs("audios", exist_ok=True)

files = os.listdir("videos")
file_num = 1

for file in files:
    if file.startswith('.'):
        continue      
    file_name = os.path.splitext(file)[0]
    
    subprocess.run([
        "ffmpeg", 
        "-i", f"videos/{file}", 
        f"audios/{file_num}_{file_name}.mp3"
    ])
    
    file_num = file_num + 1