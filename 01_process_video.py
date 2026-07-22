#1 : Process the Video Files and Convert it into .mp3(audio) format

import os
import subprocess

VIDEOS_DIR = os.environ.get("VIDEOS_DIR", "videos")
AUDIOS_DIR = os.environ.get("AUDIOS_DIR", "audios")

def extract_audio():
    os.makedirs(AUDIOS_DIR, exist_ok=True)

    if not os.path.exists(VIDEOS_DIR):
        print("Video Directory 404.")
        return
    
    video_files = [f for f in os.listdir(VIDEOS_DIR) if not f.startswith(".")]

    for video in video_files:
        base_name = os.path.splitext(video)[0]
        video_input_path = os.path.join(VIDEOS_DIR, video)
        audio_output_name = os.path.join(AUDIOS_DIR, f"{base_name}.mp3")

        if os.path.exists(audio_output_name):
            print(f"Audio {audio_output_name} already exist.")
            continue

        cmd = [
            "ffmpeg", "-y", "-i", video_input_path,
            "-vn", "-acodec", "libmp3lame", "-q:a", "2",
            audio_output_name
        ]
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

if __name__ == "__main__":
    extract_audio()