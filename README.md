# VidScout AI 🎥🕵️‍♂️

**VidScout AI** is an intelligent, multi-modal Retrieval-Augmented Generation (RAG) pipeline. It allows users to feed a library of video files into the system and ask highly specific natural language questions about the video content. The system extracts, transcribes, embeds, and retrieves the exact context needed to generate precise answers using local LLMs.

## ⚙️ Tech Stack & Architecture

This project is built using a fully local, privacy-first AI stack:
* **Audio Extraction:** FFmpeg
* **Speech-to-Text Transcription:** OpenAI-Whisper
* **Embeddings:** `bge-m3` (via Ollama)
* **LLM Engine:** `deepseek-r1:1.5b` (via Ollama)
* **Data Processing & Vector Storage:** Pandas, NumPy, Scikit-Learn, Joblib
* **API Routing:** Requests

---

## 🚀 How It Works (The Pipeline)

VidScout AI processes visual media into searchable intelligence in five distinct phases:
1. **Extraction:** Strips audio from raw video files.
2. **Transcription:** Converts audio into timestamped, chunked text (JSON).
3. **Vectorization:** Embeds text chunks into high-dimensional vectors.
4. **Storage:** Saves the vector space as a lightweight Joblib pickle dataframe.
5. **Retrieval & Generation:** Matches user queries against the vector space and feeds the context to DeepSeek-R1 to generate the final answer.

---

## 🛠️ Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/yourusername/VidScout-AI.git](https://github.com/yourusername/VidScout-AI.git)
   cd VidScout-AI
## ⚙️ Prerequisites

Before running the project, make sure you have the following installed:

- **FFmpeg** (required for audio extraction)
- **Python 3.10+**
- **Ollama**

---

## 📦 Installation

### 1. Install Python dependencies

```bash
pip install openai-whisper joblib numpy pandas scikit-learn requests
```

### 2. Set up Ollama

Make sure **Ollama** is installed and running, then download the required models:

```bash
ollama pull bge-m3
ollama pull deepseek-r1:1.5b
```

---

# 📖 Usage

Follow these steps to process your own videos and start asking questions.

## Step 1: Add Your Videos

Create a folder named **`videos`** in the project root and place your video files (`.mp4`, `.mkv`, etc.) inside it.

```
project/
│── videos/
│   ├── video1.mp4
│   ├── video2.mkv
│── ...
```

---

## Step 2: Extract Audio

Convert all videos into MP3 files.

```bash
python 01_process_video.py
```

---

## Step 3: Transcribe & Chunk Audio

Generate transcripts using **Whisper** and split them into structured JSON chunks.

```bash
python 02_chunking_audios.py
```

---

## Step 4: Generate Embeddings

Create vector embeddings using **bge-m3** and save them as a Joblib file.

```bash
python 03_vector_chunks_embed.py
```

---

## Step 5: Query Your Videos

Load the vector database and ask questions about your videos using **DeepSeek-R1**.

```bash
python 04_query_process.py
```

---

# 🚀 Workflow

```
Videos
   │
   ▼
Extract Audio
   │
   ▼
Transcribe (Whisper)
   │
   ▼
Chunk into JSON
   │
   ▼
Generate Embeddings (bge-m3)
   │
   ▼
Store Vector Database
   │
   ▼
Query with DeepSeek-R1
```

---

# 🔮 Future Roadmap

- [ ] Integrate advanced cloud models (e.g., GPT-4) for more accurate reasoning over video content.
- [ ] Build an interactive web interface using **Streamlit** or **Gradio** for drag-and-drop uploads.
- [ ] Add direct YouTube URL processing using **yt-dlp**, eliminating the need for manual downloads.
