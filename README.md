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