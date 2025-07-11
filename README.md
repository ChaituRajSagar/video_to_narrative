✅ README.md
markdown
Copy
Edit
# 🎥 Video to Narrative (Crime Scene Caption + Summary AI)

This is a Flask-based AI application that:
- Extracts frames from uploaded videos
- Captions frames using ViT-GPT2 (transformers)
- Transcribes audio using Whisper
- Summarizes visuals + audio using **Groq LLM (llama3-8b-8192)**

---

## 📁 Project Structure

video_to_narrative/
├── app.py # Main Flask backend
├── logs/
│ └── app.log # Runtime logs
├── uploads/ # Temporary video + frames (ignored in Git)
├── audio/ # Temporary audio files (ignored in Git)
├── templates/
│ └── index.html # Frontend upload page
├── v2t/ # (Your virtual environment — ignored in Git)
├── .gitignore
├── requirements.txt
├── README.md

yaml
Copy
Edit

---

## 🧪 Features

- ✅ Upload crime/dashcam/bodycam video
- ✅ Frame-by-frame captioning (ViT-GPT2)
- ✅ Audio transcription (Whisper)
- ✅ Groq-powered dual summary:
  - General Summary
  - Law Enforcement Report

---

## 🚀 Setup Instructions

### 1. Clone the repo

```bash
git clone https://github.com/yourusername/video_to_narrative.git
cd video_to_narrative
2. Create Python environment
bash
Copy
Edit
python -m venv v2t
v2t\Scripts\activate  # On Windows
# Or:
# source v2t/bin/activate  # On macOS/Linux
3. Install dependencies
bash
Copy
Edit
pip install -r requirements.txt
4. Set your Groq API Key
Create a .env file:

env
Copy
Edit
GROQ_API_KEY=your_groq_api_key_here
Or paste it directly in app.py (not recommended for production).

▶️ Run the App
bash
Copy
Edit
python app.py
Visit: http://127.0.0.1:8009

🛑 Notes
Uses CPU by default (Torch auto-detects)

All uploaded files and frames are auto-cleaned after processing

Logs stored in logs/app.log

🧼 Cleanup
App automatically removes:

Extracted frames

Audio files

Uploaded videos