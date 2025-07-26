# 🧠 MyAIGist — AI-Powered Summarization & Q&A Assistant

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![OpenAI](https://img.shields.io/badge/OpenAI-GPT--4-success)
![Gradio](https://img.shields.io/badge/Gradio-UI-orange)
![Docker](https://img.shields.io/badge/Docker-Containerized-blue)

**MyAIGist** is an AI-powered tool that lets users upload text, audio, video, or YouTube links and receive a 1–2 minute digestible summary with optional audio playback. Users can also ask follow-up questions via RAG-based conversational Q&A. Built with OpenAI, Gradio, yt-dlp, and FAISS.

---

## 🚀 Features

- 📁 Upload:
  - PDFs, DOCX, TXT
  - YouTube video links
  - Audio/Video files (.mp3, .mp4)
  - Raw text via input box
- 🧠 GPT-powered summarization
- 🔊 Text-to-speech audio playback
- ❓ Q&A over uploaded/transcribed content
- 🧠 FAISS-based RAG context retrieval
- 🎛️ Choose between OpenAI models (e.g., GPT-4, GPT-3.5)
- 📨 (Coming soon) Email input support (SES + S3 + Lambda)

---

## 🐳 Local Docker Deployment

### 1. Clone the Repository

```bash
git clone https://github.com/schwim23/ai-summarizer.git
cd ai-summarizer
```

### 2. Create a `.env` File

Inside the root directory:

```bash
touch .env
```

Paste the following into `.env` (replace with your key):

```env
OPENAI_API_KEY=sk-xxx...
```

### 3. Run with Docker Compose

```bash
docker-compose up --build
```

Gradio will launch at: [http://localhost:7860](http://localhost:7860)

### 4. (Optional) Mount Local Drive for Fast Iteration

If you want to mount local changes into the container:

```bash
docker-compose -f docker-compose.dev.yml up --build
```

(Ensure `docker-compose.dev.yml` maps your `./app` folder with a volume.)

---

## 🧪 Development (No Docker)

### 1. Install Requirements

```bash
pip install -r requirements.txt
```

### 2. Set Environment Variable

```bash
export OPENAI_API_KEY=sk-xxx...
```

### 3. Run the App

```bash
python main.py
```

---

## 🛠 Tech Stack

| Layer        | Tool                        |
|--------------|-----------------------------|
| UI           | Gradio                      |
| AI API       | OpenAI (Chat, TTS, Embeds)  |
| Audio/Video  | yt-dlp + ffmpeg             |
| RAG Search   | FAISS + tiktoken            |
| File Parsing | PyPDF2, python-docx         |

---

## 📦 Coming Soon

- ✅ Email-to-summary via AWS SES
- 💬 Session memory and chat history
- ☁️ One-click AWS deployment
- 📤 Export summaries + Q&A
- 🔒 User authentication & dashboard

---

## 📝 License

MIT License © 2025 Michael Schwimmer

---

## 🙋‍♂️ Contact

Have ideas or issues? Open an issue or pull request at [github.com/schwim23/ai-summarizer](https://github.com/schwim23/ai-summarizer)