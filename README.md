
# 🧠 MyAIGist — AI-Powered Summarization & Q&A Assistant

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![OpenAI](https://img.shields.io/badge/OpenAI-GPT--4-success)
![Gradio](https://img.shields.io/badge/Gradio-UI-orange)
![AWS Ready](https://img.shields.io/badge/AWS%20Ready-yes-yellow)

**MyAIGist** is an AI-powered tool that lets users upload text, audio, video, or YouTube links and receive a 1–2 minute digestible summary with optional audio playback. Users can also ask follow-up questions about the content via conversational Q&A (RAG). The app uses OpenAI models and provides a simple Gradio-based UI.

---

## 📸 Screenshot

![MyAIGist Screenshot](https://via.placeholder.com/800x400.png?text=MyAIGist+UI+Screenshot+Placeholder)

---

## 🚀 Features

- 🔹 Upload or link to:
  - PDF, DOCX, TXT files
  - YouTube videos
  - Audio/Video files (.mp3, .mp4)
- 🧠 Receive GPT-based summary (custom prompt-tuned)
- 🔊 Play summaries and Q&A responses via OpenAI TTS
- ❓ Ask follow-up questions in a chat-style format
- 🎛️ Choose model for summarization or Q&A
- 📨 (Coming soon) Send content via email and receive a summary

---

## 🖥️ Local Development

### 1. Clone the Repo

```bash
git clone https://github.com/your-username/ai-summarizer.git
cd ai-summarizer
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Set Your OpenAI API Key

```bash
export OPENAI_API_KEY=your-openai-api-key
```

### 4. Run the App

```bash
python main.py
```

Then visit: [http://localhost:7860](http://localhost:7860)

---

## 🔧 CI/CD (Optional)

Set up GitHub Actions for automatic deployment (e.g., to AWS or container registry):

```yaml
# .github/workflows/deploy.yml
name: Deploy App

on:
  push:
    branches: [main]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.10'
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
      - name: Run tests or deploy
        run: |
          echo "Add your deployment command here"
```

---

## 🛠 Architecture

- **Frontend**: Gradio UI
- **AI Backend**: OpenAI GPT-4 / GPT-3.5-turbo for summaries + Q&A
- **TTS**: OpenAI Text-to-Speech (TTS-1)
- **Multimedia Processing**: Pytube, ffmpeg, moviepy
- **Planned Integration**: AWS SES + S3 + Lambda for email input

---

## 📦 Coming Soon

- ✅ Summarization via email (SES + Lambda)
- 🌍 Deployment via AWS (App Runner or Lambda + API Gateway)
- 🧠 Session memory & chat history
- 🧾 Summary + Q&A export as file or email

---

## 📝 License

MIT License © 2025 [Your Name]

---

## 🙋‍♂️ Contact

Have ideas, feedback, or want to contribute? Open an issue or reach out on GitHub.
