# 🧠 MyAIGist — AI-Powered Q&A Assistant with Smart Agent

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![OpenAI](https://img.shields.io/badge/OpenAI-GPT--4-success)
![Gradio](https://img.shields.io/badge/Gradio-UI-orange)
![Docker](https://img.shields.io/badge/Docker-Containerized-blue)

**MyAIGist** is an intelligent AI-powered assistant that transforms any content into an interactive Q&A experience. Upload documents, paste text, or share YouTube links to get instant summaries and ask detailed questions with intelligent follow-up suggestions powered by a Smart Question Agent and advanced RAG technology.

---

## ✨ New in Version 2.0

### 🤖 Smart Question Agent
- **Intelligent Question Analysis** - Automatically detects ambiguous or complex questions
- **Conversation Memory** - Remembers context across your Q&A session  
- **Smart Clarification** - Asks for clarification when questions are unclear
- **Multi-step Reasoning** - Breaks down complex questions into manageable parts
- **Follow-up Suggestions** - AI-generated relevant follow-up questions

### 🔍 Enhanced RAG (Retrieval-Augmented Generation)
- **Advanced Vector Search** - Semantic similarity search with FAISS
- **Smart Chunking** - Sentence-aware chunking with configurable overlap
- **Source Attribution** - Shows which document sections were used for answers
- **Quality Filtering** - Similarity thresholds ensure relevant context
- **Multi-source Synthesis** - Combines information from multiple document sections

### 🌊 Streaming & Real-time Features
- **Streaming Responses** - See answers generated in real-time
- **Live Processing Status** - Visual indicators during content processing
- **Configurable Summary Depth** - Quick Overview, Balanced Summary, or Deep Analysis

---

## 🚀 Core Features

### 📁 Multi-Modal Content Support
- **Documents**: PDFs, DOCX, TXT files with intelligent parsing
- **YouTube Videos**: Automatic transcription and analysis
- **Audio/Video Files**: .mp3, .mp4, .wav, .m4a support
- **Raw Text**: Direct text input and analysis
- **Voice Questions**: Ask questions via audio recording

### 🧠 AI-Powered Analysis
- **GPT-4/3.5 Integration**: Choose your preferred OpenAI model
- **Text-to-Speech**: Audio playback for summaries and answers
- **Smart Summarization**: Three depth levels (Quick/Balanced/Deep)
- **Context-Aware Responses**: Maintains conversation flow and memory

### 🎯 Interactive Experience  
- **Clean Modern UI**: Professional interface with Q&A emphasis
- **Example Questions**: Quick-start suggestions for each document
- **Conversation Threading**: Build on previous questions naturally
- **Mobile Responsive**: Works seamlessly on desktop and mobile

---

## 🐳 Quick Start with Docker

### 1. Clone the Repository

```bash
git clone https://github.com/schwim23/ai-summarizer.git
cd ai-summarizer
```

### 2. Set Up Environment

```bash
# Create .env file
touch .env

# Add your OpenAI API key
echo "OPENAI_API_KEY=sk-your-key-here" > .env
```

### 3. Launch Application

```bash
docker-compose up --build
```

🎉 **Access MyAIGist at:** [http://localhost:7860](http://localhost:7860)

### 4. Development Mode (Optional)

For live code changes during development:

```bash
docker-compose -f docker-compose.dev.yml up --build
```

---

## 🧪 Local Development Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Set Environment Variable

```bash
export OPENAI_API_KEY=sk-your-key-here
```

### 3. Run the Application

```bash
python main.py
```

---

## 🛠 Advanced Tech Stack

| Component | Technology | Purpose |
|-----------|------------|---------|
| **UI Framework** | Gradio | Interactive web interface |
| **AI/LLM** | OpenAI (GPT-4, GPT-3.5, Whisper, TTS) | Core AI processing |
| **Smart Agent** | Custom Question Agent | Intelligent question processing |
| **Vector Search** | FAISS | Semantic document retrieval |
| **Audio/Video** | yt-dlp + ffmpeg | Content transcription |
| **Document Processing** | PyPDF2, python-docx | File parsing |
| **Tokenization** | tiktoken | Text chunking optimization |
| **Containerization** | Docker | Easy deployment |

---

## 🎯 How MyAIGist Works

### Step 1: Upload & Process 📁
- Upload any document, paste text, or share a YouTube link
- AI analyzes content structure and creates optimized chunks
- Builds semantic vector index for intelligent retrieval

### Step 2: Smart Summarization 📋
- Choose summary depth: Quick Overview, Balanced, or Deep Analysis  
- AI generates structured summaries with audio playback
- Real-time streaming shows content as it's generated

### Step 3: Intelligent Q&A 💬
- Ask questions in natural language or via voice
- Smart Question Agent analyzes and improves your questions
- RAG system retrieves relevant context and generates answers
- Get follow-up suggestions and conversation continuity

---

## 💡 Use Cases

- **📚 Research Analysis**: Upload academic papers and get instant insights
- **📄 Document Q&A**: Ask specific questions about contracts, reports, manuals  
- **🎥 Video Learning**: Transform YouTube videos into searchable knowledge
- **📊 Meeting Analysis**: Upload transcripts and extract key decisions
- **🎓 Study Assistant**: Interactive learning with any educational content
- **💼 Business Intelligence**: Analyze reports and extract actionable insights

---

## ⚙️ Configuration Options

### Summary Depth Settings
- **Quick Overview**: 2-3 key bullet points for busy users
- **Balanced Summary**: Comprehensive structured analysis (default)
- **Deep Analysis**: Detailed insights with implications and context

### RAG Parameters (customizable in utils.py)
```python
CHUNK_SIZE = 400          # Optimal chunk size for retrieval
CHUNK_OVERLAP = 50        # Context continuity between chunks  
TOP_K_RETRIEVAL = 5       # Number of chunks to retrieve
SIMILARITY_THRESHOLD = 0.7 # Minimum relevance score
```

### Model Selection
- **gpt-3.5-turbo**: Fast, cost-effective for most use cases
- **gpt-4**: Advanced reasoning for complex questions
- **gpt-4o**: Latest model with enhanced capabilities

---

## 🔮 Roadmap & Coming Soon

### Immediate Priorities
- ✅ **Smart Question Agent** (✅ Completed)
- ✅ **Enhanced RAG with Vector Search** (✅ Completed)  
- ✅ **Streaming Responses** (✅ Completed)
- ✅ **Configurable Summary Depth** (✅ Completed)

### Next Phase
- 🔄 **Multi-Document RAG** - Query across document collections
- 📤 **Export Functionality** - PDF and Markdown export of Q&A sessions
- 🔗 **Shareable Sessions** - Share processed content and conversations
- 📊 **Analytics Dashboard** - Usage insights and content intelligence

### Future Enhancements  
- 💬 **Team Collaboration** - Shared workspaces and permissions
- 🔌 **API Endpoints** - Programmatic access for integrations
- ☁️ **Cloud Deployment** - One-click AWS/GCP deployment
- 🔒 **Enterprise Features** - SSO, audit logs, compliance

---

## 🤝 Contributing

We welcome contributions! Here's how to get started:

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature/amazing-feature`
3. **Make your changes** with tests and documentation
4. **Commit your changes**: `git commit -m 'Add amazing feature'`
5. **Push to your branch**: `git push origin feature/amazing-feature`
6. **Open a Pull Request**

### Development Guidelines
- Follow Python PEP 8 style guidelines
- Add docstrings for new functions
- Test with multiple document types
- Update README for new features

---

## 🚨 Troubleshooting

### Common Issues

**Docker build fails:**
```bash
# Clear Docker cache and rebuild
docker-compose down
docker system prune -f
docker-compose up --build
```

**OpenAI API errors:**
- Verify your API key in `.env` file
- Check API usage limits and billing
- Ensure sufficient credits for your use case

**Audio/Video processing issues:**
- Ensure ffmpeg is installed (included in Docker)
- Check file format compatibility
- Verify file size limits

---

## 📄 License

MIT License © 2025 Michael Schwimmer

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software.

---

## 🙋‍♂️ Support & Community

- **🐛 Bug Reports**: [GitHub Issues](https://github.com/schwim23/ai-summarizer/issues)
- **💡 Feature Requests**: [GitHub Discussions](https://github.com/schwim23/ai-summarizer/discussions)  
- **📖 Documentation**: Check the `/docs` folder for detailed guides
- **💬 Community**: Join discussions and share your use cases

---

## 🌟 Acknowledgments

- **OpenAI** for powerful language models and APIs
- **Gradio** for the excellent web interface framework
- **FAISS** for efficient vector similarity search
- **Open Source Community** for the amazing libraries that make this possible

---

**⭐ If MyAIGist helps you work smarter, please star the repository and share it with others!**

*Built with ❤️ for researchers, professionals, and anyone who wants to have intelligent conversations with their content.*