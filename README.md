# 🧠 MyAIGist — AI-Powered Content Analysis & Q&A Assistant

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![OpenAI](https://img.shields.io/badge/OpenAI-GPT--4-success)
![Gradio](https://img.shields.io/badge/Gradio-UI-orange)
![Docker](https://img.shields.io/badge/Docker-Containerized-blue)

**MyAIGist** is an intelligent AI-powered assistant that transforms any content into an interactive Q&A experience. Upload documents, transcribe audio/video, or paste text to get instant summaries and ask detailed questions with intelligent follow-up suggestions powered by advanced RAG (Retrieval-Augmented Generation) technology and a Smart Question Agent.

---

## ✨ Latest Updates (v4.0) - Major Performance & UX Overhaul

### 🎨 **Professional UI & Performance Enhancements**
- **⚡ Decoupled Text/Audio Generation**: Text appears instantly while audio generates in background
- **🎙️ Auto-Transcription**: Voice questions automatically update text boxes when transcription completes
- **🎨 Modular CSS Architecture**: 5 organized stylesheets for better maintainability
- **🔧 Fixed Audio Recording Interface**: Resolved text visibility and styling issues in voice components
- **📱 Enhanced Mobile Experience**: Improved responsive design with better touch interactions

### 🚀 **Speed & Performance Improvements**
- **Instant Text Display**: Summaries and answers appear immediately after AI processing
- **Background Audio Processing**: TTS generation doesn't block user interaction
- **Progress Indicators**: Real-time visual feedback during all processing steps
- **Optimized Loading**: Reduced perceived wait times with better user feedback
- **Graceful Fallbacks**: App continues working even if audio generation fails

### 🎯 **Enhanced User Experience**
- **Auto-updating Voice Questions**: Transcribed audio immediately populates text fields
- **Visual Status Indicators**: Clear feedback for processing, success, and error states
- **Smooth Animations**: Professional micro-interactions and hover effects
- **Better Error Handling**: Informative error messages with suggested solutions
- **Professional Dark Theme**: Enhanced gradients and glassmorphism effects

---

## 🚀 Core Features

### 📁 **Multi-Modal Content Support**
- **📄 Document Upload**: PDFs, DOCX, TXT files with intelligent parsing
- **🎵 Audio/Video Transcription**: MP3, MP4, WAV, M4A support with Whisper AI
- **✍️ Direct Text Input**: Paste articles, notes, or any text content
- **🎤 Voice Questions**: Ask questions via audio recording with auto-transcription

### 🧠 **Advanced AI Processing**
- **Smart Question Analysis**: Automatically improves and clarifies questions
- **RAG-Enhanced Answers**: Retrieval-augmented generation for accurate responses
- **Configurable AI Models**: Choose between GPT-3.5-turbo, GPT-4, or GPT-4o
- **Multiple Summary Depths**: Quick Overview, Balanced Summary, or Deep Analysis
- **Instant Text + Background Audio**: Text appears immediately, audio follows

### 💬 **Intelligent Q&A System**
- **Context-Aware Responses**: Maintains conversation flow and memory
- **Source Attribution**: Shows which document sections were used
- **Follow-up Suggestions**: AI-generated relevant follow-up questions
- **Multi-format Support**: Text and voice input for questions with auto-transcription
- **Conversation Memory**: Builds on previous questions naturally

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

### 3. Set Up CSS (Important!)

```bash
# Run the CSS setup script (creates organized stylesheets)
python setup_css.py
```

### 4. Launch Application

```bash
docker-compose up --build
```

🎉 **Access MyAIGist at:** [http://localhost:7860](http://localhost:7860)

### 5. Development Mode (Optional)

For live code changes during development:

```bash
docker-compose -f docker-compose.dev.yml up --build
```

---

## 🛠 Advanced Tech Stack

| Component | Technology | Purpose |
|-----------|------------|---------|
| **UI Framework** | Gradio 4.28+ | Modern web interface with dark theme |
| **AI/LLM** | OpenAI (GPT-4, GPT-3.5, Whisper, TTS) | Core AI processing |
| **Smart Agent** | Custom Question Agent | Intelligent question processing |
| **Vector Search** | FAISS | Semantic document retrieval |
| **Audio Processing** | yt-dlp + Whisper | Content transcription |
| **Document Parsing** | PyPDF2, python-docx | File processing |
| **Text Processing** | tiktoken | Optimized text chunking |
| **TTS System** | OpenAI TTS + gTTS fallback | Audio generation |
| **Styling** | Modular CSS Architecture | Organized component styling |
| **Containerization** | Docker | Easy deployment |

---

## 🎯 How MyAIGist Works

### Step 1: Choose Your Input Method 📁
Choose from three input options via the tabbed interface:
- **📄 Upload text file**: PDF, DOCX, or TXT documents
- **🎵 Upload audio/video**: MP3, MP4, WAV, M4A files for transcription
- **✍️ Enter text**: Direct text input and analysis

### Step 2: AI Analysis & Processing ⚡
- **Smart chunking** creates optimized text segments
- **Vector embeddings** enable semantic search
- **RAG indexing** builds searchable knowledge base
- **Instant summaries** with configurable depth levels
- **Background audio generation** for accessibility

### Step 3: Intelligent Q&A 💬
- **Voice or text questions** with auto-transcription
- **Smart Question Agent** analyzes and improves questions
- **Multi-strategy retrieval** finds relevant context
- **Instant text responses** with background audio generation
- **Source attribution** shows document sections used
- **Follow-up suggestions** guide conversation flow

---

## 💡 Use Cases

- **📚 Research Analysis**: Upload academic papers and get instant insights
- **📄 Document Q&A**: Ask specific questions about contracts, reports, manuals  
- **🎥 Content Learning**: Transform audio/video content into searchable knowledge
- **📊 Meeting Analysis**: Upload transcripts and extract key decisions
- **🎓 Study Assistant**: Interactive learning with any educational content
- **💼 Business Intelligence**: Analyze reports and extract actionable insights
- **📰 News Analysis**: Quick summaries and Q&A with articles and updates
- **🎙️ Podcast/Interview Analysis**: Voice-to-text with intelligent questioning

---

## ⚙️ Configuration Options

### Summary Depth Settings
- **Quick Overview**: 2-3 key bullet points for busy users
- **Balanced Summary**: Comprehensive structured analysis (default)
- **Deep Analysis**: Detailed insights with implications and context

### AI Model Selection
- **gpt-3.5-turbo**: Fast, cost-effective for most use cases
- **gpt-4**: Advanced reasoning for complex questions
- **gpt-4o**: Latest model with enhanced capabilities

### RAG Parameters (customizable in `tools/tool_config.py`)
```python
DEFAULT_CHUNK_SIZE = 400        # Optimal chunk size for retrieval
CHUNK_OVERLAP = 50              # Context continuity between chunks  
DEFAULT_TOP_K = 5               # Number of chunks to retrieve
SIMILARITY_THRESHOLD = 0.7      # Minimum relevance score
MAX_CONTEXT_LENGTH = 4000       # Maximum context for questions
```

### TTS Configuration
```python
USE_OPENAI_TTS = True          # Primary TTS system (higher quality)
USE_GTTS_TTS = False           # Fallback option
```

### Performance Settings
```python
# Decoupled processing for faster UX
INSTANT_TEXT_DISPLAY = True    # Show text immediately
BACKGROUND_AUDIO_GEN = True    # Generate audio separately
AUTO_TRANSCRIPTION = True      # Update text from voice input
```

---

## 🏗 Project Structure

```
ai-summarizer/
├── main.py                    # Enhanced Gradio UI with decoupled processing
├── utils.py                   # Core processing with progress callbacks
├── question_agent.py          # Smart Question Agent with diagnostics
├── setup_css.py               # CSS organization script
├── requirements.txt           # Dependencies
├── docker-compose.yml         # Container configuration
├── Dockerfile                 # Container setup
├── static/                    # CSS and static assets
│   └── css/                   # Modular CSS architecture
│       ├── main.css          # Base styling and layout
│       ├── components.css    # UI components styling
│       ├── forms.css         # Form elements styling
│       ├── audio.css         # Audio components (fixed styling)
│       └── animations.css    # Animations and effects
└── tools/                     # Modular tools package
    ├── __init__.py           # Package initialization
    ├── tool_config.py        # Centralized configuration
    ├── tts.py                # Unified text-to-speech
    ├── text_summarizer.py    # Enhanced summarization
    └── chunker.py            # Smart text chunking
```

---

## 🔮 Roadmap & Coming Soon

### Recently Completed ✅
- ✅ **Decoupled Text/Audio Processing** (v4.0)
- ✅ **Auto-Transcription for Voice Input** (v4.0)
- ✅ **Modular CSS Architecture** (v4.0)
- ✅ **Fixed Audio Component Styling** (v4.0)
- ✅ **Enhanced Progress Indicators** (v4.0)
- ✅ **Modern UI Design** (v3.0)
- ✅ **Smart Question Agent** (v3.0)
- ✅ **Enhanced RAG System** (v3.0)  
- ✅ **Unified TTS System** (v3.0)

### Next Phase (v5.0)
- 🔄 **Streaming Responses** - Real-time text generation as LLM processes
- 📤 **Export Functionality** - PDF and Markdown export of Q&A sessions
- 🔗 **Shareable Sessions** - Share processed content and conversations
- 📊 **Analytics Dashboard** - Usage insights and content intelligence
- 🎥 **YouTube Integration** - Direct URL input for video analysis
- 💾 **Session Memory** - Save and resume analysis sessions

### Future Enhancements (v6.0+)
- 💬 **Multi-Document RAG** - Query across document collections
- 🌐 **Real-time Collaboration** - Multiple users on same content
- 🔌 **API Endpoints** - Programmatic access for integrations
- ☁️ **Cloud Deployment** - One-click AWS/GCP deployment
- 🔒 **Enterprise Features** - SSO, audit logs, compliance
- 🌍 **Multi-language Support** - International language processing

---

## 🤝 Contributing

We welcome contributions! Here's how to get started:

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature/amazing-feature`
3. **Make your changes** with tests and documentation
4. **Run CSS setup**: `python setup_css.py` (if modifying styles)
5. **Commit your changes**: `git commit -m 'Add amazing feature'`
6. **Push to your branch**: `git push origin feature/amazing-feature`
7. **Open a Pull Request**

### Development Guidelines
- Follow Python PEP 8 style guidelines
- Add docstrings for new functions
- Test with multiple document types and input methods
- Update README for new features
- Use modular CSS architecture for styling changes
- Ensure mobile responsiveness for UI changes
- Test audio functionality across different browsers

---

## 🚨 Troubleshooting

### Common Issues

**CSS/Styling Issues:**
```bash
# Regenerate CSS files
python setup_css.py

# Clear browser cache and reload
# Check browser developer tools for CSS errors
```

**Audio Recording Problems:**
```bash
# Ensure microphone permissions are granted
# Check browser compatibility (Chrome/Firefox recommended)
# Verify HTTPS for production deployment (required for audio)
```

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

**Performance Issues:**
- Check Docker container memory allocation
- Monitor console logs for errors
- Verify network connectivity for API calls

**Audio/Video processing issues:**
- Ensure ffmpeg is installed (included in Docker)
- Check file format compatibility (.mp3, .mp4, .wav, .m4a)
- Verify file size limits (< 25MB recommended)

---

## 📊 Performance Metrics

### Response Times (Typical)
- **Text Display**: < 2 seconds after processing
- **Audio Transcription**: 3-10 seconds (depending on length)
- **Summary Generation**: 5-15 seconds (varies by model & depth)
- **Q&A Responses**: 2-8 seconds for text, +5-10s for audio
- **RAG Retrieval**: < 1 second for semantic search

### Optimization Features
- ⚡ **Instant Text Display**: No waiting for audio generation
- 🔄 **Background Processing**: Audio generates while user reads
- 📱 **Progressive Loading**: Visual feedback during all operations
- 🎯 **Smart Caching**: Optimized vector similarity search
- 💾 **Memory Management**: Efficient chunk processing

---

## 📄 License

MIT License © 2025 Michael Schwimmer

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software.

---

## 🙋‍♂️ Support & Community

- **🐛 Bug Reports**: [GitHub Issues](https://github.com/schwim23/ai-summarizer/issues)
- **💡 Feature Requests**: [GitHub Discussions](https://github.com/schwim23/ai-summarizer/discussions)  
- **📖 Documentation**: Check the project files for detailed implementation guides
- **💬 Community**: Join discussions and share your use cases

### Getting Help
1. **Check the troubleshooting section** above for common solutions
2. **Run the CSS setup script** if experiencing styling issues: `python setup_css.py`
3. **Review console logs** for detailed error information
4. **Check browser compatibility** - Chrome/Firefox recommended for audio features
5. **Verify API keys and permissions** for OpenAI integration

---

## 🌟 Acknowledgments

- **OpenAI** for powerful language models and APIs
- **Gradio** for the excellent web interface framework
- **FAISS** for efficient vector similarity search
- **Open Source Community** for the amazing libraries that make this possible

---

## 📈 Version History

### v4.0 (Current) - Performance & UX Overhaul
- ⚡ Decoupled text/audio processing for instant responses
- 🎙️ Auto-transcription with immediate text box updates  
- 🎨 Modular CSS architecture with organized stylesheets
- 🔧 Fixed audio recording component styling issues
- 📱 Enhanced mobile responsiveness and animations

### v3.0 - Smart Agent & Modern UI
- 🤖 Smart Question Agent with diagnostic analysis
- 🎨 Professional dark theme with glassmorphism
- 🔄 Enhanced RAG system with multiple search strategies
- 🔊 Unified TTS system with fallback options

### v2.0 - Advanced RAG Implementation
- 🧠 FAISS vector search integration
- 📊 Configurable summary depths
- 🎯 Source attribution and follow-up suggestions

### v1.0 - Initial Release
- 📄 Basic document processing and summarization
- 🎵 Audio/video transcription capabilities
- 💬 Simple Q&A functionality

---

**⭐ If MyAIGist helps you work smarter, please star the repository and share it with others!**

*Built with ❤️ for researchers, professionals, and anyone who wants to have intelligent conversations with their content.*

---

## 🔐 Security & Privacy

- **API Keys**: Stored locally in `.env` file, never transmitted to third parties
- **Document Processing**: All processing happens locally or via secure OpenAI API
- **No Data Storage**: No persistent storage of user documents or conversations
- **HTTPS Ready**: Supports SSL/TLS configuration for production deployment
- **Container Isolation**: Docker provides secure, isolated runtime environment
- **Audio Privacy**: Voice recordings processed locally with Whisper API

---

## 📊 System Requirements

### Minimum Requirements
- **Python**: 3.10+
- **RAM**: 4GB (8GB recommended)
- **Storage**: 2GB free space
- **Network**: Internet connection for OpenAI API
- **Browser**: Chrome/Firefox (for audio features)

### Recommended Setup
- **Python**: 3.11+
- **RAM**: 8GB+ for large documents
- **CPU**: Multi-core processor for faster processing
- **Storage**: SSD for better performance
- **Network**: Stable broadband for audio processing

### Docker Requirements
- **Docker**: 20.10+
- **Docker Compose**: 1.29+
- **Available Ports**: 7860 (configurable)
- **Memory**: 2GB+ allocated to Docker