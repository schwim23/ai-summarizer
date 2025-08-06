# Central config for tool parameters

# TTS Configuration
USE_OPENAI_TTS = True  # Changed to True - OpenAI TTS is higher quality
USE_GTTS_TTS = False   # Fallback option

# RAG Configuration - Aligned with utils.py
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIM = 1536
DEFAULT_CHUNK_SIZE = 400  # Aligned with utils.py optimal size
CHUNK_OVERLAP = 50        # Added overlap for better context
DEFAULT_TOP_K = 5         # Increased for better retrieval
SIMILARITY_THRESHOLD = 0.7

# Model Selection
DEFAULT_SUMMARY_MODEL = "gpt-3.5-turbo"
DEFAULT_QA_MODEL = "gpt-4"

# Audio Configuration
AUDIO_OUTPUT_DIR = "/app/audio"
SUPPORTED_AUDIO_FORMATS = [".mp3", ".mp4", ".wav", ".m4a"]
SUPPORTED_DOC_FORMATS = [".pdf", ".docx", ".txt"]

# Processing Limits
MAX_CONTEXT_LENGTH = 4000
MAX_SUMMARY_LENGTH = 2000
MAX_QUESTION_LENGTH = 500