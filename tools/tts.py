import os
import uuid
import tempfile
from typing import Optional
from .tool_config import USE_OPENAI_TTS, USE_GTTS_TTS, AUDIO_OUTPUT_DIR

# Conditional imports based on configuration
if USE_OPENAI_TTS:
    try:
        from openai import OpenAI
        openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        OPENAI_AVAILABLE = True
    except ImportError:
        OPENAI_AVAILABLE = False
        print("Warning: OpenAI not available, falling back to gTTS")

if USE_GTTS_TTS or not OPENAI_AVAILABLE:
    try:
        from gtts import gTTS
        GTTS_AVAILABLE = True
    except ImportError:
        GTTS_AVAILABLE = False
        print("Warning: gTTS not available")

def text_to_speech(text: str, output_dir: str = AUDIO_OUTPUT_DIR, voice: str = "nova") -> str:
    """
    Unified text-to-speech function with OpenAI TTS primary and gTTS fallback
    
    Args:
        text: Text to convert to speech
        output_dir: Directory to save audio file
        voice: Voice to use (OpenAI voices: alloy, echo, fable, onyx, nova, shimmer)
    
    Returns:
        Path to generated audio file
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate unique filename
    filename = f"{uuid.uuid4()}.mp3"
    path = os.path.join(output_dir, filename)
    
    # Try OpenAI TTS first (higher quality)
    if USE_OPENAI_TTS and OPENAI_AVAILABLE:
        try:
            print("Using OpenAI TTS...")
            response = openai_client.audio.speech.create(
                model="tts-1",
                voice=voice,
                input=text[:4000]  # OpenAI TTS has text limits
            )
            response.stream_to_file(path)
            print(f"OpenAI TTS successful: {path}")
            return path
        except Exception as e:
            print(f"OpenAI TTS failed: {e}, falling back to gTTS...")
    
    # Fallback to gTTS
    if GTTS_AVAILABLE:
        try:
            print("Using gTTS...")
            # Split long text for gTTS (it has limits too)
            if len(text) > 5000:
                text = text[:5000] + "..."
            
            tts = gTTS(text=text, lang='en', slow=False)
            tts.save(path)
            print(f"gTTS successful: {path}")
            return path
        except Exception as e:
            print(f"gTTS failed: {e}")
            raise RuntimeError(f"All TTS methods failed. OpenAI: {USE_OPENAI_TTS and OPENAI_AVAILABLE}, gTTS: {GTTS_AVAILABLE}")
    
    # If we get here, no TTS method worked
    raise RuntimeError("No TTS method available. Please install OpenAI or gTTS dependencies.")

def text_to_speech_openai(text: str, voice: str = "nova") -> str:
    """Direct OpenAI TTS function for compatibility"""
    return text_to_speech(text, voice=voice)

def text_to_speech_gtts(text: str) -> str:
    """Direct gTTS function for compatibility"""
    # Temporarily disable OpenAI for this call
    global USE_OPENAI_TTS
    original_setting = USE_OPENAI_TTS
    USE_OPENAI_TTS = False
    
    try:
        result = text_to_speech(text)
        return result
    finally:
        USE_OPENAI_TTS = original_setting