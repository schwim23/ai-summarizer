
from gtts import gTTS
import os
import uuid

def text_to_speech(text: str, output_dir: str = "/app/audio") -> str:
    os.makedirs(output_dir, exist_ok=True)
    filename = f"{uuid.uuid4()}.mp3"
    path = os.path.join(output_dir, filename)
    tts = gTTS(text)
    tts.save(path)
    return path
