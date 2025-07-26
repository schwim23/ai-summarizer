import os
import tempfile
import subprocess
import faiss
import numpy as np
import tiktoken
from PyPDF2 import PdfReader
from docx import Document
from openai import OpenAI
import yt_dlp
import uuid


client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

embedding_model = "text-embedding-3-small"
embedding_dim = 1536
embedding_index = None
text_chunks = []
global_full_text = ""

def extract_text(file):
    name = file.name
    if name.endswith(".pdf"):
        reader = PdfReader(name)
        return "\n".join(p.extract_text() for p in reader.pages if p.extract_text())
    elif name.endswith(".docx"):
        doc = Document(name)
        return "\n".join(p.text for p in doc.paragraphs)
    elif name.endswith(".txt"):
        return file.read().decode("utf-8")
    return "Unsupported format"


def download_youtube_audio(url):
    temp_dir = tempfile.mkdtemp()
    unique_filename = f"{uuid.uuid4()}.%(ext)s"
    output_path_template = os.path.join(temp_dir, unique_filename)

    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': output_path_template,
        'quiet': True,
        'overwrites': True,
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            output_path = ydl.prepare_filename(info).replace('.webm', '.mp3').replace('.m4a', '.mp3')
        return output_path
    except Exception as e:
        raise RuntimeError(f"yt_dlp failed: {e}")



def transcribe_audio(file_path):
    with open(file_path, "rb") as f:
        return client.audio.transcriptions.create(model="whisper-1", file=f).text

def summarize_text(text, model_name):
    prompt = f"""Summarize the following content into a clear, digestible 1–2 minute summary as if explaining to an intelligent but busy person.
Content:
{text[:6000]}"""
    result = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
    )
    return result.choices[0].message.content

def text_to_speech(text):
    response = client.audio.speech.create(model="tts-1", voice="nova", input=text)
    path = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False).name
    response.stream_to_file(path)
    return path

def chunk_text(text, max_tokens=300):
    tokenizer = tiktoken.encoding_for_model("gpt-4")
    tokens = tokenizer.encode(text)
    return [tokenizer.decode(tokens[i:i+max_tokens]) for i in range(0, len(tokens), max_tokens)]

def get_embedding(text):
    result = client.embeddings.create(model=embedding_model, input=[text])
    return np.array(result.data[0].embedding, dtype=np.float32)

def build_faiss_index(text):
    global embedding_index, text_chunks
    text_chunks = chunk_text(text)
    embeddings = [get_embedding(chunk) for chunk in text_chunks]
    index = faiss.IndexFlatL2(embedding_dim)
    index.add(np.array(embeddings))
    embedding_index = index

def retrieve_context(query, top_k=4):
    global embedding_index, text_chunks
    if not embedding_index or not text_chunks:
        return global_full_text[:4000]
    query_vec = get_embedding(query).reshape(1, -1)
    _, indices = embedding_index.search(query_vec, top_k)
    return "\n---\n".join([text_chunks[i] for i in indices[0] if i != -1])

def handle_input(youtube_url, uploaded_file, video_audio_file, raw_text_input, summary_model):
    global global_full_text
    if raw_text_input:
        text = raw_text_input
    elif youtube_url:
        audio_path = download_youtube_audio(youtube_url)
        text = transcribe_audio(audio_path)
    elif video_audio_file:
        audio_path = video_audio_file.name
        text = transcribe_audio(audio_path)
    elif uploaded_file:
        text = extract_text(uploaded_file)
    else:
        return "Please upload a file or paste text.", None, "", ""
    global_full_text = text
    build_faiss_index(text)
    summary = summarize_text(text, summary_model)
    audio_summary = text_to_speech(summary)
    return summary, audio_summary, text, ""

def answer_question(question_text, audio_path, model_name):
    if not question_text and audio_path:
        question_text = transcribe_audio(audio_path)
    if not question_text:
        return "No question provided.", "Please ask something.", None
    context = retrieve_context(question_text)
    prompt = f"Use the following context to answer concisely:\n\nContext:\n{context}\n\nQuestion:\n{question_text}"
    result = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
    )
    answer = result.choices[0].message.content
    audio = text_to_speech(answer)
    return question_text, answer, audio
