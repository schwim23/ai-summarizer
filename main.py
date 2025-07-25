
import os
import tempfile
import subprocess
import gradio as gr
from pytube import YouTube
from pydub import AudioSegment
from openai import OpenAI
from PyPDF2 import PdfReader
from docx import Document

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
global_full_text = ""

def extract_text(file):
    name = file.name
    if name.endswith(".pdf"):
        reader = PdfReader(name)
        return "\n".join(page.extract_text() for page in reader.pages if page.extract_text())
    elif name.endswith(".docx"):
        doc = Document(name)
        return "\n".join(p.text for p in doc.paragraphs)
    elif name.endswith(".txt"):
        with open(name, "r") as f:
            return f.read()
    else:
        return "Unsupported text file format."

def download_youtube_audio(url):
    yt = YouTube(url)
    stream = yt.streams.filter(only_audio=True).first()
    temp_audio = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
    stream.download(filename=temp_audio.name)
    return temp_audio.name

def convert_video_to_audio(video_file):
    audio_path = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False).name
    command = ["ffmpeg", "-i", video_file, "-vn", "-acodec", "libmp3lame", "-y", audio_path]
    subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return audio_path

def transcribe_audio(file_path):
    with open(file_path, "rb") as f:
        transcript = client.audio.transcriptions.create(model="whisper-1", file=f)
    return transcript.text

def summarize_text(text, model_name):
    prompt = f"""Summarize the following content into a clear, digestible 1–2 minute summary as if explaining it to an intelligent but busy person. Highlight key takeaways, main arguments, and anything interesting or surprising.
Format the output with a summary header that should be between 3-4 sentences maximum, a list of key points in bullet format and a conclusion where you, as the summarizer write the key points the reader should remember.

For Example:
Summary: Google beat analyst expectations and became the largest search engine in the world. Advertisering and Cloud Revenue grew while costs continued to rise.
Key Points:
- Revenue is up
- Costs are up and there is increased competition
Conclusion: Google is still the market leader in search engines and has a clear advantage in the AI and Cloud industries but competition is heating up. 

Content:
{text[:6000]}"""
    chat_completion = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
    )
    return chat_completion.choices[0].message.content

def text_to_speech(text):
    response = client.audio.speech.create(model="tts-1", voice="nova", input=text)
    path = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False).name
    response.stream_to_file(path)
    return path

def handle_input(youtube_url, uploaded_file, video_audio_file, summary_model):
    global global_full_text
    if youtube_url:
        audio_path = download_youtube_audio(youtube_url)
        text = transcribe_audio(audio_path)
    elif video_audio_file is not None:
        audio_path = convert_video_to_audio(video_audio_file.name)
        text = transcribe_audio(audio_path)
    elif uploaded_file is not None:
        text = extract_text(uploaded_file)
    else:
        return "Please upload a file or provide a YouTube link.", None, "", ""

    global_full_text = text
    summary = summarize_text(text, summary_model)
    audio_path = text_to_speech(summary)
    return summary, audio_path, text, ""

def answer_question(question, qa_model):
    global global_full_text
    if not global_full_text:
        return "No content loaded to answer from.", None

    prompt = f"""You are a helpful assistant. Given the following source content and a question, provide a clear, helpful answer.

Source:
{global_full_text[:10000]}

Question:
{question}

Answer:"""
    chat_completion = client.chat.completions.create(
        model=qa_model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
    )
    answer = chat_completion.choices[0].message.content
    audio_path = text_to_speech(answer)
    return answer, audio_path

with gr.Blocks() as demo:
    gr.Markdown("# 🎙️ AI Summarizer + Q&A + Audio")

    with gr.Row():
        youtube_input = gr.Textbox(label="Paste YouTube URL")
        uploaded_file = gr.File(label="Upload .pdf, .docx, or .txt")
        video_audio_file = gr.File(label="Upload audio/video file (.mp3, .mp4)")

    with gr.Row():
        summary_model_dropdown = gr.Dropdown(
            label="Model for Summarization",
            choices=["gpt-4", "gpt-4o", "gpt-3.5-turbo"],
            value="gpt-4"
        )
        qa_model_dropdown = gr.Dropdown(
            label="Model for Q&A",
            choices=["gpt-4", "gpt-4o", "gpt-3.5-turbo"],
            value="gpt-4"
        )

    with gr.Row():
        submit_btn = gr.Button("Generate Summary")

    output_summary = gr.Textbox(label="Text Summary", lines=10)
    audio_summary = gr.Audio(label="Listen to Summary", autoplay=False)
    hidden_fulltext = gr.Textbox(visible=False)
    user_question = gr.Textbox(label="Ask a follow-up question", placeholder="Type a question here...")
    ask_btn = gr.Button("Ask")
    answer_output = gr.Textbox(label="Answer", lines=8)
    answer_audio = gr.Audio(label="Listen to Answer", autoplay=False)

    submit_btn.click(fn=handle_input,
                     inputs=[youtube_input, uploaded_file, video_audio_file, summary_model_dropdown],
                     outputs=[output_summary, audio_summary, hidden_fulltext, answer_output])

    ask_btn.click(fn=answer_question,
                  inputs=[user_question, qa_model_dropdown],
                  outputs=[answer_output, answer_audio])

demo.launch(server_name="0.0.0.0", server_port=7860)

