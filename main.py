import gradio as gr
from utils import handle_input, answer_question

summary_model_choices = ["gpt-3.5-turbo", "gpt-4", "gpt-4o"]

with gr.Blocks(title="My AI Gist") as demo:
    gr.Markdown("# 🧠 My AI Gist\nUpload content or share a YouTube link to get a summary, audio, and ask questions!")

    with gr.Row():
        youtube_url = gr.Textbox(label="📺 YouTube Link", placeholder="https://www.youtube.com/watch?v=...")
        uploaded_file = gr.File(label="📄 Upload Text File (PDF, DOCX, TXT)")
        video_audio_file = gr.File(label="🎞️ Upload Audio or Video", type="filepath")
        raw_text_input = gr.Textbox(label="✍️ Paste Raw Text", lines=4, placeholder="Paste any text content...")

    with gr.Row():
        summary_model = gr.Dropdown(choices=summary_model_choices, value="gpt-3.5-turbo", label="🔧 Summary Model")

    summarize_button = gr.Button("▶️ Summarize")
    summary_output = gr.Textbox(label="📝 Summary", lines=8)
    audio_output = gr.Audio(label="🔊 Summary Audio", autoplay=True)
    full_text_output = gr.Textbox(visible=False)

    with gr.Row():
        question_text = gr.Textbox(label="❓ Ask a Question about the Content")
        question_audio = gr.Audio(label="🎤 Ask via Audio (Optional)", type="filepath")
        question_button = gr.Button("💬 Get Answer")

    answer_output = gr.Textbox(label="🧠 Answer")
    answer_audio_output = gr.Audio(label="🔊 Answer Audio", autoplay=True)

    summarize_button.click(
        fn=handle_input,
        inputs=[youtube_url, uploaded_file, video_audio_file, raw_text_input, summary_model],
        outputs=[summary_output, audio_output, full_text_output, gr.Textbox(visible=False)],
    )

    question_button.click(
        fn=answer_question,
        inputs=[question_text, question_audio, summary_model],
        outputs=[question_text, answer_output, answer_audio_output],
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
