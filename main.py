import gradio as gr
from utils import handle_input, answer_question, transcribe_audio, text_to_speech
import time
import os

summary_model_choices = ["gpt-3.5-turbo", "gpt-4", "gpt-4o"]
summary_depth_choices = ["Quick Overview", "Balanced Summary", "Deep Analysis"]

# Load CSS files
def load_css_files():
    """Load all CSS files and combine them"""
    css_files = [
        "static/css/main.css",
        "static/css/components.css", 
        "static/css/forms.css",
        "static/css/audio.css",
        "static/css/animations.css"
    ]
    
    combined_css = ""
    
    for css_file in css_files:
        try:
            with open(css_file, 'r', encoding='utf-8') as f:
                combined_css += f.read() + "\n"
        except FileNotFoundError:
            print(f"Warning: CSS file {css_file} not found. Using inline fallback.")
            # Fallback to inline CSS if files don't exist
            combined_css += get_fallback_css()
            break
    
    return combined_css

def get_fallback_css():
    """Fallback CSS in case external files aren't available"""
    return """
    /* Fallback CSS - Essential styling */
    body, html {
        background: linear-gradient(135deg, #0f0f23 0%, #1a1a2e 30%, #16213e 70%, #0f1419 100%) !important;
        color: #ffffff !important;
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Inter', sans-serif !important;
    }
    
    .gradio-container {
        max-width: 1200px !important;
        margin: 0 auto !important;
        background: transparent !important;
        color: #ffffff !important;
        padding: 0 20px !important;
    }
    
    /* Audio Recording Component - FIXED STYLING */
    .gr-audio,
    .gradio-audio,
    .audio-container {
        background: linear-gradient(135deg, rgba(20,20,40,0.9) 0%, rgba(30,30,60,0.9) 100%) !important;
        border: 2px solid rgba(99, 102, 241, 0.4) !important;
        border-radius: 16px !important;
        padding: 1.5rem !important;
        color: #ffffff !important;
    }
    
    .gr-audio *,
    .gradio-audio *,
    .gr-audio .gr-text,
    .gradio-audio .gr-text,
    .gr-audio span,
    .gradio-audio span,
    .gr-audio p,
    .gradio-audio p,
    .gr-audio .label,
    .gradio-audio .label,
    .gr-audio .gr-label,
    .gradio-audio .gr-label {
        color: #ffffff !important;
        text-shadow: none !important;
    }
    
    .gr-audio .gr-button,
    .gradio-audio .gr-button {
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%) !important;
        border: none !important;
        border-radius: 12px !important;
        color: #ffffff !important;
        font-weight: 600 !important;
        padding: 12px 24px !important;
        box-shadow: 0 4px 12px rgba(99, 102, 241, 0.3) !important;
    }
    
    .brand-header {
        text-align: center;
        padding: 3rem 0 2rem 0;
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 25%, #a855f7 50%, #d946ef 75%, #ec4899 100%);
        margin: -20px -20px 2rem -20px;
        border-radius: 0 0 24px 24px;
        box-shadow: 0 20px 40px rgba(99, 102, 241, 0.4);
    }
    
    .input-container, .settings-section, .output-section {
        background: linear-gradient(135deg, rgba(30,30,60,0.9) 0%, rgba(40,40,80,0.8) 100%) !important;
        border: 1px solid rgba(99, 102, 241, 0.4) !important;
        border-radius: 24px !important;
        padding: 3rem !important;
        margin-bottom: 2rem !important;
        box-shadow: 0 12px 40px rgba(0,0,0,0.4) !important;
    }
    
    .primary-button {
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 50%, #a855f7 100%) !important;
        border: none !important;
        border-radius: 18px !important;
        color: white !important;
        font-weight: 700 !important;
        padding: 22px 55px !important;
        font-size: 1.35rem !important;
        width: 100% !important;
        max-width: 380px !important;
        margin: 3rem auto !important;
        display: block !important;
        cursor: pointer !important;
        text-transform: uppercase !important;
        letter-spacing: 1.2px !important;
    }
    """

# Create CSS directories if they don't exist
def ensure_css_directories():
    """Ensure CSS directory structure exists"""
    css_dir = "static/css"
    os.makedirs(css_dir, exist_ok=True)

# Initialize CSS
ensure_css_directories()
custom_css = load_css_files()

with gr.Blocks(title="MyAIGist - AI-Powered Q&A Assistant", css=custom_css, theme=gr.themes.Base()) as demo:
    
    # State to track if content has been processed
    content_processed = gr.State(False)
    
    # Brand Header with Logo
    with gr.Row():
        gr.HTML("""
            <div class="brand-header">
                <div class="logo-container">
                    <div class="logo-icon">🧠</div>
                    <div class="logo-text">MYAIGIST</div>
                </div>
                <h1 class="main-title">AI-Powered Content Analysis & Q&A</h1>
                <p class="subtitle">Upload documents, analyze content, and get intelligent answers</p>
            </div>
        """)
    
    # Three horizontal tabs
    with gr.Row():
        with gr.Column():
            with gr.Group(elem_classes=["tab-container"]):
                with gr.Row():
                    tab_document_btn = gr.Button("📄 Upload text file", elem_classes=["tab-button"])
                    tab_audio_btn = gr.Button("🎵 Upload audio/video", elem_classes=["tab-button"]) 
                    tab_text_btn = gr.Button("✍️ Enter text", elem_classes=["tab-button", "selected"])
    
    # Main input area
    with gr.Row():
        with gr.Column():
            with gr.Group(elem_classes=["input-container"]):
                # Document file upload (hidden by default)
                uploaded_file = gr.File(
                    label="📄 Upload Document (PDF, DOCX, TXT)",
                    file_types=[".pdf", ".docx", ".txt"],
                    elem_classes=["file-upload"],
                    visible=False
                )
                
                # Audio file upload (hidden by default)  
                audio_file = gr.File(
                    label="🎵 Upload Audio/Video File (MP3, MP4, WAV, M4A)",
                    file_types=[".mp3", ".mp4", ".wav", ".m4a"],
                    elem_classes=["file-upload"],
                    visible=False
                )
                
                # Text input (visible by default)
                raw_text_input = gr.Textbox(
                    label="",
                    placeholder="Enter your text here for AI analysis and intelligent Q&A...",
                    lines=12,
                    max_lines=20,
                    elem_classes=["main-textarea"],
                    show_label=False,
                    visible=True
                )
    
    # Settings section
    with gr.Row():
        with gr.Column():
            with gr.Group(elem_classes=["settings-section"]):
                gr.HTML('<div class="settings-title">⚙️ AI Configuration</div>')
                with gr.Row():
                    with gr.Column(scale=1):
                        summary_model = gr.Dropdown(
                            choices=summary_model_choices,
                            value="gpt-3.5-turbo",
                            label="🤖 AI Model"
                        )
                    with gr.Column(scale=1):
                        summary_depth = gr.Dropdown(
                            choices=summary_depth_choices,
                            value="Balanced Summary",
                            label="📊 Analysis Depth"
                        )
    
    # Process button
    with gr.Row():
        process_button = gr.Button(
            "🚀 Analyze Content",
            elem_classes=["primary-button"],
            size="lg"
        )
    
    # Progress indicator
    with gr.Row(visible=False) as progress_row:
        with gr.Column():
            progress_html = gr.HTML()
    
    # Status indicator
    with gr.Row():
        status_output = gr.HTML(
            value='<div class="status-indicator">Ready to process your content</div>',
            visible=True
        )
    
    # Summary output (hidden by default)
    with gr.Row(visible=False) as summary_section:
        with gr.Column():
            with gr.Group(elem_classes=["output-section"]):
                gr.HTML('<div class="section-title">📋 Content Summary</div>')
                summary_output = gr.Textbox(
                    label="",
                    lines=8,
                    elem_classes=["summary-text"],
                    show_label=False,
                    interactive=False,
                    placeholder="Your AI-generated summary will appear here..."
                )
                
                # Audio status indicator for summary
                summary_audio_status = gr.HTML(
                    value='<div class="status-indicator audio-generating" style="display: none;"><div class="spinner"></div>Generating audio...</div>',
                    visible=False
                )
                
                audio_output = gr.Audio(
                    label="🔊 Listen to Summary",
                    autoplay=False
                )
    
    # Q&A Section (hidden by default)
    with gr.Row(visible=False) as qa_section:
        with gr.Column():
            with gr.Group(elem_classes=["output-section"]):
                gr.HTML('<div class="section-title">💬 Smart Q&A Assistant</div>')
                
                with gr.Row():
                    with gr.Column(scale=3):
                        question_text = gr.Textbox(
                            label="",
                            placeholder="Ask anything about your content...",
                            lines=2,
                            elem_classes=["question-input"],
                            show_label=False
                        )
                    with gr.Column(scale=1):
                        question_audio = gr.Audio(
                            label="🎤 Voice Question",
                            type="filepath",
                            elem_classes=["audio-container"]
                        )
                
                ask_button = gr.Button(
                    "🤖 Get AI Answer",
                    elem_classes=["primary-button"]
                )
                
                # Q&A Progress indicator
                with gr.Row(visible=False) as qa_progress_row:
                    with gr.Column():
                        qa_progress_html = gr.HTML()
                
                answer_output = gr.Textbox(
                    label="",
                    lines=8,
                    elem_classes=["summary-text"],
                    show_label=False,
                    interactive=False,
                    placeholder="Your intelligent AI answer will appear here..."
                )
                
                # Audio status indicator for Q&A
                answer_audio_status = gr.HTML(
                    value='<div class="status-indicator audio-generating" style="display: none;"><div class="spinner"></div>Generating audio...</div>',
                    visible=False
                )
                
                answer_audio_output = gr.Audio(
                    label="🔊 Listen to Answer",
                    autoplay=False
                )
    
    # Hidden components for compatibility
    full_text_output = gr.Textbox(visible=False)
    youtube_url = gr.Textbox(value="", visible=False)
    
    # Audio transcription function
    def transcribe_question_audio(audio_path):
        """Transcribe audio and update text box immediately"""
        if audio_path is None:
            return ""
        
        try:
            transcribed_text = transcribe_audio(audio_path)
            return transcribed_text
        except Exception as e:
            print(f"Transcription error: {e}")
            return ""
    
    # Tab switching functions
    def show_document_tab():
        return (
            gr.update(visible=True),   # uploaded_file
            gr.update(visible=False),  # audio_file  
            gr.update(visible=False),  # raw_text_input
            gr.update(elem_classes=["tab-button", "selected"]),  # tab_document_btn
            gr.update(elem_classes=["tab-button"]),              # tab_audio_btn
            gr.update(elem_classes=["tab-button"])               # tab_text_btn
        )
    
    def show_audio_tab():
        return (
            gr.update(visible=False),  # uploaded_file
            gr.update(visible=True),   # audio_file
            gr.update(visible=False),  # raw_text_input
            gr.update(elem_classes=["tab-button"]),              # tab_document_btn
            gr.update(elem_classes=["tab-button", "selected"]),  # tab_audio_btn
            gr.update(elem_classes=["tab-button"])               # tab_text_btn
        )
    
    def show_text_tab():
        return (
            gr.update(visible=False),  # uploaded_file
            gr.update(visible=False),  # audio_file
            gr.update(visible=True),   # raw_text_input
            gr.update(elem_classes=["tab-button"]),              # tab_document_btn
            gr.update(elem_classes=["tab-button"]),              # tab_audio_btn
            gr.update(elem_classes=["tab-button", "selected"])   # tab_text_btn
        )
    
    # Event handlers for tabs
    tab_document_btn.click(
        fn=show_document_tab,
        outputs=[uploaded_file, audio_file, raw_text_input, tab_document_btn, tab_audio_btn, tab_text_btn]
    )
    
    tab_audio_btn.click(
        fn=show_audio_tab,
        outputs=[uploaded_file, audio_file, raw_text_input, tab_document_btn, tab_audio_btn, tab_text_btn]
    )
    
    tab_text_btn.click(
        fn=show_text_tab,
        outputs=[uploaded_file, audio_file, raw_text_input, tab_document_btn, tab_audio_btn, tab_text_btn]
    )
    
    # Audio transcription handler - updates text immediately
    question_audio.change(
        fn=transcribe_question_audio,
        inputs=[question_audio],
        outputs=[question_text]
    )
    
    # Processing function with decoupled text/audio
    def process_content_decoupled(youtube_url_val, uploaded_file_val, audio_file_val, raw_text_val, model, depth):
        """Process content with immediate text display and background audio generation"""
        
        # Show initial progress
        yield (
            gr.update(visible=True),  # progress_row
            '''<div class="progress-container">
                <div class="progress-bar">
                    <div class="progress-fill" style="width: 10%;"></div>
                </div>
                <div class="progress-text">
                    <div class="spinner"></div>
                    Starting content processing...
                </div>
            </div>''',
            '<div class="status-indicator processing">🔄 Processing...</div>',
            "",  # summary_output
            gr.update(visible=False),  # summary_audio_status
            None,  # audio_output
            "",   # full_text_output
            False,  # content_processed
            gr.update(visible=False),  # summary_section
            gr.update(visible=False)   # qa_section
        )
        
        # Simulate progress steps
        progress_steps = [
            (25, "Extracting and cleaning content..."),
            (45, "Building semantic knowledge base..."),
            (65, "Generating AI summary..."),
        ]
        
        for i, (progress, message) in enumerate(progress_steps):
            time.sleep(0.6)  # Reduced time for faster feedback
            
            progress_html = f'''<div class="progress-container">
                <div class="progress-bar">
                    <div class="progress-fill" style="width: {progress}%;"></div>
                </div>
                <div class="progress-text">
                    <div class="spinner"></div>
                    {message}
                </div>
            </div>'''
            
            yield (
                gr.update(visible=True),  # progress_row
                progress_html,
                '<div class="status-indicator processing">🔄 Processing...</div>',
                "",  # summary_output
                gr.update(visible=False),  # summary_audio_status
                None,  # audio_output
                "",   # full_text_output
                False,  # content_processed
                gr.update(visible=False),  # summary_section
                gr.update(visible=False)   # qa_section
            )
        
        try:
            # Get text content first (without audio)
            summary, _, full_text, status = handle_input(
                youtube_url_val, uploaded_file_val, audio_file_val, raw_text_val, model, depth
            )
            
            # Show text immediately
            if summary and "Error" not in summary and "Please upload" not in summary:
                yield (
                    gr.update(visible=False),  # progress_row
                    "",  # progress_html
                    '<div class="status-indicator success">✅ Content processed! Generating audio...</div>',
                    summary,  # summary_output
                    gr.update(visible=True),  # summary_audio_status
                    None,  # audio_output (still empty)
                    full_text,   # full_text_output
                    True,  # content_processed
                    gr.update(visible=True),   # summary_section
                    gr.update(visible=True)    # qa_section
                )
                
                # Generate audio in background
                try:
                    audio_summary = text_to_speech(summary)
                    # Update with audio when ready
                    yield (
                        gr.update(visible=False),  # progress_row
                        "",  # progress_html
                        '<div class="status-indicator success">✅ Content processed successfully! Ready for Q&A.</div>',
                        summary,  # summary_output
                        gr.update(visible=False),  # summary_audio_status
                        audio_summary,  # audio_output
                        full_text,   # full_text_output
                        True,  # content_processed
                        gr.update(visible=True),   # summary_section
                        gr.update(visible=True)    # qa_section
                    )
                except Exception as audio_error:
                    print(f"Audio generation failed: {audio_error}")
                    # Continue without audio
                    yield (
                        gr.update(visible=False),  # progress_row
                        "",  # progress_html
                        '<div class="status-indicator success">✅ Content processed successfully! (Audio generation failed)</div>',
                        summary,  # summary_output
                        gr.update(visible=False),  # summary_audio_status
                        None,  # audio_output
                        full_text,   # full_text_output
                        True,  # content_processed
                        gr.update(visible=True),   # summary_section
                        gr.update(visible=True)    # qa_section
                    )
            else:
                yield (
                    gr.update(visible=False),  # progress_row
                    "",  # progress_html
                    '<div class="status-indicator error">❌ Please provide valid content to analyze.</div>',
                    summary or "Please provide content to analyze.",  # summary_output
                    gr.update(visible=False),  # summary_audio_status
                    None,  # audio_output
                    "",   # full_text_output
                    False,  # content_processed
                    gr.update(visible=False),  # summary_section
                    gr.update(visible=False)   # qa_section
                )
                
        except Exception as e:
            yield (
                gr.update(visible=False),  # progress_row
                "",  # progress_html
                f'<div class="status-indicator error">❌ Processing failed: {str(e)}</div>',
                f"Error processing content: {str(e)}",  # summary_output
                gr.update(visible=False),  # summary_audio_status
                None,  # audio_output
                "",   # full_text_output
                False,  # content_processed
                gr.update(visible=False),  # summary_section
                gr.update(visible=False)   # qa_section
            )
    
    # Q&A function with decoupled text/audio
    def answer_question_decoupled(question_text_val, audio_path_val, model, is_processed):
        """Answer questions with immediate text display and background audio generation"""
        if not is_processed:
            return (
                question_text_val, 
                "Please process some content first before asking questions.", 
                gr.update(visible=False),  # answer_audio_status
                None,  # answer_audio_output
                gr.update(visible=False),  # qa_progress_row
                ""
            )
        
        # Show Q&A progress
        qa_steps = [
            "Analyzing question context...",
            "Retrieving relevant information...", 
            "Generating intelligent response..."
        ]
        
        for i, step in enumerate(qa_steps):
            progress = int((i + 1) / len(qa_steps) * 100)
            progress_html = f'''<div class="progress-container">
                <div class="progress-bar">
                    <div class="progress-fill" style="width: {progress}%;"></div>
                </div>
                <div class="progress-text">
                    <div class="spinner"></div>
                    {step}
                </div>
            </div>'''
            
            yield (
                question_text_val,
                "",  # answer_output
                gr.update(visible=False),  # answer_audio_status
                None,  # answer_audio_output
                gr.update(visible=True),  # qa_progress_row
                progress_html
            )
            time.sleep(0.5)  # Reduced time for faster feedback
        
        try:
            # Get text answer first (without audio)
            processed_question, answer, _ = answer_question(question_text_val, audio_path_val, model)
            
            # Show text immediately
            yield (
                processed_question, 
                answer, 
                gr.update(visible=True),  # answer_audio_status
                None,  # answer_audio_output (still empty)
                gr.update(visible=False),  # qa_progress_row
                ""
            )
            
            # Generate audio in background
            try:
                audio_response = text_to_speech(answer)
                # Update with audio when ready
                yield (
                    processed_question, 
                    answer, 
                    gr.update(visible=False),  # answer_audio_status
                    audio_response,  # answer_audio_output
                    gr.update(visible=False),  # qa_progress_row
                    ""
                )
            except Exception as audio_error:
                print(f"Audio generation failed: {audio_error}")
                # Continue without audio
                yield (
                    processed_question, 
                    answer, 
                    gr.update(visible=False),  # answer_audio_status
                    None,  # answer_audio_output
                    gr.update(visible=False),  # qa_progress_row
                    ""
                )
            
        except Exception as e:
            yield (
                question_text_val, 
                f"Error generating answer: {str(e)}", 
                gr.update(visible=False),  # answer_audio_status
                None,  # answer_audio_output
                gr.update(visible=False),  # qa_progress_row
                ""
            )
    
    # Process content event with decoupled text/audio
    process_button.click(
        fn=process_content_decoupled,
        inputs=[youtube_url, uploaded_file, audio_file, raw_text_input, summary_model, summary_depth],
        outputs=[progress_row, progress_html, status_output, summary_output, summary_audio_status, audio_output, full_text_output, content_processed, summary_section, qa_section]
    )
    
    # Ask questions event with decoupled text/audio
    ask_button.click(
        fn=answer_question_decoupled,
        inputs=[question_text, question_audio, summary_model, content_processed],
        outputs=[question_text, answer_output, answer_audio_status, answer_audio_output, qa_progress_row, qa_progress_html]
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)