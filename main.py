import gradio as gr
from utils import handle_input, answer_question

summary_model_choices = ["gpt-3.5-turbo", "gpt-4", "gpt-4o"]
summary_depth_choices = ["Quick Overview", "Balanced Summary", "Deep Analysis"]

# Custom CSS emphasizing Q&A functionality
custom_css = """
/* Main container styling */
.gradio-container {
    max-width: 1400px !important;
    margin: 0 auto !important;
    background: linear-gradient(135deg, #f8faff 0%, #f0f4ff 100%) !important;
}

/* Header styling - emphasizing Q&A */
.header-container {
    text-align: center;
    padding: 2.5rem 0;
    background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
    border-radius: 20px;
    margin-bottom: 2rem;
    color: white;
    position: relative;
    overflow: hidden;
}

.header-container::before {
    content: '';
    position: absolute;
    top: -50%;
    left: -50%;
    width: 200%;
    height: 200%;
    background: radial-gradient(circle, rgba(255,255,255,0.1) 0%, transparent 70%);
    animation: pulse 4s ease-in-out infinite;
}

@keyframes pulse {
    0%, 100% { transform: scale(1); opacity: 0.3; }
    50% { transform: scale(1.1); opacity: 0.1; }
}

.logo-text {
    font-size: 3rem;
    font-weight: 800;
    margin: 0;
    letter-spacing: -0.02em;
    position: relative;
    z-index: 2;
}

.subtitle {
    font-size: 1.3rem;
    opacity: 0.95;
    margin-top: 0.5rem;
    font-weight: 500;
    position: relative;
    z-index: 2;
}

.tagline {
    font-size: 1rem;
    opacity: 0.8;
    margin-top: 0.25rem;
    font-style: italic;
    position: relative;
    z-index: 2;
}

/* Main workflow styling */
.workflow-container {
    display: flex;
    gap: 1.5rem;
    margin-bottom: 2rem;
}

.workflow-step {
    flex: 1;
    background: white;
    border-radius: 16px;
    padding: 1.5rem;
    box-shadow: 0 4px 20px rgba(99, 102, 241, 0.1);
    border: 2px solid transparent;
    transition: all 0.3s ease;
}

.workflow-step:hover {
    border-color: #6366f1;
    transform: translateY(-2px);
    box-shadow: 0 8px 30px rgba(99, 102, 241, 0.15);
}

.step-number {
    width: 40px;
    height: 40px;
    background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    color: white;
    font-weight: 700;
    font-size: 1.2rem;
    margin-bottom: 1rem;
}

.step-title {
    font-size: 1.1rem;
    font-weight: 600;
    color: #1e293b;
    margin-bottom: 0.5rem;
}

.step-description {
    color: #64748b;
    font-size: 0.9rem;
    line-height: 1.4;
}

/* Input section styling */
.input-section {
    background: white;
    border-radius: 16px;
    padding: 2rem;
    box-shadow: 0 4px 20px rgba(99, 102, 241, 0.1);
    margin-bottom: 1.5rem;
    border: 1px solid #e2e8f0;
}

.section-title {
    font-size: 1.4rem;
    font-weight: 700;
    color: #1e293b;
    margin-bottom: 1.5rem;
    display: flex;
    align-items: center;
    gap: 0.75rem;
    padding-bottom: 0.5rem;
    border-bottom: 2px solid #f1f5f9;
}

/* Q&A emphasis styling */
.qa-highlight {
    background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
    color: white;
    padding: 1.5rem;
    border-radius: 16px;
    margin-bottom: 1.5rem;
    text-align: center;
}

.qa-title {
    font-size: 1.8rem;
    font-weight: 700;
    margin-bottom: 0.5rem;
}

.qa-subtitle {
    font-size: 1rem;
    opacity: 0.9;
}

/* Button styling */
.primary-button {
    background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%) !important;
    border: none !important;
    border-radius: 12px !important;
    color: white !important;
    font-weight: 600 !important;
    padding: 1rem 2.5rem !important;
    font-size: 1.1rem !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 4px 15px rgba(99, 102, 241, 0.3) !important;
    width: 100% !important;
}

.primary-button:hover {
    transform: translateY(-3px) !important;
    box-shadow: 0 8px 25px rgba(99, 102, 241, 0.4) !important;
}

.secondary-button {
    background: white !important;
    border: 2px solid #6366f1 !important;
    border-radius: 12px !important;
    color: #6366f1 !important;
    font-weight: 600 !important;
    padding: 0.75rem 2rem !important;
    font-size: 1rem !important;
    transition: all 0.3s ease !important;
}

.secondary-button:hover {
    background: #6366f1 !important;
    color: white !important;
    transform: translateY(-2px) !important;
}

/* Output styling */
.output-section {
    background: white;
    border-radius: 16px;
    padding: 2rem;
    box-shadow: 0 4px 20px rgba(99, 102, 241, 0.1);
    margin-bottom: 1.5rem;
    border: 1px solid #e2e8f0;
}

.summary-card {
    background: linear-gradient(135deg, #f8faff 0%, #f0f4ff 100%);
    border: 2px solid #e0e7ff;
    border-radius: 12px;
    padding: 1.5rem;
    margin-bottom: 1rem;
}

.answer-card {
    background: linear-gradient(135deg, #fefce8 0%, #fef3c7 100%);
    border: 2px solid #fde047;
    border-radius: 12px;
    padding: 1.5rem;
    margin-bottom: 1rem;
}

/* Form field styling */
.gradio-textbox, .gradio-dropdown, .gradio-file {
    border-radius: 12px !important;
    border: 2px solid #e2e8f0 !important;
    transition: all 0.3s ease !important;
    font-size: 1rem !important;
}

.gradio-textbox:focus, .gradio-dropdown:focus {
    border-color: #6366f1 !important;
    box-shadow: 0 0 0 4px rgba(99, 102, 241, 0.1) !important;
    transform: translateY(-1px) !important;
}

/* Audio player styling */
audio {
    border-radius: 12px !important;
    box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1) !important;
    width: 100% !important;
}

/* Status indicators */
.status-indicator {
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
    padding: 0.5rem 1rem;
    border-radius: 8px;
    font-size: 0.9rem;
    font-weight: 500;
    margin-bottom: 1rem;
}

.status-ready {
    background: #dcfce7;
    color: #166534;
    border: 1px solid #bbf7d0;
}

.status-processing {
    background: #fef3c7;
    color: #92400e;
    border: 1px solid #fde047;
}

/* Mobile responsiveness */
@media (max-width: 1024px) {
    .workflow-container {
        flex-direction: column;
        gap: 1rem;
    }
    
    .gradio-container {
        max-width: 95% !important;
    }
}

@media (max-width: 768px) {
    .logo-text {
        font-size: 2.2rem;
    }
    
    .subtitle {
        font-size: 1.1rem;
    }
    
    .input-section, .output-section {
        padding: 1.5rem;
        margin-bottom: 1rem;
    }
    
    .workflow-step {
        padding: 1rem;
    }
}

/* Hide Gradio footer */
.footer {
    display: none !important;
}

/* Custom scrollbar */
::-webkit-scrollbar {
    width: 8px;
}

::-webkit-scrollbar-track {
    background: #f1f5f9;
    border-radius: 4px;
}

::-webkit-scrollbar-thumb {
    background: #6366f1;
    border-radius: 4px;
}

::-webkit-scrollbar-thumb:hover {
    background: #4f46e5;
}
"""

with gr.Blocks(title="MyAIGist - AI-Powered Q&A Assistant", css=custom_css, theme=gr.themes.Soft()) as demo:
    
    # Enhanced Header with Q&A emphasis
    with gr.Row():
        gr.HTML("""
            <div class="header-container">
                <h1 class="logo-text">🧠 MYAIGIST</h1>
                <p class="subtitle">AI-Powered Q&A Assistant</p>
                <p class="tagline">Get answers from your content with the help of AI</p>
            </div>
        """)
    
    # Workflow visualization
    with gr.Row():
        gr.HTML("""
            <div class="workflow-container">
                <div class="workflow-step">
                    <div class="step-number">1</div>
                    <div class="step-title">📁 Upload Content</div>
                    <div class="step-description">Upload documents, paste text, or share YouTube links</div>
                </div>
                <div class="workflow-step">
                    <div class="step-number">2</div>
                    <div class="step-title">⚡ AI Processing</div>
                    <div class="step-description">AI analyzes and creates searchable knowledge base</div>
                </div>
                <div class="workflow-step">
                    <div class="step-number">3</div>
                    <div class="step-title">💬 Ask Questions</div>
                    <div class="step-description">Get instant answers with audio playback</div>
                </div>
            </div>
        """)
    
    # Content Upload Section
    with gr.Group():
        with gr.Row():
            gr.HTML('<div class="section-title">📥 Upload Your Content</div>')
        
        with gr.Row():
            with gr.Column(scale=1):
                youtube_url = gr.Textbox(
                    label="🎥 YouTube URL - Paste any YouTube video URL for AI transcription", 
                    placeholder="https://www.youtube.com/watch?v=...",
                    lines=1
                )
                raw_text_input = gr.Textbox(
                    label="✍️ Raw Text Input - Paste articles, notes, or any text content", 
                    placeholder="Paste your text content here for analysis...",
                    lines=6
                )
            
            with gr.Column(scale=1):
                uploaded_file = gr.File(
                    label="📄 Document Upload - PDF, Word docs, or text files",
                    file_types=[".pdf", ".docx", ".txt"]
                )
                video_audio_file = gr.File(
                    label="🎵 Audio/Video File - Upload for transcription",
                    file_types=[".mp3", ".mp4", ".wav", ".m4a"]
                )
        
        # Model and depth selection with process button
        with gr.Row():
            with gr.Column(scale=1):
                summary_model = gr.Dropdown(
                    choices=summary_model_choices, 
                    value="gpt-3.5-turbo", 
                    label="🤖 AI Model - Choose processing model"
                )
            with gr.Column(scale=1):
                summary_depth = gr.Dropdown(
                    choices=summary_depth_choices,
                    value="Balanced Summary",
                    label="📊 Summary Depth - Choose detail level"
                )
            with gr.Column(scale=1):
                process_button = gr.Button(
                    "🚀 Process Content", 
                    elem_classes=["primary-button"],
                    size="lg"
                )
    
    # Processing Status
    with gr.Row():
        status_display = gr.HTML('<div class="status-indicator status-ready">✅ Ready to process content</div>')
    
    # Summary Output (Secondary)
    with gr.Group():
        with gr.Row():
            gr.HTML('<div class="section-title">📋 Content Summary</div>')
        
        with gr.Row():
            with gr.Column(scale=2):
                summary_output = gr.Textbox(
                    label="📝 AI-Generated Summary", 
                    lines=6,
                    placeholder="A concise summary of your content will appear here...",
                    elem_classes=["summary-card"]
                )
            with gr.Column(scale=1):
                audio_output = gr.Audio(
                    label="🔊 Summary Audio - Listen to your summary",
                    autoplay=False
                )
    
    # Q&A Section (Primary Focus)
    with gr.Group():
        with gr.Row():
            gr.HTML("""
                <div class="qa-highlight">
                    <div class="qa-title">💬 Smart Q&A with AI Agent</div>
                    <div class="qa-subtitle">Ask questions and get intelligent responses with follow-up suggestions!</div>
                </div>
            """)
        
        with gr.Row():
            with gr.Column(scale=3):
                question_text = gr.Textbox(
                    label="❓ What would you like to know? - AI agent will help clarify and improve your questions",
                    placeholder="Ask anything about your uploaded content...",
                    lines=3
                )
            with gr.Column(scale=1):
                question_audio = gr.Audio(
                    label="🎤 Voice Question - Or ask by voice",
                    type="filepath"
                )
        
        with gr.Row():
            ask_button = gr.Button(
                "🤖 Get Smart AI Answer", 
                elem_classes=["primary-button"],
                size="lg"
            )
        
        with gr.Row():
            with gr.Column(scale=2):
                answer_output = gr.Textbox(
                    label="🤖 Smart AI Answer - Enhanced with suggestions and follow-ups",
                    lines=10,
                    placeholder="Your AI-powered answer with smart suggestions will appear here...",
                    elem_classes=["answer-card"]
                )
            with gr.Column(scale=1):
                answer_audio_output = gr.Audio(
                    label="🔊 Answer Audio - Listen to the answer",
                    autoplay=False
                )
    
    # Quick Example Questions
    with gr.Group():
        with gr.Row():
            gr.HTML('<div class="section-title">💡 Example Questions</div>')
        
        with gr.Row():
            example_buttons = []
            example_questions = [
                "What are the main points?",
                "Can you explain this in simple terms?", 
                "What are the key takeaways?",
                "How does this relate to...?"
            ]
            
            for question in example_questions:
                btn = gr.Button(
                    question, 
                    elem_classes=["secondary-button"],
                    size="sm"
                )
                example_buttons.append(btn)
    
    # Hidden field for full text
    full_text_output = gr.Textbox(visible=False)
    
    # Event handlers
    def update_status_processing():
        return '<div class="status-indicator status-processing">⏳ Processing your content...</div>'
    
    def update_status_ready():
        return '<div class="status-indicator status-ready">✅ Content processed! Ready for questions.</div>'
    
    # Ask questions with proper input handling
    def process_question_and_clear(question_text_input, audio_input, model):
        # Debug what we're getting
        print(f"MAIN DEBUG - Received question_text_input: '{question_text_input}' (type: {type(question_text_input)})")
        print(f"MAIN DEBUG - Received audio_input: '{audio_input}' (type: {type(audio_input)})")
        
        # Get the answer - this handles both text and audio input
        processed_question, answer, audio_response = answer_question(question_text_input, audio_input, model)
        
        # Return the processed question (for display), answer, and audio
        return processed_question, answer, audio_response
    
    # Process content
    process_button.click(
        fn=update_status_processing,
        outputs=[status_display]
    ).then(
        fn=handle_input,
        inputs=[youtube_url, uploaded_file, video_audio_file, raw_text_input, summary_model, summary_depth],
        outputs=[summary_output, audio_output, full_text_output, gr.Textbox(visible=False)],
    ).then(
        fn=update_status_ready,
        outputs=[status_display]
    )

    # Ask questions
    ask_button.click(
        fn=process_question_and_clear,
        inputs=[question_text, question_audio, summary_model],
        outputs=[question_text, answer_output, answer_audio_output],
    )
    
    # Example question handlers
    for i, btn in enumerate(example_buttons):
        btn.click(
            fn=lambda q=example_questions[i]: q,
            outputs=[question_text]
        )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)