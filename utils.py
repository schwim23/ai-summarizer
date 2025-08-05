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
import re
from typing import List, Dict, Tuple
import json
from question_agent import SmartQuestionAgent

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# Enhanced RAG Configuration
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIM = 1536
CHUNK_SIZE = 400  # Optimal chunk size for retrieval
CHUNK_OVERLAP = 50  # Overlap between chunks
TOP_K_RETRIEVAL = 5  # Number of chunks to retrieve
SIMILARITY_THRESHOLD = 0.7  # Minimum similarity score

# Global RAG state
vector_store = None
text_chunks = []
chunk_metadata = []
global_full_text = ""
document_title = ""

# Initialize Smart Question Agent
question_agent = SmartQuestionAgent()

class EnhancedRAGVectorStore:
    """Enhanced vector store with metadata and improved retrieval"""
    
    def __init__(self):
        self.index = None
        self.chunks = []
        self.metadata = []
        self.embeddings = []
        
    def add_documents(self, chunks: List[str], metadata: List[Dict] = None):
        """Add documents with metadata to the vector store"""
        if metadata is None:
            metadata = [{"chunk_id": i} for i in range(len(chunks))]
            
        # Generate embeddings for all chunks
        embeddings = []
        for chunk in chunks:
            embedding = self._get_embedding(chunk)
            embeddings.append(embedding)
            
        # Create FAISS index
        embeddings_array = np.array(embeddings, dtype=np.float32)
        self.index = faiss.IndexFlatL2(EMBEDDING_DIM)
        self.index.add(embeddings_array)
        
        # Store chunks and metadata
        self.chunks = chunks
        self.metadata = metadata
        self.embeddings = embeddings
        
    def similarity_search(self, query: str, k: int = TOP_K_RETRIEVAL) -> List[Dict]:
        """Perform similarity search and return results with metadata"""
        if not self.index or not self.chunks:
            return []
            
        # Get query embedding
        query_embedding = self._get_embedding(query).reshape(1, -1)
        
        # Search for similar chunks
        distances, indices = self.index.search(query_embedding, k)
        
        results = []
        for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
            if idx == -1:  # Invalid index
                continue
                
            # Convert distance to similarity score (0-1)
            similarity_score = 1 / (1 + distance)
            
            # Only include results above threshold
            if similarity_score >= SIMILARITY_THRESHOLD:
                results.append({
                    'content': self.chunks[idx],
                    'metadata': self.metadata[idx],
                    'similarity_score': similarity_score,
                    'rank': i + 1
                })
                
        return results
    
    def _get_embedding(self, text: str) -> np.ndarray:
        """Generate embedding for text"""
        result = client.embeddings.create(model=EMBEDDING_MODEL, input=[text])
        return np.array(result.data[0].embedding, dtype=np.float32)

def extract_text(file):
    """Enhanced text extraction with better formatting"""
    name = file.name
    text = ""
    
    if name.endswith(".pdf"):
        reader = PdfReader(name)
        text = "\n".join(p.extract_text() for p in reader.pages if p.extract_text())
    elif name.endswith(".docx"):
        doc = Document(name)
        text = "\n".join(p.text for p in doc.paragraphs)
    elif name.endswith(".txt"):
        text = file.read().decode("utf-8")
    else:
        return "Unsupported format"
    
    # Clean up the text
    text = clean_text(text)
    return text

def clean_text(text: str) -> str:
    """Clean and normalize text for better processing"""
    # Remove excessive whitespace
    text = re.sub(r'\n+', '\n', text)
    text = re.sub(r' +', ' ', text)
    
    # Remove special characters but keep punctuation
    text = re.sub(r'[^\w\s\.\,\!\?\;\:\-\(\)]', '', text)
    
    # Normalize line breaks
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    
    return text.strip()

def smart_chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[Dict]:
    """Smart text chunking with sentence boundaries and overlap"""
    # Split into sentences first
    sentences = re.split(r'(?<=[.!?])\s+', text)
    
    chunks = []
    current_chunk = ""
    current_length = 0
    
    tokenizer = tiktoken.encoding_for_model("gpt-4")
    
    for i, sentence in enumerate(sentences):
        sentence_tokens = len(tokenizer.encode(sentence))
        
        # If adding this sentence would exceed chunk size, save current chunk
        if current_length + sentence_tokens > chunk_size and current_chunk:
            chunks.append({
                'content': current_chunk.strip(),
                'start_sentence': len(chunks) * 10,  # Approximate
                'end_sentence': i,
                'token_count': current_length
            })
            
            # Start new chunk with overlap
            overlap_text = get_overlap_text(current_chunk, overlap, tokenizer)
            current_chunk = overlap_text + " " + sentence
            current_length = len(tokenizer.encode(current_chunk))
        else:
            current_chunk += " " + sentence if current_chunk else sentence
            current_length += sentence_tokens
    
    # Add the last chunk
    if current_chunk:
        chunks.append({
            'content': current_chunk.strip(),
            'start_sentence': len(chunks) * 10,
            'end_sentence': len(sentences),
            'token_count': current_length
        })
    
    return chunks

def get_overlap_text(text: str, overlap_tokens: int, tokenizer) -> str:
    """Get the last N tokens from text for overlap"""
    tokens = tokenizer.encode(text)
    if len(tokens) <= overlap_tokens:
        return text
    
    overlap_tokens_list = tokens[-overlap_tokens:]
    return tokenizer.decode(overlap_tokens_list)

def download_youtube_audio(url):
    """Download YouTube audio with better error handling"""
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
            title = info.get('title', 'YouTube Video')
            output_path = ydl.prepare_filename(info).replace('.webm', '.mp3').replace('.m4a', '.mp3')
        return output_path, title
    except Exception as e:
        raise RuntimeError(f"yt_dlp failed: {e}")

def transcribe_audio(file_path):
    """Transcribe audio with better formatting"""
    with open(file_path, "rb") as f:
        result = client.audio.transcriptions.create(model="whisper-1", file=f)
        return clean_text(result.text)

def enhanced_summarize_text(text: str, model_name: str, summary_depth: str = "Balanced Summary") -> str:
    """Enhanced summarization with configurable depth"""
    
    # Define prompts for different summary depths
    depth_prompts = {
        "Quick Overview": f"""Provide a brief, concise summary of the following content in 2-3 key bullet points:

Content:
{text[:6000]}

Quick Summary (2-3 bullet points):""",
        
        "Balanced Summary": f"""Provide a comprehensive summary of the following content. Structure your summary with:
1. Main Topic/Theme
2. Key Points (3-5 bullet points)
3. Important Details
4. Conclusions/Takeaways

Content:
{text[:8000]}

Summary:""",
        
        "Deep Analysis": f"""Provide a detailed, comprehensive analysis of the following content. Include:
1. Executive Summary
2. Main Topics and Themes
3. Detailed Key Points (5-8 points)
4. Supporting Details and Evidence
5. Context and Background
6. Implications and Conclusions
7. Additional Insights

Content:
{text[:10000]}

Deep Analysis:"""
    }
    
    prompt = depth_prompts.get(summary_depth, depth_prompts["Balanced Summary"])
    
    # Adjust temperature based on depth
    temperature = {
        "Quick Overview": 0.1,      # More focused
        "Balanced Summary": 0.3,    # Standard
        "Deep Analysis": 0.4        # More creative
    }.get(summary_depth, 0.3)
    
    result = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        stream=True  # Enable streaming
    )
    
    # Collect streamed response
    full_response = ""
    for chunk in result:
        if chunk.choices[0].delta.content:
            full_response += chunk.choices[0].delta.content
    
    return full_response

def text_to_speech(text):
    """Generate speech from text"""
    response = client.audio.speech.create(model="tts-1", voice="nova", input=text)
    path = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False).name
    response.stream_to_file(path)
    return path

def build_rag_index(text: str, title: str = "Document"):
    """Build enhanced RAG index with metadata"""
    global vector_store, text_chunks, chunk_metadata, document_title, question_agent
    
    document_title = title
    
    # Clear agent memory for new document
    question_agent.clear_memory()
    print("DEBUG - Cleared question agent memory for new document")
    
    # Smart chunking
    chunks_data = smart_chunk_text(text)
    
    # Extract content and create metadata
    text_chunks = [chunk['content'] for chunk in chunks_data]
    chunk_metadata = []
    
    for i, chunk_data in enumerate(chunks_data):
        metadata = {
            'chunk_id': i,
            'document_title': title,
            'token_count': chunk_data['token_count'],
            'start_sentence': chunk_data.get('start_sentence', 0),
            'end_sentence': chunk_data.get('end_sentence', 0),
            'chunk_type': 'content'
        }
        chunk_metadata.append(metadata)
    
    # Initialize and populate vector store
    vector_store = EnhancedRAGVectorStore()
    vector_store.add_documents(text_chunks, chunk_metadata)
    
    print(f"RAG Index built: {len(text_chunks)} chunks from '{title}'")
    print(f"Smart Question Agent ready for '{title}'")

def enhanced_retrieve_context(query: str, max_context_length: int = 4000) -> Dict:
    """Enhanced context retrieval with ranking and filtering"""
    global vector_store
    
    if not vector_store:
        return {
            'context': global_full_text[:max_context_length],
            'sources': [],
            'retrieval_info': 'No RAG index available'
        }
    
    # Perform similarity search
    results = vector_store.similarity_search(query, k=TOP_K_RETRIEVAL)
    
    if not results:
        return {
            'context': global_full_text[:max_context_length],
            'sources': [],
            'retrieval_info': 'No relevant chunks found'
        }
    
    # Build context from top results
    context_parts = []
    sources = []
    total_length = 0
    
    tokenizer = tiktoken.encoding_for_model("gpt-4")
    
    for result in results:
        chunk_content = result['content']
        chunk_tokens = len(tokenizer.encode(chunk_content))
        
        if total_length + chunk_tokens <= max_context_length:
            context_parts.append(f"[Source {result['rank']}]: {chunk_content}")
            sources.append({
                'rank': result['rank'],
                'similarity_score': result['similarity_score'],
                'metadata': result['metadata']
            })
            total_length += chunk_tokens
        else:
            break
    
    context = "\n\n".join(context_parts)
    
    retrieval_info = f"Retrieved {len(sources)} relevant chunks (avg similarity: {np.mean([s['similarity_score'] for s in sources]):.3f})"
    
    return {
        'context': context,
        'sources': sources,
        'retrieval_info': retrieval_info
    }

def handle_input(youtube_url, uploaded_file, video_audio_file, raw_text_input, summary_model, summary_depth="Balanced Summary"):
    """Enhanced input handling with RAG indexing and configurable summary depth"""
    global global_full_text
    
    title = "Document"
    
    try:
        if raw_text_input:
            text = raw_text_input
            title = "Raw Text Input"
        elif youtube_url:
            audio_path, video_title = download_youtube_audio(youtube_url)
            text = transcribe_audio(audio_path)
            title = video_title
        elif video_audio_file:
            audio_path = video_audio_file.name
            text = transcribe_audio(audio_path)
            title = os.path.basename(audio_path)
        elif uploaded_file:
            text = extract_text(uploaded_file)
            title = os.path.basename(uploaded_file.name)
        else:
            return "Please upload a file or paste text.", None, "", ""
        
        global_full_text = text
        
        # Build RAG index
        build_rag_index(text, title)
        
        # Generate enhanced summary with configurable depth
        summary = enhanced_summarize_text(text, summary_model, summary_depth)
        
        # Generate audio
        audio_summary = text_to_speech(summary)
        
        return summary, audio_summary, text, f"✅ RAG Index built with {len(text_chunks)} chunks"
        
    except Exception as e:
        return f"Error processing content: {str(e)}", None, "", ""

def enhanced_answer_question(question_text, audio_path, model_name):
    """Enhanced Q&A with Smart Question Agent and RAG retrieval"""
    
    # Debug: Print what we received
    print(f"DEBUG - question_text: '{question_text}' (type: {type(question_text)})")
    print(f"DEBUG - audio_path: '{audio_path}' (type: {type(audio_path)})")
    
    # Handle different input scenarios
    actual_question = ""
    
    # Priority logic: if there's audio input, always use it (new question)
    # If no audio, then use text input
    if audio_path and audio_path is not None:
        print("DEBUG - Audio provided, transcribing...")
        actual_question = transcribe_audio(audio_path)
        print(f"DEBUG - Transcribed question: '{actual_question}'")
    elif question_text and str(question_text).strip():
        actual_question = str(question_text).strip()
        print(f"DEBUG - Using text question: '{actual_question}'")
    
    # Check if we have a valid question
    if not actual_question:
        print("DEBUG - No valid question found")
        return "No question provided.", "Please ask something about your uploaded content.", None

    try:
        # 🤖 NEW: Use Smart Question Agent to process the question
        print("DEBUG - Processing question with Smart Agent...")
        agent_result = question_agent.process_question(actual_question, global_full_text[:1000])
        print(f"DEBUG - Agent result type: {agent_result.get('type')}")
        
        # Handle different agent response types
        if agent_result.get("type") == "clarification_needed":
            # Agent needs clarification
            clarification_response = f"""I need some clarification to give you the best answer:

{agent_result.get('clarifying_questions', '')}

Would you like to rephrase your question or choose one of these alternatives?"""
            
            audio_response = text_to_speech(clarification_response)
            return actual_question, clarification_response, audio_response
        
        elif agent_result.get("type") == "multi_step_plan":
            # Agent broke down complex question
            planning_response = f"""This is a complex question. Let me break it down:

{agent_result.get('sub_questions', '')}

I'll work through these systematically to give you a comprehensive answer."""
            
            # Continue with RAG processing using original question
            processed_question = actual_question
            
        elif agent_result.get("type") == "direct_with_suggestions":
            # Agent has suggestions but will proceed
            processed_question = actual_question
            
        elif agent_result.get("type") == "contextual":
            # Agent enhanced question with context
            processed_question = agent_result.get("enhanced_question", actual_question)
            
        else:
            # Direct processing
            processed_question = agent_result.get("processed_question", actual_question)
        
        # Retrieve relevant context using RAG
        retrieval_result = enhanced_retrieve_context(processed_question)
        context = retrieval_result['context']
        sources = retrieval_result['sources']
        retrieval_info = retrieval_result['retrieval_info']
        
        print(f"DEBUG - Retrieved context length: {len(context)}")
        print(f"DEBUG - Number of sources: {len(sources)}")
        
        # Enhanced prompt with source attribution
        prompt = f"""You are an AI assistant answering questions based on the provided context. Use the context to provide accurate, detailed answers.

Context from {document_title}:
{context}

Question: {processed_question}

Instructions:
1. Answer the question using primarily the provided context
2. Be specific and detailed when the context allows
3. If the context doesn't fully address the question, mention what information is missing
4. Reference relevant source sections when applicable

Answer:"""
        
        # Generate answer with streaming
        result = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            stream=True  # Enable streaming
        )
        
        # Collect streamed response
        answer = ""
        for chunk in result:
            if chunk.choices[0].delta.content:
                answer += chunk.choices[0].delta.content
        
        # Add agent suggestions if available
        if agent_result.get("type") == "direct_with_suggestions":
            suggestions = agent_result.get("suggested_alternatives", [])
            if suggestions:
                answer += f"\n\n💡 **Alternative questions you might ask:**\n"
                for i, suggestion in enumerate(suggestions[:3], 1):
                    answer += f"{i}. {suggestion}\n"
        
        # Add source information
        if sources:
            source_info = f"\n\n📚 Sources used ({len(sources)} chunks, {retrieval_info})"
            answer += source_info
        
        # 🤖 NEW: Generate smart follow-up questions
        followups = question_agent.get_suggested_followups(processed_question, answer)
        if followups:
            answer += f"\n\n🔄 **Suggested follow-up questions:**\n"
            for i, followup in enumerate(followups[:3], 1):
                answer += f"{i}. {followup}\n"
        
        # Generate audio
        audio_response = text_to_speech(answer)
        
        return actual_question, answer, audio_response
        
    except Exception as e:
        print(f"DEBUG - Error: {str(e)}")
        return actual_question, f"Error generating answer: {str(e)}", None

# Backwards compatibility - keep the old function names
answer_question = enhanced_answer_question
summarize_text = enhanced_summarize_text