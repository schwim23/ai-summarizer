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
from typing import List, Dict, Tuple, Optional, Callable
import json
from question_agent import SmartQuestionAgent
import time

# Import from tools
from tools.tool_config import (
    EMBEDDING_MODEL, EMBEDDING_DIM, DEFAULT_CHUNK_SIZE, CHUNK_OVERLAP,
    DEFAULT_TOP_K, SIMILARITY_THRESHOLD, MAX_CONTEXT_LENGTH,
    USE_OPENAI_TTS, DEFAULT_SUMMARY_MODEL
)
from tools.tts import text_to_speech as tools_tts
from tools.text_summarizer import summarize_with_depth
from tools.chunker import smart_chunk_text as tools_smart_chunk

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# Enhanced RAG Configuration
CHUNK_SIZE = DEFAULT_CHUNK_SIZE
TOP_K_RETRIEVAL = DEFAULT_TOP_K
# Lower similarity threshold for better retrieval
SIMILARITY_THRESHOLD_OVERRIDE = 0.3

# Global RAG state
vector_store = None
text_chunks = []
chunk_metadata = []
global_full_text = ""
document_title = ""

# Initialize Smart Question Agent
question_agent = SmartQuestionAgent()

def debug_chunks_content():
    """Debug function to examine what's in our chunks - CONSOLE ONLY"""
    global text_chunks, document_title
    
    print(f"\n=== DEBUG CHUNKS CONTENT ===")
    print(f"Document: {document_title}")
    print(f"Total chunks: {len(text_chunks)}")
    
    for i, chunk in enumerate(text_chunks[:5]):  # Show first 5 chunks
        print(f"\n--- Chunk {i} (length: {len(chunk)}) ---")
        print(chunk[:200] + "..." if len(chunk) > 200 else chunk)
    
    print(f"=== END DEBUG CHUNKS ===\n")

class EnhancedRAGVectorStore:
    """Enhanced vector store with aggressive retrieval settings"""
    
    def __init__(self, progress_callback: Optional[Callable] = None):
        self.index = None
        self.chunks = []
        self.metadata = []
        self.embeddings = []
        self.progress_callback = progress_callback
        
    def add_documents(self, chunks: List[str], metadata: List[Dict] = None):
        """Add documents with metadata to the vector store"""
        if metadata is None:
            metadata = [{"chunk_id": i} for i in range(len(chunks))]
            
        print(f"DEBUG - Adding {len(chunks)} chunks to vector store")
        if self.progress_callback:
            self.progress_callback("Generating embeddings for semantic search...")
        
        # Generate embeddings for all chunks
        embeddings = []
        total_chunks = len(chunks)
        
        for i, chunk in enumerate(chunks):
            embedding = self._get_embedding(chunk)
            embeddings.append(embedding)
            print(f"DEBUG - Generated embedding for chunk {i+1}/{total_chunks}: {len(chunk)} chars")
            
            # Progress callback for embedding generation
            if self.progress_callback and i % max(1, total_chunks // 10) == 0:
                progress_pct = int((i / total_chunks) * 30) + 40  # 40-70% range
                self.progress_callback(f"Processing chunk {i+1}/{total_chunks}...")
            
        if self.progress_callback:
            self.progress_callback("Building vector index for fast retrieval...")
            
        # Create FAISS index
        embeddings_array = np.array(embeddings, dtype=np.float32)
        self.index = faiss.IndexFlatL2(EMBEDDING_DIM)
        self.index.add(embeddings_array)
        
        # Store chunks and metadata
        self.chunks = chunks
        self.metadata = metadata
        self.embeddings = embeddings
        
        print(f"DEBUG - FAISS index created with {len(chunks)} chunks")
        
    def similarity_search(self, query: str, k: int = TOP_K_RETRIEVAL) -> List[Dict]:
        """Perform aggressive similarity search with multiple strategies"""
        if not self.index or not self.chunks:
            print("DEBUG - No index or chunks available")
            return []
            
        print(f"DEBUG - Searching for: '{query}'")
        
        # Strategy 1: Direct search
        results = self._search_with_query(query, k)
        
        # Strategy 2: If no results, try keyword extraction
        if not results:
            print("DEBUG - No direct results, trying keyword search")
            keywords = self._extract_keywords(query)
            if keywords:
                keyword_query = " ".join(keywords)
                print(f"DEBUG - Keyword query: '{keyword_query}'")
                results = self._search_with_query(keyword_query, k)
        
        # Strategy 3: If still no results, try entity-based search
        if not results:
            print("DEBUG - No keyword results, trying entity search")
            entities = self._extract_entities(query)
            if entities:
                entity_query = " ".join(entities)
                print(f"DEBUG - Entity query: '{entity_query}'")
                results = self._search_with_query(entity_query, k)
        
        # Strategy 4: If still nothing, return top chunks by brute force
        if not results:
            print("DEBUG - No entity results, using brute force top chunks")
            results = self._get_top_chunks_brute_force(k)
        
        print(f"DEBUG - Final search results: {len(results)} chunks")
        return results
    
    def _search_with_query(self, query: str, k: int) -> List[Dict]:
        """Perform search with a specific query"""
        try:
            # Get query embedding
            query_embedding = self._get_embedding(query).reshape(1, -1)
            
            # Search for similar chunks
            distances, indices = self.index.search(query_embedding, k)
            
            results = []
            for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
                if idx == -1:
                    continue
                    
                # Convert distance to similarity score
                similarity_score = 1 / (1 + distance)
                
                print(f"DEBUG - Chunk {idx}: distance={distance:.3f}, similarity={similarity_score:.3f}")
                
                # Much more lenient threshold
                if similarity_score >= SIMILARITY_THRESHOLD_OVERRIDE:
                    results.append({
                        'content': self.chunks[idx],
                        'metadata': self.metadata[idx],
                        'similarity_score': similarity_score,
                        'rank': i + 1
                    })
                    
            return results
        except Exception as e:
            print(f"DEBUG - Search error: {e}")
            return []
    
    def _extract_keywords(self, query: str) -> List[str]:
        """Extract meaningful keywords from query"""
        # Remove common words and extract meaningful terms
        stop_words = {'what', 'did', 'say', 'about', 'the', 'have', 'to', 'is', 'are', 'was', 'were', 'a', 'an', 'and', 'or', 'but'}
        words = re.findall(r'\b\w{3,}\b', query.lower())
        keywords = [w for w in words if w not in stop_words]
        return keywords[:5]  # Top 5 keywords
    
    def _extract_entities(self, query: str) -> List[str]:
        """Extract named entities and important terms"""
        # Look for capitalized words and known entities
        entities = []
        
        # Find capitalized words (potential names/companies)
        cap_words = re.findall(r'\b[A-Z][a-z]+\b', query)
        entities.extend(cap_words)
        
        # Known important terms for this document
        important_terms = ['Tesla', 'Samsung', 'Musk', 'deal', 'chip', 'foundry', 'AI', 'Texas', 'TSMC']
        for term in important_terms:
            if term.lower() in query.lower():
                entities.append(term)
        
        return list(set(entities))  # Remove duplicates
    
    def _get_top_chunks_brute_force(self, k: int) -> List[Dict]:
        """Return top chunks as last resort"""
        results = []
        for i in range(min(k, len(self.chunks))):
            results.append({
                'content': self.chunks[i],
                'metadata': self.metadata[i],
                'similarity_score': 0.4,  # Assign a default score
                'rank': i + 1
            })
        return results
    
    def _get_embedding(self, text: str) -> np.ndarray:
        """Generate embedding for text"""
        try:
            result = client.embeddings.create(model=EMBEDDING_MODEL, input=[text])
            return np.array(result.data[0].embedding, dtype=np.float32)
        except Exception as e:
            print(f"DEBUG - Embedding error: {e}")
            # Return zero vector as fallback
            return np.zeros(EMBEDDING_DIM, dtype=np.float32)

def extract_text(file, progress_callback: Optional[Callable] = None):
    """Enhanced text extraction with better formatting and progress"""
    name = file.name
    text = ""
    
    if progress_callback:
        progress_callback("Extracting text from document...")
    
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
    
    if progress_callback:
        progress_callback("Text extraction completed")
        
    return text

def clean_text(text: str) -> str:
    """Clean and normalize text for better processing"""
    if not text:
        return ""
    
    # More gentle cleaning to preserve content
    text = re.sub(r'\n\s*\n', '\n\n', text)
    text = re.sub(r' +', ' ', text)
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    
    return text.strip()

def smart_chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[Dict]:
    """Smart text chunking - use tools version"""
    return tools_smart_chunk(text, chunk_size, overlap)

def download_youtube_audio(url, progress_callback: Optional[Callable] = None):
    """Download YouTube audio with better error handling and progress"""
    if progress_callback:
        progress_callback("Starting YouTube audio download...")
        
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
        if progress_callback:
            progress_callback("Downloading audio from YouTube...")
            
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            title = info.get('title', 'YouTube Video')
            output_path = ydl.prepare_filename(info).replace('.webm', '.mp3').replace('.m4a', '.mp3')
            
        if progress_callback:
            progress_callback("YouTube download completed")
            
        return output_path, title
    except Exception as e:
        raise RuntimeError(f"yt_dlp failed: {e}")

def transcribe_audio(file_path, progress_callback: Optional[Callable] = None):
    """Transcribe audio with better formatting and progress"""
    if progress_callback:
        progress_callback("Transcribing audio with Whisper AI...")
        
    with open(file_path, "rb") as f:
        result = client.audio.transcriptions.create(model="whisper-1", file=f)
        
    if progress_callback:
        progress_callback("Audio transcription completed")
        
    return clean_text(result.text)

def enhanced_summarize_text(text: str, model_name: str, summary_depth: str = "Balanced Summary", 
                          progress_callback: Optional[Callable] = None) -> str:
    """Enhanced summarization using tools system with progress"""
    if progress_callback:
        progress_callback(f"Generating {summary_depth.lower()} with {model_name}...")
    
    result = summarize_with_depth(text, depth=summary_depth, model=model_name)
    
    if progress_callback:
        progress_callback("Summary generation completed")
        
    return result

def text_to_speech(text, progress_callback: Optional[Callable] = None):
    """Generate speech from text using unified TTS system with progress"""
    if progress_callback:
        progress_callback("Converting text to speech...")
    
    result = tools_tts(text)
    
    if progress_callback:
        progress_callback("Audio generation completed")
        
    return result

def build_rag_index(text: str, title: str = "Document", progress_callback: Optional[Callable] = None):
    """Build enhanced RAG index with metadata and progress"""
    global vector_store, text_chunks, chunk_metadata, document_title, question_agent, global_full_text
    
    document_title = title
    global_full_text = text
    
    if progress_callback:
        progress_callback("Initializing AI knowledge base...")
    
    # Clear agent memory for new document
    question_agent.clear_memory()
    print("DEBUG - Cleared question agent memory for new document")
    
    if progress_callback:
        progress_callback("Creating intelligent text chunks...")
    
    # Smart chunking
    chunks_data = smart_chunk_text(text)
    
    # Extract content and create metadata
    text_chunks = [chunk['content'] for chunk in chunks_data]
    chunk_metadata = []
    
    for i, chunk_data in enumerate(chunks_data):
        metadata = {
            'chunk_id': i,
            'document_title': title,
            'token_count': chunk_data.get('token_count', 0),
            'start_sentence': chunk_data.get('start_sentence', 0),
            'end_sentence': chunk_data.get('end_sentence', 0),
            'chunk_type': 'content'
        }
        chunk_metadata.append(metadata)
    
    if progress_callback:
        progress_callback("Building semantic vector index...")
    
    # Initialize and populate vector store
    vector_store = EnhancedRAGVectorStore(progress_callback=progress_callback)
    vector_store.add_documents(text_chunks, chunk_metadata)
    
    print(f"RAG Index built: {len(text_chunks)} chunks from '{title}'")
    print(f"Smart Question Agent ready for '{title}'")
    
    if progress_callback:
        progress_callback("Knowledge base ready for intelligent Q&A")

def enhanced_retrieve_context(query: str, max_context_length: int = MAX_CONTEXT_LENGTH) -> Dict:
    """Enhanced context retrieval with multiple fallback strategies"""
    global vector_store, global_full_text, document_title
    
    print(f"DEBUG - Retrieving context for query: '{query}'")
    print(f"DEBUG - Vector store exists: {vector_store is not None}")
    print(f"DEBUG - Global text length: {len(global_full_text) if global_full_text else 0}")
    
    if not vector_store:
        print("DEBUG - No vector store, using direct text fallback")
        fallback_text = global_full_text[:max_context_length] if global_full_text else ""
        return {
            'context': fallback_text,
            'sources': [],
            'retrieval_info': 'No RAG index - using direct text'
        }
    
    # Perform aggressive similarity search
    results = vector_store.similarity_search(query, k=TOP_K_RETRIEVAL * 2)  # Try more chunks
    print(f"DEBUG - Vector search returned {len(results)} results")
    
    if not results:
        print("DEBUG - No vector results, creating focused fallback")
        fallback_text = create_focused_fallback(query, global_full_text, max_context_length)
        return {
            'context': fallback_text,
            'sources': [],
            'retrieval_info': 'No vector matches - using focused text'
        }
    
    # Build context from results
    context_parts = []
    sources = []
    total_length = 0
    
    tokenizer = tiktoken.encoding_for_model("gpt-4")
    
    for result in results:
        chunk_content = result['content']
        chunk_tokens = len(tokenizer.encode(chunk_content))
        
        print(f"DEBUG - Chunk {result['rank']}: {chunk_tokens} tokens, similarity {result['similarity_score']:.3f}")
        
        if total_length + chunk_tokens <= max_context_length:
            context_parts.append(f"[Source {result['rank']}]: {chunk_content}")
            sources.append({
                'rank': result['rank'],
                'similarity_score': result['similarity_score'],
                'metadata': result['metadata'],
                'token_count': chunk_tokens
            })
            total_length += chunk_tokens
        else:
            # Try to fit partial content
            remaining_tokens = max_context_length - total_length
            if remaining_tokens > 100:
                partial_tokens = tokenizer.encode(chunk_content)[:remaining_tokens-50]
                partial_content = tokenizer.decode(partial_tokens) + "..."
                context_parts.append(f"[Source {result['rank']} (partial)]: {partial_content}")
                sources.append({
                    'rank': result['rank'],
                    'similarity_score': result['similarity_score'],
                    'metadata': result['metadata'],
                    'token_count': remaining_tokens,
                    'partial': True
                })
            break
    
    context = "\n\n".join(context_parts)
    
    # Enhanced retrieval info
    avg_similarity = np.mean([s['similarity_score'] for s in sources]) if sources else 0
    retrieval_info = f"Retrieved {len(sources)} chunks ({total_length} tokens, avg similarity: {avg_similarity:.3f})"
    
    print(f"DEBUG - Final context length: {len(context)}")
    print(f"DEBUG - Number of sources used: {len(sources)}")
    
    return {
        'context': context,
        'sources': sources,
        'retrieval_info': retrieval_info,
        'total_tokens': total_length
    }

def create_focused_fallback(query: str, full_text: str, max_length: int) -> str:
    """Create a focused excerpt from full text based on query terms"""
    if not full_text:
        return ""
    
    # Extract key terms from query
    query_terms = re.findall(r'\b\w{3,}\b', query.lower())
    
    # Find paragraphs containing query terms
    paragraphs = full_text.split('\n\n')
    scored_paragraphs = []
    
    for i, para in enumerate(paragraphs):
        para_lower = para.lower()
        score = sum(para_lower.count(term) for term in query_terms)
        if score > 0:
            scored_paragraphs.append((score, i, para))
    
    # Sort by relevance and take top paragraphs
    scored_paragraphs.sort(reverse=True)
    
    # Build focused context
    context_parts = []
    current_length = 0
    
    for score, idx, para in scored_paragraphs:
        if current_length + len(para) <= max_length:
            context_parts.append(para)
            current_length += len(para)
        else:
            break
    
    result = "\n\n".join(context_parts) if context_parts else full_text[:max_length]
    print(f"DEBUG - Focused fallback created: {len(result)} chars from {len(scored_paragraphs)} relevant paragraphs")
    return result

def handle_input(youtube_url, uploaded_file, video_audio_file, raw_text_input, summary_model, summary_depth="Balanced Summary", progress_callback: Optional[Callable] = None):
    """Enhanced input handling with RAG indexing and progress tracking"""
    global global_full_text
    
    title = "Document"
    
    try:
        if progress_callback:
            progress_callback("Starting content processing...")
            
        if raw_text_input:
            text = raw_text_input
            title = "Raw Text Input"
            if progress_callback:
                progress_callback("Processing text input...")
        elif youtube_url:
            audio_path, video_title = download_youtube_audio(youtube_url, progress_callback)
            text = transcribe_audio(audio_path, progress_callback)
            title = video_title
        elif video_audio_file:
            audio_path = video_audio_file.name
            text = transcribe_audio(audio_path, progress_callback)
            title = os.path.basename(audio_path)
        elif uploaded_file:
            text = extract_text(uploaded_file, progress_callback)
            title = os.path.basename(uploaded_file.name)
        else:
            return "Please upload a file or paste text.", None, "", "Please provide content to summarize."
        
        # Store globally before building index
        global_full_text = text
        
        # Build RAG index with progress
        build_rag_index(text, title, progress_callback)
        
        # Generate enhanced summary with progress
        summary = enhanced_summarize_text(text, summary_model, summary_depth, progress_callback)
        
        # Generate audio with progress
        audio_summary = text_to_speech(summary, progress_callback)
        
        if progress_callback:
            progress_callback("Processing completed successfully!")
        
        # Return clean success message for UI
        return summary, audio_summary, text, f"✅ Content processed successfully ({len(text_chunks)} sections indexed)"
        
    except Exception as e:
        print(f"ERROR in handle_input: {str(e)}")  # Log to console
        if progress_callback:
            progress_callback(f"Error: {str(e)}")
        return f"Error processing content. Please try again.", None, "", "❌ Processing failed"

def enhanced_answer_question(question_text, audio_path, model_name, progress_callback: Optional[Callable] = None):
    """Enhanced Q&A with progress tracking"""
    
    # Console logging only
    print(f"\n=== STARTING Q&A PROCESS ===")
    print(f"DEBUG - question_text: '{question_text}' (type: {type(question_text)})")
    print(f"DEBUG - audio_path: '{audio_path}' (type: {type(audio_path)})")
    
    if progress_callback:
        progress_callback("Processing question...")
    
    # Handle input
    actual_question = ""
    
    if audio_path and str(audio_path) != "None" and audio_path is not None:
        try:
            print("DEBUG - Audio provided, transcribing...")
            if progress_callback:
                progress_callback("Transcribing audio question...")
            actual_question = transcribe_audio(audio_path)
            print(f"DEBUG - Transcribed question: '{actual_question}'")
        except Exception as e:
            print(f"DEBUG - Audio transcription failed: {e}")
            if question_text and str(question_text).strip():
                actual_question = str(question_text).strip()
    elif question_text and str(question_text).strip() and str(question_text) != "None":
        actual_question = str(question_text).strip()
        print(f"DEBUG - Using text question: '{actual_question}'")
    
    if not actual_question:
        print("DEBUG - No valid question found")
        return "No question provided.", "Please ask a question about your content.", None

    try:
        if progress_callback:
            progress_callback("Analyzing question context...")
            
        # Debug our chunks (console only)
        debug_chunks_content()
        
        # Context diagnostic (console only)
        context_preview = global_full_text[:2000] if global_full_text else ""
        
        print(f"\n=== CONTEXT DIAGNOSTIC ===")
        print(f"Global text length: {len(global_full_text)}")
        print(f"Context preview length: {len(context_preview)}")
        print(f"Context preview content: '{context_preview[:300]}...'")
        
        # Check if context contains key terms
        question_terms = ['tesla', 'samsung', 'musk', 'deal', 'elon']
        context_lower = context_preview.lower()
        
        print(f"Question terms in context:")
        for term in question_terms:
            count = context_lower.count(term.lower())
            print(f"  '{term}': {count} occurrences")
        
        print(f"=== END CONTEXT DIAGNOSTIC ===\n")
        
        if progress_callback:
            progress_callback("Processing with Smart Question Agent...")
            
        # Process with Smart Agent
        print("DEBUG - Processing question with Smart Agent...")
        agent_result = question_agent.process_question(actual_question, context_preview)
        print(f"DEBUG - Agent result type: {agent_result.get('type')}")
        
        # Handle agent results
        if agent_result.get("type") == "direct":
            processed_question = actual_question
            print("DEBUG - Using direct processing")
        elif agent_result.get("type") == "contextual":
            processed_question = agent_result.get("enhanced_question", actual_question)
            print("DEBUG - Using contextual processing")
        elif agent_result.get("type") == "clarification_needed":
            print("DEBUG - Agent requesting clarification")
            print(f"DEBUG - Clarification request: {agent_result.get('clarifying_questions', 'N/A')}")
            
            # Return clean clarification to UI
            clarification_response = f"""I need some clarification to give you the best answer:

{agent_result.get('clarifying_questions', '')}

Would you like to rephrase your question?"""
            
            audio_response = text_to_speech(clarification_response, progress_callback)
            return actual_question, clarification_response, audio_response
        else:
            processed_question = actual_question
            print(f"DEBUG - Using fallback processing for type: {agent_result.get('type')}")
        
        if progress_callback:
            progress_callback("Retrieving relevant information...")
            
        # RAG retrieval
        print("DEBUG - Starting RAG retrieval...")
        retrieval_result = enhanced_retrieve_context(processed_question)
        context = retrieval_result['context']
        sources = retrieval_result['sources']
        retrieval_info = retrieval_result['retrieval_info']
        
        print(f"DEBUG - Retrieved context length: {len(context)}")
        print(f"DEBUG - Number of sources: {len(sources)}")
        print(f"DEBUG - Retrieval info: {retrieval_info}")
        
        if sources:
            print("DEBUG - Source details:")
            for i, source in enumerate(sources):
                print(f"  Source {i+1}: similarity={source['similarity_score']:.3f}, tokens={source.get('token_count', 'N/A')}")
        
        if progress_callback:
            progress_callback("Generating intelligent response...")
            
        # Generate answer
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
        
        print("DEBUG - Generating answer with LLM...")
        result = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2
        )
        
        answer = result.choices[0].message.content
        print(f"DEBUG - Generated answer length: {len(answer)}")
        
        # Add source information if available (clean version for UI)
        if sources:
            source_count = len(sources)
            avg_similarity = np.mean([s['similarity_score'] for s in sources])
            source_info = f"\n\n📚 *Answer based on {source_count} relevant sections (avg. relevance: {avg_similarity:.1%})*"
            answer += source_info
        
        if progress_callback:
            progress_callback("Generating follow-up suggestions...")
            
        # Generate smart follow-up questions
        print("DEBUG - Generating follow-up questions...")
        followups = question_agent.get_suggested_followups(processed_question, answer)
        if followups:
            answer += f"\n\n**💭 You might also ask:**\n"
            for i, followup in enumerate(followups[:3], 1):
                answer += f"• {followup}\n"
        
        if progress_callback:
            progress_callback("Creating audio response...")
            
        # Generate audio
        print("DEBUG - Generating audio response...")
        audio_response = text_to_speech(answer, progress_callback)
        
        print("DEBUG - Q&A process completed successfully")
        print(f"=== END Q&A PROCESS ===\n")
        
        if progress_callback:
            progress_callback("Response completed!")
        
        # Return clean UI response
        return actual_question, answer, audio_response
        
    except Exception as e:
        print(f"DEBUG - ERROR in Q&A process: {str(e)}")
        import traceback
        traceback.print_exc()
        if progress_callback:
            progress_callback(f"Error: {str(e)}")
        return actual_question, f"I encountered an error while processing your question. Please try again.", None

# Backwards compatibility
answer_question = enhanced_answer_question
summarize_text = enhanced_summarize_text