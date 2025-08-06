import re
import tiktoken
from typing import List, Dict
from .tool_config import DEFAULT_CHUNK_SIZE, CHUNK_OVERLAP

def chunk_text(text: str, max_len: int = DEFAULT_CHUNK_SIZE) -> List[str]:
    """
    Simple text chunking function for backward compatibility
    """
    import textwrap
    return textwrap.wrap(text, max_len)

def smart_chunk_text(text: str, chunk_size: int = DEFAULT_CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[Dict]:
    """
    Smart text chunking with sentence boundaries and metadata
    
    Args:
        text: Text to chunk
        chunk_size: Target size in tokens
        overlap: Overlap size in tokens
        
    Returns:
        List of chunk dictionaries with content and metadata
    """
    # Clean text first
    text = clean_text(text)
    
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
                'start_sentence': max(0, len(chunks) * 10 - overlap//10),  # Approximate
                'end_sentence': i,
                'token_count': current_length,
                'chunk_id': len(chunks)
            })
            
            # Start new chunk with overlap
            overlap_text = get_overlap_text(current_chunk, overlap, tokenizer)
            current_chunk = overlap_text + " " + sentence if overlap_text else sentence
            current_length = len(tokenizer.encode(current_chunk))
        else:
            current_chunk += " " + sentence if current_chunk else sentence
            current_length += sentence_tokens
    
    # Add the last chunk
    if current_chunk:
        chunks.append({
            'content': current_chunk.strip(),
            'start_sentence': max(0, len(chunks) * 10 - overlap//10),
            'end_sentence': len(sentences),
            'token_count': current_length,
            'chunk_id': len(chunks)
        })
    
    return chunks

def get_overlap_text(text: str, overlap_tokens: int, tokenizer) -> str:
    """Get the last N tokens from text for overlap"""
    if overlap_tokens <= 0:
        return ""
        
    tokens = tokenizer.encode(text)
    if len(tokens) <= overlap_tokens:
        return text
    
    overlap_tokens_list = tokens[-overlap_tokens:]
    return tokenizer.decode(overlap_tokens_list)

def clean_text(text: str) -> str:
    """Clean and normalize text for better processing"""
    if not text:
        return ""
    
    # Remove excessive whitespace
    text = re.sub(r'\n+', '\n', text)
    text = re.sub(r' +', ' ', text)
    
    # Remove special characters but keep punctuation
    text = re.sub(r'[^\w\s\.\,\!\?\;\:\-\(\)]', '', text)
    
    # Normalize line breaks
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    
    return text.strip()

def chunk_by_paragraphs(text: str, max_chunk_size: int = DEFAULT_CHUNK_SIZE) -> List[Dict]:
    """
    Alternative chunking method that preserves paragraph boundaries
    """
    paragraphs = text.split('\n\n')
    chunks = []
    current_chunk = ""
    tokenizer = tiktoken.encoding_for_model("gpt-4")
    
    for i, para in enumerate(paragraphs):
        para = para.strip()
        if not para:
            continue
            
        para_tokens = len(tokenizer.encode(para))
        current_tokens = len(tokenizer.encode(current_chunk))
        
        if current_tokens + para_tokens > max_chunk_size and current_chunk:
            # Save current chunk
            chunks.append({
                'content': current_chunk.strip(),
                'chunk_type': 'paragraph_based',
                'paragraph_start': max(0, i - len(current_chunk.split('\n\n'))),
                'paragraph_end': i - 1,
                'token_count': current_tokens,
                'chunk_id': len(chunks)
            })
            current_chunk = para
        else:
            current_chunk += "\n\n" + para if current_chunk else para
    
    # Add final chunk
    if current_chunk:
        chunks.append({
            'content': current_chunk.strip(),
            'chunk_type': 'paragraph_based',
            'paragraph_start': len(paragraphs) - len(current_chunk.split('\n\n')),
            'paragraph_end': len(paragraphs) - 1,
            'token_count': len(tokenizer.encode(current_chunk)),
            'chunk_id': len(chunks)
        })
    
    return chunks

def chunk_with_headers(text: str, max_chunk_size: int = DEFAULT_CHUNK_SIZE) -> List[Dict]:
    """
    Advanced chunking that preserves document structure with headers
    """
    # Detect headers (lines that are short and followed by content)
    lines = text.split('\n')
    chunks = []
    current_chunk = ""
    current_header = ""
    tokenizer = tiktoken.encoding_for_model("gpt-4")
    
    for i, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue
            
        # Simple header detection (short line followed by longer content)
        is_header = (len(line) < 80 and 
                    i < len(lines) - 1 and 
                    len(lines[i + 1].strip()) > len(line) and
                    not line.endswith('.'))
        
        if is_header:
            # Save previous chunk if it exists
            if current_chunk:
                chunks.append({
                    'content': current_chunk.strip(),
                    'header': current_header,
                    'chunk_type': 'header_based',
                    'token_count': len(tokenizer.encode(current_chunk)),
                    'chunk_id': len(chunks)
                })
            
            current_header = line
            current_chunk = line + "\n"
        else:
            current_chunk += line + "\n"
            
            # Check if chunk is getting too large
            if len(tokenizer.encode(current_chunk)) > max_chunk_size:
                chunks.append({
                    'content': current_chunk.strip(),
                    'header': current_header,
                    'chunk_type': 'header_based',
                    'token_count': len(tokenizer.encode(current_chunk)),
                    'chunk_id': len(chunks)
                })
                current_chunk = ""
    
    # Add final chunk
    if current_chunk:
        chunks.append({
            'content': current_chunk.strip(),
            'header': current_header,
            'chunk_type': 'header_based',
            'token_count': len(tokenizer.encode(current_chunk)),
            'chunk_id': len(chunks)
        })
    
    return chunks

# Backward compatibility
def chunk_text_simple(text: str, max_len: int = 512) -> List[str]:
    """Simple chunking for backward compatibility"""
    import textwrap
    return textwrap.wrap(text, max_len)