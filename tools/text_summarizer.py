import os
from openai import OpenAI
from typing import Optional
from .tool_config import DEFAULT_SUMMARY_MODEL, MAX_SUMMARY_LENGTH

# Initialize OpenAI client
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

def summarize_text_factual(text: str, model: str = DEFAULT_SUMMARY_MODEL) -> str:
    """
    Original factual summarization function for backward compatibility
    """
    prompt = f"Summarize the following text in 5-7 key bullet points:\n\n{text[:8000]}\n\nBullet Summary:"
    
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=MAX_SUMMARY_LENGTH
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Summarization error: {str(e)}"

def summarize_with_depth(text: str, depth: str = "Balanced Summary", model: str = DEFAULT_SUMMARY_MODEL) -> str:
    """
    Enhanced summarization with configurable depth levels
    
    Args:
        text: Text to summarize
        depth: "Quick Overview", "Balanced Summary", or "Deep Analysis"
        model: OpenAI model to use
    
    Returns:
        Formatted summary
    """
    
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
    
    prompt = depth_prompts.get(depth, depth_prompts["Balanced Summary"])
    
    # Adjust temperature and max_tokens based on depth
    temperature_map = {
        "Quick Overview": 0.1,      # More focused
        "Balanced Summary": 0.3,    # Standard
        "Deep Analysis": 0.4        # More creative
    }
    
    max_tokens_map = {
        "Quick Overview": 500,
        "Balanced Summary": MAX_SUMMARY_LENGTH,
        "Deep Analysis": MAX_SUMMARY_LENGTH * 2
    }
    
    temperature = temperature_map.get(depth, 0.3)
    max_tokens = max_tokens_map.get(depth, MAX_SUMMARY_LENGTH)
    
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        return response.choices[0].message.content.strip()
    
    except Exception as e:
        return f"Summarization error: {str(e)}"

def summarize_with_streaming(text: str, depth: str = "Balanced Summary", model: str = DEFAULT_SUMMARY_MODEL):
    """
    Streaming version of summarization for real-time display
    
    Yields:
        Chunks of the summary as they're generated
    """
    depth_prompts = {
        "Quick Overview": f"Provide a brief, concise summary in 2-3 key bullet points:\n\n{text[:6000]}",
        "Balanced Summary": f"Provide a comprehensive summary with main topics, key points, and conclusions:\n\n{text[:8000]}",
        "Deep Analysis": f"Provide a detailed analysis with executive summary, themes, evidence, and implications:\n\n{text[:10000]}"
    }
    
    prompt = depth_prompts.get(depth, depth_prompts["Balanced Summary"])
    
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=MAX_SUMMARY_LENGTH,
            stream=True
        )
        
        for chunk in response:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
                
    except Exception as e:
        yield f"Streaming error: {str(e)}"

# Backward compatibility aliases
summarize_text = summarize_text_factual
enhanced_summarize_text = summarize_with_depth