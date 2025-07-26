
from openai import OpenAI

def summarize_text_factual(text: str) -> str:
    prompt = f"Summarize the following text in 5-7 key bullet points:\n\n{text}\n\nBullet Summary:"
    response = OpenAI().chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.5
    )
    return response.choices[0].message.content.strip()
