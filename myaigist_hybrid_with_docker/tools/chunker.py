
def chunk_text(text: str, max_len: int = 512) -> list:
    import textwrap
    return textwrap.wrap(text, max_len)
