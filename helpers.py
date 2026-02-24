import re

def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def truncate_text(text: str, max_words: int = 512) -> str:
    words = text.split()
    if len(words) > max_words:
        return " ".join(words[:max_words])
    return text

def preprocess(text: str, max_words: int = 512) -> str:
    return truncate_text(clean_text(text), max_words)