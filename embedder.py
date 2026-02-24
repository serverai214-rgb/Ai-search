from sentence_transformers import SentenceTransformer
import numpy as np

_model = SentenceTransformer("all-MiniLM-L6-v2")

def embed_text(text: str) -> np.ndarray:
    return _model.encode([text])[0].astype(np.float32)

def embed_texts(texts: list) -> np.ndarray:
    return _model.encode(texts).astype(np.float32)