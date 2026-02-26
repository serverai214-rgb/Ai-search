import faiss
import numpy as np
import json
import os

INDEX_FILE = "resume_index.faiss"
META_FILE = "resume_meta.json"
DIMENSION = 384
MIN_SCORE = 0.4  # Only return resumes with 40%+ relevance

_index = None
_meta = []

def _get_index():
    global _index
    if _index is None:
        if os.path.exists(INDEX_FILE):
            _index = faiss.read_index(INDEX_FILE)
        else:
            _index = faiss.IndexFlatL2(DIMENSION)
    return _index

def _get_meta():
    global _meta
    if not _meta and os.path.exists(META_FILE):
        with open(META_FILE, "r") as f:
            _meta = json.load(f)
    return _meta

def _save_index():
    faiss.write_index(_get_index(), INDEX_FILE)

def _save_meta():
    with open(META_FILE, "w") as f:
        json.dump(_get_meta(), f, indent=2)

def add_resume(filename: str, text: str, embedding: np.ndarray):
    """Add resume to vector store"""
    index = _get_index()
    meta = _get_meta()
    vec = np.array([embedding], dtype=np.float32)
    index.add(vec)
    meta.append({
        "id": index.ntotal - 1,
        "filename": filename,
        "text": text[:1000]
    })
    _save_index()
    _save_meta()

def search_resumes(query_embedding: np.ndarray, top_k: int = 10, min_score: float = MIN_SCORE):
    """
    Search resumes using semantic similarity.
    Only returns results with score >= min_score.
    
    Score guide:
    - 0.5+ = Highly relevant
    - 0.4-0.5 = Relevant  
    - 0.3-0.4 = Somewhat relevant
    - <0.4 = Not shown (filtered out)
    """
    index = _get_index()
    meta = _get_meta()

    if index.ntotal == 0:
        return []

    vec = np.array([query_embedding], dtype=np.float32)
    # Search more to filter properly
    search_k = min(top_k * 5, index.ntotal)
    distances, indices = index.search(vec, search_k)

    results = []
    for dist, idx in zip(distances[0], indices[0]):
        if idx < len(meta):
            score = round(float(1 / (1 + dist)), 4)
            
            # Only include if meets minimum threshold
            if score >= min_score:
                entry = meta[idx].copy()
                entry["score"] = score
                results.append(entry)
            
            # Stop once we have enough
            if len(results) >= top_k:
                break

    results.sort(key=lambda x: x["score"], reverse=True)
    return results

def get_all_resumes():
    return _get_meta()

def delete_resume(filename: str):
    global _index, _meta
    meta = _get_meta()
    new_meta = [m for m in meta if m["filename"] != filename]

    if len(new_meta) == len(meta):
        return False

    from embedder import embed_text
    _index = faiss.IndexFlatL2(DIMENSION)
    _meta = []

    for i, entry in enumerate(new_meta):
        vec = np.array([embed_text(entry["text"])], dtype=np.float32)
        _index.add(vec)
        entry["id"] = i
        _meta.append(entry)

    _save_index()
    _save_meta()
    return True

def clear_all():
    global _index, _meta
    _index = faiss.IndexFlatL2(DIMENSION)
    _meta = []
    if os.path.exists(INDEX_FILE):
        os.remove(INDEX_FILE)
    if os.path.exists(META_FILE):
        os.remove(META_FILE)
