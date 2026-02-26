import streamlit as st
import numpy as np
import os
import json
import io
import re

# ── PDF extraction ──────────────────────────────────────────
def extract_text_from_pdf(file_bytes: bytes) -> str:
    try:
        from PyPDF2 import PdfReader
        reader = PdfReader(io.BytesIO(file_bytes))
        text = ""
        for page in reader.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted + "\n"
        return text.strip()
    except Exception as e:
        return ""

def extract_text_from_txt(file_bytes: bytes) -> str:
    return file_bytes.decode("utf-8", errors="ignore").strip()

# ── Text helpers ────────────────────────────────────────────
def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def truncate_text(text: str, max_words: int = 512) -> str:
    words = text.split()
    return " ".join(words[:max_words]) if len(words) > max_words else text

def preprocess(text: str) -> str:
    return truncate_text(clean_text(text))

# ── Embedder (cached so model loads only once) ──────────────
@st.cache_resource(show_spinner="Loading AI model...")
def get_model():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer("all-MiniLM-L6-v2")

def embed_text(text: str) -> np.ndarray:
    model = get_model()
    return model.encode([text])[0].astype(np.float32)

# ── In-memory vector store (persists during session) ────────
DIMENSION = 384
MIN_SCORE = 0.4

def _init_store():
    if "faiss_index" not in st.session_state:
        import faiss
        st.session_state.faiss_index = faiss.IndexFlatL2(DIMENSION)
        st.session_state.resume_meta = []

def add_resume(filename: str, text: str, embedding: np.ndarray):
    import faiss
    _init_store()
    vec = np.array([embedding], dtype=np.float32)
    st.session_state.faiss_index.add(vec)
    st.session_state.resume_meta.append({
        "id": st.session_state.faiss_index.ntotal - 1,
        "filename": filename,
        "text": text[:1000]
    })

def search_resumes(query_embedding: np.ndarray, top_k: int = 10):
    _init_store()
    index = st.session_state.faiss_index
    meta = st.session_state.resume_meta

    if index.ntotal == 0:
        return []

    vec = np.array([query_embedding], dtype=np.float32)
    search_k = min(top_k * 5, index.ntotal)
    distances, indices = index.search(vec, search_k)

    results = []
    for dist, idx in zip(distances[0], indices[0]):
        if idx < len(meta):
            score = round(float(1 / (1 + dist)), 4)
            if score >= MIN_SCORE:
                entry = meta[idx].copy()
                entry["score"] = score
                results.append(entry)
            if len(results) >= top_k:
                break

    results.sort(key=lambda x: x["score"], reverse=True)
    return results

def get_all_resumes():
    _init_store()
    return st.session_state.resume_meta

def delete_resume(filename: str):
    import faiss
    _init_store()
    meta = st.session_state.resume_meta
    new_meta = [m for m in meta if m["filename"] != filename]
    if len(new_meta) == len(meta):
        return False

    new_index = faiss.IndexFlatL2(DIMENSION)
    updated_meta = []
    for i, entry in enumerate(new_meta):
        vec = np.array([embed_text(entry["text"])], dtype=np.float32)
        new_index.add(vec)
        entry["id"] = i
        updated_meta.append(entry)

    st.session_state.faiss_index = new_index
    st.session_state.resume_meta = updated_meta
    return True

def clear_all():
    import faiss
    st.session_state.faiss_index = faiss.IndexFlatL2(DIMENSION)
    st.session_state.resume_meta = []

# ── UI ──────────────────────────────────────────────────────
st.set_page_config(page_title="Resume Semantic Search", page_icon="🔍")
st.title("🔍 Resume Semantic Search")

_init_store()

tab1, tab2, tab3 = st.tabs([
    "👤 Candidate — Upload Resume",
    "🔎 Recruiter — Find Candidates",
    "📋 Admin — Manage"
])

# ── TAB 1: CANDIDATE ────────────────────────────────────────
with tab1:
    st.subheader("Upload Your Resume")
    st.write("Upload your resume and recruiters will be able to find you.")

    resume_file = st.file_uploader("Upload Resume (PDF or TXT)", type=["pdf", "txt"], key="candidate_uploader")

    if st.button("Submit Resume"):
        if not resume_file:
            st.warning("Please upload your resume first.")
        else:
            # Check for duplicate
            existing = [r["filename"] for r in get_all_resumes()]
            if resume_file.name in existing:
                st.warning(f"⚠️ '{resume_file.name}' is already in the system.")
            else:
                with st.spinner("Processing resume..."):
                    content = resume_file.read()
                    if resume_file.name.endswith(".pdf"):
                        text = extract_text_from_pdf(content)
                    else:
                        text = extract_text_from_txt(content)

                    if not text.strip():
                        st.error("Could not extract text from resume. Please try a different file.")
                    else:
                        clean = preprocess(text)
                        embedding = embed_text(clean)
                        add_resume(resume_file.name, text, embedding)
                        st.success(f"✅ Resume '{resume_file.name}' submitted! Recruiters can now find you.")

# ── TAB 2: RECRUITER ────────────────────────────────────────
with tab2:
    st.subheader("Find Top Candidates")
    st.write("Enter a job description to find the most similar candidates.")

    query = st.text_area(
        "Job Description",
        height=200,
        placeholder="e.g. React developer with 3 years experience, REST APIs, TypeScript..."
    )
    top_k = st.slider("Number of candidates", min_value=1, max_value=50, value=10)

    if st.button("Find Top Candidates"):
        if not query.strip():
            st.warning("Please enter a job description.")
        elif st.session_state.faiss_index.ntotal == 0:
            st.warning("No resumes in the system yet. Ask candidates to upload their resumes first.")
        else:
            with st.spinner("Searching candidates..."):
                import pandas as pd
                clean_query = preprocess(query)
                query_embedding = embed_text(clean_query)
                results = search_resumes(query_embedding, top_k=top_k)

                if not results:
                    st.warning("❌ No relevant candidates found. Try adjusting your search or upload more resumes.")
                else:
                    st.success(f"✅ Found {len(results)} relevant candidate(s)")

                    for i, r in enumerate(results, 1):
                        score_pct = int(r["score"] * 100)
                        with st.expander(f"#{i} — {r['filename']} | Match: {score_pct}%"):
                            st.write(r["text"])

                    st.divider()
                    df = pd.DataFrame([{
                        "Rank": i + 1,
                        "Candidate": r["filename"],
                        "Match Score": f"{int(r['score'] * 100)}%",
                        "Resume Preview": r["text"][:300] + "..."
                    } for i, r in enumerate(results)])

                    st.dataframe(df, use_container_width=True)

                    csv = df.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        label="📥 Download as CSV",
                        data=csv,
                        file_name="top_candidates.csv",
                        mime="text/csv"
                    )

# ── TAB 3: ADMIN ────────────────────────────────────────────
with tab3:
    st.subheader("Manage Candidate Pool")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔄 Refresh"):
            st.rerun()
    with col2:
        if st.button("🗑️ Clear All Candidates", type="primary"):
            clear_all()
            st.success("All candidates cleared.")
            st.rerun()

    st.divider()

    resumes = get_all_resumes()
    st.write(f"**Total candidates in pool: {len(resumes)}**")

    if not resumes:
        st.info("No candidates yet. Ask users to upload their resumes.")
    else:
        for r in resumes:
            col1, col2 = st.columns([4, 1])
            with col1:
                st.write(f"📄 {r['filename']}")
            with col2:
                if st.button("Delete", key=f"del_{r['filename']}"):
                    with st.spinner("Deleting..."):
                        delete_resume(r["filename"])
                    st.rerun()
