from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import List
from embedder import embed_text
from pdf_extractor import extract_text_from_pdf, extract_text_from_txt
from helpers import preprocess
from vector_store import add_resume, search_resumes, get_all_resumes, delete_resume, clear_all

app = FastAPI(title="Resume Semantic Search API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def home():
    return {"message": "Resume Semantic Search API running"}

@app.post("/submit-resume")
async def submit_resume(file: UploadFile = File(...)):
    """Candidate submits resume — stored in FAISS"""
    content = await file.read()
    if file.filename.endswith(".pdf"):
        text = extract_text_from_pdf(content)
    else:
        text = extract_text_from_txt(content)

    if not text.strip():
        raise HTTPException(status_code=400, detail="Could not extract text from resume")

    clean = preprocess(text)
    embedding = embed_text(clean)
    add_resume(file.filename, text, embedding)

    return {"message": "Resume submitted successfully", "filename": file.filename}

@app.post("/search")
def search(query: str = Form(...), top_k: int = Form(10)):
    """Recruiter searches by job description"""
    clean_query = preprocess(query)
    query_embedding = embed_text(clean_query)
    results = search_resumes(query_embedding, top_k=top_k)
    return {"query": query, "results": results}

@app.get("/resumes")
def list_resumes():
    """List all candidates in pool"""
    resumes = get_all_resumes()
    return {"total": len(resumes), "resumes": resumes}

@app.delete("/resumes/{filename}")
def remove_resume(filename: str):
    """Delete a specific resume"""
    success = delete_resume(filename)
    if not success:
        raise HTTPException(status_code=404, detail=f"{filename} not found")
    return {"deleted": filename}

@app.delete("/resumes")
def clear_index():
    """Clear all resumes"""
    clear_all()
    return {"message": "All resumes cleared"}