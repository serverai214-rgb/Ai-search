import streamlit as st
import requests
import pandas as pd

API_URL = "http://localhost:8000"

st.set_page_config(page_title="Resume Semantic Search", page_icon="🔍")
st.title("🔍 Resume Semantic Search")

tab1, tab2, tab3 = st.tabs([
    "👤 Candidate — Upload Resume",
    "🔎 Recruiter — Find Candidates",
    "📋 Admin — Manage"
])

# ── TAB 1: CANDIDATE ──────────────────────────────────────
with tab1:
    st.subheader("Upload Your Resume")
    st.write("Upload your resume and recruiters will be able to find you.")

    resume_file = st.file_uploader("Upload Resume (PDF or TXT)", type=["pdf", "txt"], key="candidate_uploader")

    if st.button("Submit Resume"):
        if not resume_file:
            st.warning("Please upload your resume first.")
        else:
            with st.spinner("Submitting..."):
                try:
                    files = {"file": (resume_file.name, resume_file.read(), resume_file.type)}
                    res = requests.post(f"{API_URL}/submit-resume", files=files, timeout=30)
                    res.raise_for_status()
                    st.success("✅ Resume submitted! Recruiters can now find you.")
                except requests.exceptions.ConnectionError:
                    st.error("Cannot connect to server. Make sure the API is running.")
                except Exception as e:
                    st.error(f"Error: {e}")

# ── TAB 2: RECRUITER ──────────────────────────────────────
with tab2:
    st.subheader("Find Top Candidates")
    st.write("Type a job description to find the most relevant candidates.")

    query = st.text_area(
        "Job Description",
        height=200,
        placeholder="e.g. React developer with 3 years experience, Redux, REST APIs, TypeScript..."
    )
    top_k = st.slider("Number of candidates", min_value=1, max_value=50, value=10)

    if st.button("Find Top Candidates"):
        if not query.strip():
            st.warning("Please enter a job description.")
        else:
            with st.spinner("Searching candidates..."):
                try:
                    res = requests.post(
                        f"{API_URL}/search",
                        data={"query": query, "top_k": top_k},
                        timeout=30
                    )
                    res.raise_for_status()
                    results = res.json()["results"]

                    if not results:
                        st.warning("No candidates found. Ask candidates to upload their resumes first.")
                    else:
                        st.success(f"Top {len(results)} candidates found")

                        for i, r in enumerate(results, 1):
                            with st.expander(f"#{i} — {r['filename']} | Score: {r['score']}"):
                                st.write(r["text"])

                        st.divider()
                        df = pd.DataFrame([{
                            "Rank": i + 1,
                            "Candidate": r["filename"],
                            "Match Score": r["score"],
                            "Resume": r["text"][:500]
                        } for i, r in enumerate(results)])

                        csv = df.to_csv(index=False).encode("utf-8")
                        st.download_button(
                            label="📥 Download as CSV",
                            data=csv,
                            file_name="top_candidates.csv",
                            mime="text/csv"
                        )

                except requests.exceptions.ConnectionError:
                    st.error("Cannot connect to server. Make sure the API is running.")
                except requests.exceptions.Timeout:
                    st.error("Search timed out. Try again.")
                except Exception as e:
                    st.error(f"Error: {e}")

# ── TAB 3: ADMIN ──────────────────────────────────────────
with tab3:
    st.subheader("Manage Candidate Pool")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔄 Refresh"):
            st.rerun()
    with col2:
        if st.button("🗑️ Clear All Candidates", type="primary"):
            requests.delete(f"{API_URL}/resumes")
            st.success("All candidates cleared")
            st.rerun()

    st.divider()

    try:
        res = requests.get(f"{API_URL}/resumes", timeout=10)
        data = res.json()
        st.write(f"**Total candidates in pool: {data['total']}**")

        resumes = data["resumes"]
        if not resumes:
            st.info("No candidates yet. Ask users to upload their resumes.")
        else:
            for r in resumes:
                col1, col2 = st.columns([4, 1])
                with col1:
                    st.write(f"📄 {r['filename']}")
                with col2:
                    if st.button("Delete", key=r["filename"]):
                        requests.delete(f"{API_URL}/resumes/{r['filename']}")
                        st.rerun()

    except requests.exceptions.ConnectionError:
        st.error("Cannot connect to server. Make sure the API is running.")
    except Exception as e:
        st.error(f"Error: {e}")