import streamlit as st
import openai
import PyPDF2
import numpy as np
from io import BytesIO
from sklearn.metrics.pairwise import cosine_similarity
import os
# 🔐 Set your OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")
client = openai.OpenAI(api_key=openai.api_key)

st.set_page_config(page_title="Resume Ranker (Embeddings)", layout="wide")
st.title("🧠 Resume Ranker using Semantic Embeddings")

# 📄 Upload JD
jd_file = st.file_uploader("📄 Upload Job Description (PDF or TXT)", type=["pdf", "txt"])
# 📁 Upload Resumes
resume_files = st.file_uploader("📁 Upload Resumes (PDFs)", type=["pdf"], accept_multiple_files=True)

# Extract text from PDF
def extract_text_from_pdf(file):
    reader = PyPDF2.PdfReader(file)
    return "\n".join(page.extract_text() or "" for page in reader.pages).strip()

# Chunk long text into ~500 token chunks
def chunk_text(text, max_words=250):
    words = text.split()
    return [" ".join(words[i:i+max_words]) for i in range(0, len(words), max_words)]

# Get OpenAI embedding for a single string
def get_embedding(text):
    response = client.embeddings.create(
        input=[text],
        model="text-embedding-3-small"
    )
    return np.array(response.data[0].embedding)

# Rank resumes by cosine similarity to JD
def rank_by_embedding(jd_text, resumes):
    jd_embedding = get_embedding(jd_text)

    ranked = []
    for resume_file in resumes:
        resume_text = extract_text_from_pdf(resume_file)
        chunks = chunk_text(resume_text)
        chunk_embeddings = [get_embedding(chunk) for chunk in chunks]

        # Compare all chunks to JD, take mean similarity
        sims = [cosine_similarity([jd_embedding], [chunk])[0][0] for chunk in chunk_embeddings]
        avg_score = round(np.mean(sims) * 100, 2)

        ranked.append({
            "filename": resume_file.name,
            "score": avg_score,
            "text": resume_text,
            "reason": f"Average embedding similarity with JD across {len(chunks)} chunks."
        })

    return sorted(ranked, key=lambda x: x["score"], reverse=True)

# JD extractor
def extract_jd_text(jd_file):
    if jd_file.type == "application/pdf":
        return extract_text_from_pdf(jd_file)
    else:
        return jd_file.read().decode("utf-8")

# 🚀 Run app
if jd_file and resume_files:
    with st.spinner("Generating embeddings and ranking resumes..."):
        jd_text = extract_jd_text(jd_file)
        ranked_resumes = rank_by_embedding(jd_text, resume_files)

    st.success(f"✅ Ranked {len(ranked_resumes)} resumes using embeddings!")

    st.subheader("🏆 Resume Leaderboard (Semantic Similarity)")
    for i, res in enumerate(ranked_resumes):
        with st.expander(f"{i+1}. {res['filename']} — Score: {res['score']}"):
            st.markdown(f"**Reason:** {res['reason']}")
            with st.expander("📄 Preview Resume Text"):
                st.text(res['text'][:3000])  # limit preview to 3000 chars
