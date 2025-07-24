import streamlit as st
import openai
import PyPDF2
import numpy as np
from io import BytesIO
from sklearn.metrics.pairwise import cosine_similarity

# 🔑 Set your OpenAI API key directly here
openai.api_key = "sk-proj-ydk9KGZKXxVZZ4UCzbFQT3BlbkFJv1mW14omh2UkGXlEPBsW
"

st.set_page_config(page_title="Resume Ranker", layout="wide")
st.title("🧠 Resume Ranker using Embeddings")

# 📄 Upload Job Description (JD)
jd_file = st.file_uploader("📄 Upload Job Description (PDF or TXT)", type=["pdf", "txt"])

# 📁 Upload Resumes
resume_files = st.file_uploader("📁 Upload Resumes (PDFs)", type=["pdf"], accept_multiple_files=True)

# 📜 Extract text from PDF
def extract_text_from_pdf(file):
    reader = PyPDF2.PdfReader(file)
    return "\n".join(page.extract_text() or "" for page in reader.pages).strip()

# 🔄 Chunk text into ~250 words
def chunk_text(text, max_words=250):
    words = text.split()
    return [" ".join(words[i:i+max_words]) for i in range(0, len(words), max_words)]

# 🧠 Get OpenAI embedding
def get_embedding(text):
    try:
        text = text[:8000]  # Truncate for safety
        response = openai.Embedding.create(
            input=[text],
            model="text-embedding-3-small"
        )
        return np.array(response['data'][0]['embedding'])
    except Exception as e:
        st.error(f"❌ Error generating embedding: {e}")
        return np.zeros((1536,))

# 🏆 Rank resumes by similarity
def rank_resumes(jd_text, resumes):
    jd_embedding = get_embedding(jd_text)
    ranked = []

    for resume_file in resumes:
        resume_text = extract_text_from_pdf(resume_file)
        chunks = chunk_text(resume_text)
        embeddings = [get_embedding(chunk) for chunk in chunks]

        similarities = [cosine_similarity([jd_embedding], [emb])[0][0] for emb in embeddings]
        avg_score = round(np.mean(similarities) * 100, 2)

        ranked.append({
            "filename": resume_file.name,
            "score": avg_score,
            "text": resume_text,
            "reason": f"Average similarity with JD across {len(chunks)} chunks"
        })

    return sorted(ranked, key=lambda x: x['score'], reverse=True)

# 🔍 Extract JD text
def extract_jd_text(jd_file):
    if jd_file.type == "application/pdf":
        return extract_text_from_pdf(jd_file)
    else:
        return jd_file.read().decode("utf-8")

# 🚀 Run App
if jd_file and resume_files:
    with st.spinner("Ranking resumes..."):
        jd_text = extract_jd_text(jd_file)
        results = rank_resumes(jd_text, resume_files)

    st.success(f"✅ Ranked {len(results)} resumes!")

    st.subheader("📊 Resume Leaderboard")
    for i, res in enumerate(results):
        with st.expander(f"{i+1}. {res['filename']} — Score: {res['score']}"):
            st.markdown(f"**Why?** {res['reason']}")
            with st.expander("📄 Preview Resume Text"):
                st.text(res['text'][:3000])
