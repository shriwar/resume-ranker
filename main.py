import streamlit as st
from openai import OpenAI
import PyPDF2
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv
import os
import tiktoken
import docx
from pdf2image import convert_from_bytes
import pytesseract
from PIL import Image

# Load environment variables
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

st.set_page_config(page_title="🧠 Resume Ranker", layout="wide")
st.title("🧠 Resume Ranker using OpenAI Embeddings with Chunking + OCR")

# 🎓 Priority Colleges and Companies
top_colleges = ["iit", "nit", "nift", "iim", "iiit", "bhu", "bits", "nifd", "nid", "srishti"]
top_companies = ["google", "microsoft", "amazon", "apple", "meta", "netflix", "tcs", "infosys", "wipro", "adobe", "zoho"]

# 🔍 Chunking for Long Texts
def chunk_text(text, max_tokens=800):
    enc = tiktoken.encoding_for_model("text-embedding-3-small")
    tokens = enc.encode(text)
    chunks = []

    for i in range(0, len(tokens), max_tokens):
        chunk = tokens[i:i + max_tokens]
        decoded = enc.decode(chunk)
        chunks.append(decoded)

    return chunks

# 📊 Get Embedding with Chunking
@st.cache_data(show_spinner=False)
def get_embedding(text):
    chunks = chunk_text(text)
    embeddings = []

    for chunk in chunks:
        try:
            response = client.embeddings.create(
                input=[chunk],
                model="text-embedding-3-small"
            )
            embedding = response.data[0].embedding
            embeddings.append(embedding)
        except Exception as e:
            st.error(f"❌ Embedding failed on a chunk: {e}")
            continue

    if not embeddings:
        return np.zeros(1536)

    return np.mean(np.array(embeddings), axis=0)

# 📄 Extract text from PDF with OCR fallback
def extract_text_from_pdf(file):
    try:
        reader = PyPDF2.PdfReader(file)
        text = "\n".join([page.extract_text() or "" for page in reader.pages])
        file.seek(0)

        if text.strip():
            return text

        st.info(f"🖼️ Using OCR for image-based PDF: `{file.name}`")
        images = convert_from_bytes(file.read(), dpi=300)
        ocr_text = ""
        for img in images:
            ocr_text += pytesseract.image_to_string(img)
        file.seek(0)
        return ocr_text
    except Exception as e:
        st.error(f"❌ PDF text extraction failed: {e}")
        return ""

# 📄 Extract text from DOCX
def extract_text_from_docx(file):
    try:
        doc = docx.Document(file)
        return "\n".join([para.text for para in doc.paragraphs])
    except Exception as e:
        st.error(f"❌ Failed to read DOCX: {e}")
        return ""

# 🎯 Scoring Boost (bonus logic)
def compute_bonus_score(text, jd_text):
    text_lower = text.lower()
    jd_lower = jd_text.lower()

    bonus = 0

    if any(college in text_lower for college in top_colleges):
        bonus += 0.05
    if any(company in text_lower for company in top_companies):
        bonus += 0.05
    if any(keyword in text_lower for keyword in jd_lower.split()):
        bonus += 0.05

    return min(bonus, 0.15)

# 🧠 Rank resumes
def rank_by_embedding(jd_text, resume_files):
    jd_embedding = get_embedding(jd_text)
    scores = []
    resume_text_cache = {}

    for resume_file in resume_files:
        ext = resume_file.name.lower().split('.')[-1]
        if ext == "pdf":
            resume_text = extract_text_from_pdf(resume_file)
        elif ext == "docx":
            resume_text = extract_text_from_docx(resume_file)
        else:
            st.warning(f"⚠️ Unsupported file format: {resume_file.name}")
            continue

        if not resume_text.strip():
            st.warning(f"⚠️ No text extracted from `{resume_file.name}`.")
            continue

        resume_text_cache[resume_file.name] = resume_text
        resume_embedding = get_embedding(resume_text)
        similarity = cosine_similarity([jd_embedding], [resume_embedding])[0][0]

        bonus = compute_bonus_score(resume_text, jd_text)
        final_score = similarity + bonus
        scores.append((resume_file.name, final_score))

    ranked = sorted(scores, key=lambda x: x[1], reverse=True)
    return ranked, resume_text_cache

# ---------------- UI ---------------- #

# 📥 JD Input
jd_input_method = st.radio("📌 Choose Job Description Input Method:", ["Paste Text", "Upload File (PDF/DOCX)"])

if jd_input_method == "Paste Text":
    jd_text = st.text_area("📄 Paste the Job Description here:", height=200, key="jd_text_input")
else:
    jd_file = st.file_uploader("📁 Upload JD File", type=["pdf", "docx"])
    jd_text = ""
    if jd_file:
        ext = jd_file.name.lower().split('.')[-1]
        if ext == "pdf":
            jd_text = extract_text_from_pdf(jd_file)
        elif ext == "docx":
            jd_text = extract_text_from_docx(jd_file)
        else:
            st.warning("⚠️ Only PDF and DOCX supported for JD upload.")

# 📁 Resume Upload
resume_files = st.file_uploader("📁 Upload Resumes (PDF or DOCX)", type=["pdf", "docx"], accept_multiple_files=True)

# 🚀 Trigger Ranking
if st.button("🚀 Rank Resumes"):
    if not jd_text.strip() or not resume_files:
        st.warning("⚠️ Provide both Job Description and at least one Resume.")
    else:
        with st.spinner("🔍 Analyzing resumes using embeddings..."):
            ranked_resumes, resume_text_cache = rank_by_embedding(jd_text, resume_files)

        st.success("✅ Ranking complete!")
        st.subheader("🏆 Ranked Resumes")

        for i, (name, score) in enumerate(ranked_resumes, start=1):
            st.markdown(f"**{i}. {name}** — Similarity Score: `{score:.4f}`")
            text = resume_text_cache.get(name)
            if text:
                with st.expander(f"📄 Preview Resume: {name}"):
                    st.text_area("📄 Resume Preview", text, height=300, key=f"resume_preview_{i}_{name}")

