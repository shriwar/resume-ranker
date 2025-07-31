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
import smtplib
from email.message import EmailMessage
import re

# Load environment variables
load_dotenv()

# Configuration
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
EMAIL_SENDER = os.getenv("EMAIL_ADDRESS")
EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD")

# Streamlit page configuration
st.set_page_config(page_title="🧠 Resume Ranker", layout="wide")
st.title("🧠 Resume Ranker + AI Screening Test Generator")

# Initialize session state
session_defaults = {
    "ranked_resumes": [],
    "resume_texts": {},
    "selected_candidates": [],
    "screening_test": "",
    "emails": {},
    "rank_triggered": False
}
for key, default in session_defaults.items():
    if key not in st.session_state:
        st.session_state[key] = default

# Constants
TOP_COLLEGES = ["iit", "nit", "nift", "iim", "iiit", "bhu", "bits", "nifd", "nid", "srishti"]
TOP_COMPANIES = ["google", "microsoft", "amazon", "apple", "meta", "netflix", "tcs", "infosys", "wipro", "adobe", "zoho"]

# Utility functions
def chunk_text(text, max_tokens=800):
    enc = tiktoken.encoding_for_model("text-embedding-3-small")
    tokens = enc.encode(text)
    return [enc.decode(tokens[i:i + max_tokens]) for i in range(0, len(tokens), max_tokens)]

@st.cache_data(show_spinner=False)
def get_embedding(text):
    chunks = chunk_text(text)
    embeddings = []
    for chunk in chunks:
        try:
            response = client.embeddings.create(input=[chunk], model="text-embedding-3-small")
            embeddings.append(response.data[0].embedding)
        except:
            continue
    return np.mean(np.array(embeddings), axis=0) if embeddings else np.zeros(1536)

def extract_text_from_pdf(file):
    try:
        reader = PyPDF2.PdfReader(file)
        text = "\n".join([page.extract_text() or "" for page in reader.pages])
        file.seek(0)
        if text.strip():
            return text
        images = convert_from_bytes(file.read(), dpi=300)
        return "".join([pytesseract.image_to_string(img) for img in images])
    except:
        return ""

def extract_text_from_docx(file):
    try:
        doc = docx.Document(file)
        return "\n".join([para.text for para in doc.paragraphs])
    except:
        return ""

def compute_bonus_score(text, jd_text):
    text_lower, jd_lower = text.lower(), jd_text.lower()
    bonus = 0
    if any(college in text_lower for college in TOP_COLLEGES): bonus += 0.05
    if any(company in text_lower for company in TOP_COMPANIES): bonus += 0.05
    if any(keyword in text_lower for keyword in jd_lower.split()): bonus += 0.05
    return min(bonus, 0.15)

def extract_email(text):
    match = re.search(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}", text)
    return match.group(0) if match else ""

def rank_by_embedding(jd_text, resume_files, custom_text=""):
    jd_embedding = get_embedding(jd_text)
    custom_embedding = get_embedding(custom_text) if custom_text.strip() else None
    scores = []
    resume_texts = {}

    for resume_file in resume_files:
        name = resume_file.name
        ext = name.lower().split('.')[-1]
        resume_text = extract_text_from_pdf(resume_file) if ext == "pdf" else extract_text_from_docx(resume_file)
        if not resume_text.strip():
            continue

        resume_texts[name] = resume_text
        emb = get_embedding(resume_text)

        jd_similarity = cosine_similarity([jd_embedding], [emb])[0][0]
        custom_similarity = cosine_similarity([custom_embedding], [emb])[0][0] if custom_embedding is not None else 0.0

        combined_similarity = 0.8 * jd_similarity + 0.2 * custom_similarity
        final_score = combined_similarity + compute_bonus_score(resume_text, jd_text)

        scores.append((name, final_score))

    return sorted(scores, key=lambda x: x[1], reverse=True), resume_texts

# def generate_screening_test(jd_text):
#     prompt = f"""Generate 5 short screening questions for this job description:\n\n{jd_text}\n\nReturn only the questions."""
#     try:
#         res = client.chat.completions.create(
#             model="gpt-4",
#             messages=[{"role": "user", "content": prompt}],
#             temperature=0.7
#         )
#         return res.choices[0].message.content.strip()
#     except:
#         return ""

# def send_email(to_email, subject, body):
#     try:
#         msg = EmailMessage()
#         msg.set_content(body)
#         msg["Subject"] = subject
#         msg["From"] = EMAIL_SENDER
#         msg["To"] = to_email
#         with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:
#             smtp.login(EMAIL_SENDER, EMAIL_PASSWORD)
#             smtp.send_message(msg)
#         return True
#     except Exception as e:
#         st.error(f"Email failed: {e}")
#         return False

# ---------- User Interface ----------

# JD Input
jd_input_method = st.radio("📌 Job Description Input Method", ["Paste Text", "Upload File (PDF/DOCX)"])
jd_text = ""

if jd_input_method == "Paste Text":
    jd_text = st.text_area("Paste JD here", height=200)
else:
    jd_file = st.file_uploader("Upload JD File", type=["pdf", "docx"])
    if jd_file:
        ext = jd_file.name.lower().split('.')[-1]
        jd_text = extract_text_from_pdf(jd_file) if ext == "pdf" else extract_text_from_docx(jd_file)

# Custom Emphasis
st.markdown("#### 🌟 Custom Emphasis (Optional)")
custom_input_text = st.text_area("Enter any custom preferences (skills, experience, tools, etc.):", height=120)

# Upload Resumes
resume_files = st.file_uploader("Upload Resumes (PDF/DOCX)", type=["pdf", "docx"], accept_multiple_files=True)

# Trigger Resume Ranking
if st.button("🚀 Rank Resumes"):
    if not jd_text.strip() or not resume_files:
        st.warning("Please provide both JD and resumes.")
    else:
        with st.spinner("Ranking resumes..."):
            ranked, text_map = rank_by_embedding(jd_text, resume_files, custom_input_text)
            st.session_state.ranked_resumes = ranked
            st.session_state.resume_texts = text_map
            st.session_state.rank_triggered = True
            st.session_state.selected_candidates = []

# Display Ranked Resumes
if st.session_state.rank_triggered and st.session_state.ranked_resumes:
    st.subheader("🏆 Ranked Resumes")
    for i, (name, score) in enumerate(st.session_state.ranked_resumes, start=1):
        col1, col2 = st.columns([5, 1])
        with col1:
            st.markdown(f"**{i}. {name}** — Score: `{score:.4f}`")
        with col2:
            checkbox_key = f"check_{name}"
            if st.checkbox("Select", key=checkbox_key):
                if name not in st.session_state.selected_candidates:
                    st.session_state.selected_candidates.append(name)
            elif name in st.session_state.selected_candidates:
                st.session_state.selected_candidates.remove(name)
        with st.expander(f"📄 Preview Resume: {name}"):
            st.text_area("Text Content", st.session_state.resume_texts[name], height=200, key=f"text_{name}")

# Screening Test (Disabled)
# if st.session_state.selected_candidates:
#     st.subheader("🧠 Generate & Send Screening Test")

#     if st.button("🧪 Generate Test"):
#         st.session_state.screening_test = generate_screening_test(jd_text)

#     if st.session_state.screening_test:
#         st.text_area("📝 Screening Test", st.session_state.screening_test, height=250)

#         for name in st.session_state.selected_candidates:
#             extracted_text = st.session_state.resume_texts.get(name, "")
#             auto_email = extract_email(extracted_text)
#             default_email = st.session_state.emails.get(name, auto_email)

#             email = st.text_input(f"📧 Email for {name}", value=default_email, key=f"email_{name}")
#             st.session_state.emails[name] = email

#             if st.button(f"📨 Send to {name}", key=f"send_{name}"):
#                 if email and st.session_state.screening_test:
#                     sent = send_email(email, "Your Screening Test", st.session_state.screening_test)
#                     if sent:
#                         st.success(f"✅ Sent to {email}")
#                 else:
#                     st.warning("Missing email or test.")

#         if st.button("📤 Send to All Selected"):
#             if st.session_state.screening_test:
#                 for name, email in st.session_state.emails.items():
#                     if email:
#                         send_email(email, "Your Screening Test", st.session_state.screening_test)
#                 st.success("✅ Screening test sent to all selected candidates.")
#             else:
#                 st.warning("No screening test generated.")

# Hide Streamlit footer
st.markdown("""
    <style>
        footer {visibility: hidden;}
        .stApp {bottom: 0;}
    </style>
""", unsafe_allow_html=True)
