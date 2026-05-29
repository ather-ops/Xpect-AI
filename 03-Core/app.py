"""
Xpect AI - Netflix Recommendation Engine
Author: ather-ops
"""

import os
import streamlit as st
import chromadb
import google.generativeai as genai
from sentence_transformers import SentenceTransformer

CHROMA_PATH = "./chroma_data"
COLLECTION = "netflix_titles"
EMBED_MODEL = "all-MiniLM-L6-v2"
LLM_MODEL = "models/gemini-2.5-flash"

st.set_page_config(
    page_title="Xpect AI",
    page_icon="🎬",
    layout="centered",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:ital,wght@0,400;0,500;0,600;0,700;1,700;1,900&display=swap');

/* BLACK BACKGROUND EVERYWHERE */
html, body, .stApp, 
[data-testid="stAppViewContainer"],
[data-testid="stHeader"],
[data-testid="stBottomBlockContainer"],
section, main, div {
    background-color: #000000 !important;
    font-family: 'Inter', sans-serif;
}

.block-container {
    max-width: 780px;
    padding: 2rem 1rem 5rem;
}

header, footer, #MainMenu, [data-testid="stToolbar"] {
    visibility: hidden;
}

/* HERO SECTION */
.hero {
    text-align: center;
    padding: 48px 20px 36px;
    margin-bottom: 32px;
}

.hero-title {
    font-size: 68px;
    line-height: 1;
    font-weight: 900;
    font-style: italic;
    letter-spacing: -2px;
    margin: 0;
    color: #ffffff;
}

.hero-subtitle {
    color: #9ca3af;
    font-size: 16px;
    font-style: italic;
    margin-top: 16px;
    line-height: 1.7;
}

/* QUICK PROMPTS */
.stButton > button {
    background-color: #000000 !important;
    color: #9ca3af !important;
    border: 1px solid #333333 !important;
    border-radius: 8px !important;
    font-size: 13px !important;
    padding: 10px 14px !important;
    width: 100% !important;
    transition: all 0.2s ease !important;
}

.stButton > button:hover {
    background: linear-gradient(135deg, #3b82f6, #ec4899) !important;
    color: #ffffff !important;
    border-color: transparent !important;
}

/* USER MESSAGE - SIMPLE PINK */
.user-msg {
    display: block;
    width: fit-content;
    max-width: 75%;
    margin: 14px 0 14px auto;
    padding: 14px 18px;
    border-radius: 18px;
    background-color: #ec4899;
    color: #ffffff;
    font-size: 15px;
    line-height: 1.6;
}

/* AI MESSAGE - SIMPLE BLUE/PINK GRADIENT */
.ai-msg {
    display: block;
    width: fit-content;
    max-width: 85%;
    margin: 14px auto 14px 0;
    padding: 14px 18px;
    border-radius: 18px;
    background: linear-gradient(135deg, #3b82f6, #ec4899);
    color: #ffffff;
    font-size: 15px;
    line-height: 1.7;
    white-space: pre-wrap;
}

/* EMPTY STATE */
.empty-box {
    text-align: center;
    margin: 48px 0 36px;
    padding: 36px 24px;
    border: 1px solid #1f1f1f;
    border-radius: 16px;
}

.empty-title {
    color: #f9fafb;
    font-size: 20px;
    font-weight: 700;
    margin-bottom: 12px;
}

.empty-text {
    color: #6b7280;
    font-size: 14px;
    line-height: 1.7;
}

/* CHAT INPUT */
[data-testid="stChatInputContainer"],
[data-testid="stChatInput"],
[data-testid="stChatInput"] > div,
[data-testid="stChatInput"] form,
[data-testid="stBottomBlockContainer"] {
    background-color: #000000 !important;
}

[data-testid="stChatInput"] textarea {
    background-color: #0a0a0a !important;
    color: #f9fafb !important;
    border: 1px solid #333333 !important;
    border-radius: 12px !important;
    caret-color: #ec4899 !important;
    font-size: 15px !important;
    padding: 14px 18px !important;
}

[data-testid="stChatInput"] textarea::placeholder {
    color: #6b7280 !important;
}

[data-testid="stChatInput"] textarea:focus {
    border: 1px solid #3b82f6 !important;
    outline: none !important;
}

/* SEND BUTTON */
[data-testid="stChatInput"] button {
    background: linear-gradient(135deg, #3b82f6, #ec4899) !important;
    color: #ffffff !important;
    border-radius: 999px !important;
    border: none !important;
}

[data-testid="stChatInput"] button:hover {
    opacity: 0.9;
}

/* FOOTER */
.footer {
    text-align: center;
    color: #4b5563;
    font-size: 11px;
    margin-top: 40px;
}

</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner="Loading Xpect AI...")
def load_engine():
    try:
        api_key = st.secrets["GEMINI_API_KEY"]
    except Exception:
        api_key = os.getenv("GEMINI_API_KEY", "")

    if not api_key:
        raise ValueError("GEMINI_API_KEY not found")

    genai.configure(api_key=api_key)

    embed = SentenceTransformer(EMBED_MODEL)
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    coll = client.get_collection(name=COLLECTION)
    llm = genai.GenerativeModel(LLM_MODEL)

    return embed, coll, llm


def get_answer(query, collection, embed_model, llm):
    q_emb = embed_model.encode([query])[0]

    results = collection.query(
        query_embeddings=[q_emb.tolist()],
        n_results=5,
        include=["metadatas"]
    )

    seen = set()
    lines = []

    for meta in results["metadatas"][0]:
        title = str(meta.get("title", "Unknown"))
        if title in seen:
            continue
        seen.add(title)

        year = meta.get("release_year", "Unknown")
        genre = meta.get("listed_in", "Unknown")

        lines.append(f"- {title} ({year})\n  Genre: {genre}")

    context = "\n\n".join(lines)

    if not context.strip():
        return "No matches found. Try different words."

    prompt = f"""You are Xpect AI.

User query: {query}

Retrieved titles:
{context}

Recommend top 3. For each: title, year, one reason.
Be concise. Only use retrieved titles."""

    response = llm.generate_content(prompt)
    return response.text


# HERO
st.markdown("""
<div class="hero">
    <h1 class="hero-title">Xpect AI</h1>
    <div class="hero-subtitle">
        Describe your mood. Get Netflix recommendations.
    </div>
</div>
""", unsafe_allow_html=True)

# LOAD
try:
    embed_model, collection, llm = load_engine()
except Exception as e:
    st.error(f"Error: {e}")
    st.stop()

# SESSION
if "messages" not in st.session_state:
    st.session_state.messages = []

# QUICK PROMPTS
prompts = [
    "emotional",
    "thriller",
    "comedy",
    "sci-fi",
    "classic",
    "family",
    "hidden gem",
    "series"
]

cols = st.columns(4)
clicked = None

for i, p in enumerate(prompts):
    with cols[i % 4]:
        if st.button(p, key=f"q{i}"):
            clicked = p

# EMPTY STATE
if not st.session_state.messages:
    st.markdown("""
    <div class="empty-box">
        <div class="empty-title">What do you want to watch?</div>
        <div class="empty-text">
            Try: emotional drama, thriller, sci-fi, or describe a feeling.
        </div>
    </div>
    """, unsafe_allow_html=True)

# MESSAGES
for msg in st.session_state.messages:
    safe = msg["content"].replace("<", "&lt;").replace(">", "&gt;")

    if msg["role"] == "user":
        st.markdown(f'<div class="user-msg">{safe}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="ai-msg">{safe}</div>', unsafe_allow_html=True)

# INPUT
typed = st.chat_input("Ask Xpect AI...")

query = clicked or typed

if query:
    query = query.strip()
    if query:
        st.session_state.messages.append({"role": "user", "content": query})

        with st.spinner("Thinking..."):
            try:
                answer = get_answer(query, collection, embed_model, llm)
            except Exception as e:
                answer = f"Error: {str(e)}"

        st.session_state.messages.append({"role": "assistant", "content": answer})
        st.rerun()

# CLEAR
if st.session_state.messages:
    if st.button("Clear", key="clr"):
        st.session_state.messages = []
        st.rerun()

# FOOTER
st.markdown('<div class="footer">Xpect AI · ather-ops</div>', unsafe_allow_html=True)
