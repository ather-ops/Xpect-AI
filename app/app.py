import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
import streamlit as st
from src.rag_engine import ask_xpect

st.set_page_config(
    page_title="X ai",
    page_icon="✨",
    layout="centered"
)

st.title("X ai")
st.write("AI-powered Netflix movie discovery using semantic search and RAG")

query=st.text_input(
    "Ask X ai",
    placeholder="get your movie recommendation!"
)
if st.button("Find Movies"):
    if not query.strip():
        st.warning("First enter your mood!")
    else:
        try:
           with st.spinner("X is Thinking"):
              answer=ask_xpect(query)
              st.write(answer)
        except Exception as e:
            st.error("Something went wrong. Please try again!")
