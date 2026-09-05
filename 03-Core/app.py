import os
import sys

import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import diagnostics, load_engine, get_answer

st.set_page_config(
    page_title="Xpect AI",
    page_icon="\U0001fa85",
    layout="centered"
)


def log_diagnostics():
    """Report where the vector store was found, visible in deploy logs."""
    print("[xpect] ---- startup diagnostics ----")
    print(f"[xpect] cwd: {os.getcwd()}")
    for path, status in diagnostics():
        print(f"[xpect]   {status:<13} {path}")
    print("[xpect] -----------------------------")


log_diagnostics()


@st.cache_resource(show_spinner=False)
def load_engine_cached():
    """Load the engine, building the vector store on first boot if needed."""
    status_box = st.empty()

    def progress(msg):
        status_box.info(f"\u26a1 First-time setup: {msg}")

    try:
        embed, coll, llm = load_engine(progress=progress)
    except Exception as exc:
        status_box.empty()
        st.error(f"Error: {exc}")
        return None, None, None

    status_box.empty()
    return embed, coll, llm


with st.spinner("Warming up Xpect AI... first boot builds the movie index."):
    embed_model, collection, llm = load_engine_cached()

if collection is None:
    st.error("Could not load the movie database.")
    with st.expander("Diagnostics"):
        st.write(f"Working directory: `{os.getcwd()}`")
        for path, status in diagnostics():
            st.write(f"- `{status}` - `{path}`")
        st.caption(
            "`lfs-pointers` means chroma_data was committed via Git LFS but the "
            "deploy never downloaded the real files. Check that GROQ_API_KEY is "
            "set in Settings > Secrets, then reboot."
        )
    st.stop()

st.title("Xpect AI")
st.caption("Describe your mood. Get Netflix recommendations.")


if "messages" not in st.session_state:
    st.session_state.messages = []

if "last_query" not in st.session_state:
    st.session_state.last_query = ""

if "question_count" not in st.session_state:
    st.session_state.question_count = 0

MAX_QUESTIONS = 5

remaining = MAX_QUESTIONS - st.session_state.question_count

if remaining > 0:
    st.info(f"⚙️ {remaining} questions left. Don't waste them on 'what's the weather' — we both know you're here for the movies 😏")
else:
    st.warning("🤦‍♂️ You've used all 5 questions! Go watch one of the movies you searched for — you’ve earned it. Xpect AI is taking a 30-minute break too. My tokens are crying and honestly so am I. Come back after a movie and some snacks 🍿😭")
    st.stop()

@st.cache_data
def get_quick_prompts():
    return ["emotional", "thriller", "comedy", "sci-fi", "classic", "family", "hidden gem", "series"]

quick_prompts = get_quick_prompts()

with st.container():
    cols = st.columns(4)
    clicked = None
    for i, p in enumerate(quick_prompts):
        with cols[i % 4]:
            if st.button(p, key=f"q_{i}", use_container_width=True):
                clicked = p



for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

prompt = st.chat_input("Ask Xpect AI...")

if clicked:
    prompt = clicked


if prompt and prompt != st.session_state.last_query:
    if st.session_state.question_count >= MAX_QUESTIONS:
        st.warning("🤦‍♂️ You've used all 5 questions! Go watch one of the movies you searched for — you’ve earned it. Xpect AI is taking a 30-minute break too. My tokens are crying and honestly so am I. Come back after a movie and some snacks 🍿😭")
        st.stop()

    st.session_state.last_query = prompt
    st.session_state.question_count += 1
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.write(prompt)
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                response = get_answer(prompt, collection, embed_model, llm)
                st.write(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
            except Exception as e:
                st.error(f"Error: {e}")

    st.rerun()
if st.session_state.messages:
    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.session_state.last_query = ""
        st.session_state.question_count = 0
        st.rerun()
