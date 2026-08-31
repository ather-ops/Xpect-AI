import streamlit as st
from config import load_engine, get_answer

st.set_page_config(
    page_title="Xpect AI",
    page_icon="🪅",
    layout="centered"
)

st.title("Xpect AI")
st.caption("Describe your mood. Get Netflix recommendations.")

# Load engine
@st.cache_resource
def load_engine_cached():
    try:
        embed, coll, llm = load_engine()
        return embed, coll, llm
    except Exception as e:
        st.error(f"Error: {e}")
        return None, None, None

embed_model, collection, llm = load_engine_cached()

if not collection:
    st.warning("Collection not found. Run pipeline first.")
    st.stop()

# Session
if "messages" not in st.session_state:
    st.session_state.messages = []

# Show messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# Quick prompts
st.divider()
cols = st.columns(4)
prompts = ["emotional", "thriller", "comedy", "sci-fi", "classic", "family", "hidden gem", "series"]

clicked = None
for i, p in enumerate(prompts):
    with cols[i % 4]:
        if st.button(p, key=f"q{i}", use_container_width=True):
            clicked = p

st.divider()

# Input
prompt = st.chat_input("Ask Xpect AI...")

if clicked:
    prompt = clicked

if prompt:
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

# Clear button
if st.session_state.messages:
    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.rerun()
