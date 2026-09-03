import os

from dotenv import load_dotenv
from groq import Groq
import chromadb
from sentence_transformers import SentenceTransformer
import streamlit as st

try:
    from paths import (
        COLLECTION,
        EMBED_MODEL,
        resolve_chroma_path,
        store_status,
    )
except ImportError:  # pragma: no cover
    from .paths import (
        COLLECTION,
        EMBED_MODEL,
        resolve_chroma_path,
        store_status,
    )

load_dotenv()

GROQ_MODEL = "openai/gpt-oss-120b"

# Resolved lazily so importing config never triggers filesystem work.
CHROMA_PATH = None


def _get_api_key():
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        try:
            api_key = st.secrets["GROQ_API_KEY"]
        except Exception:
            pass
    return api_key


def load_engine(allow_build=True, progress=None):
    """
    Load the RAG engine (embedding model, Chroma collection, Groq client).

    Resolves `chroma_data` across several candidate locations, detects Git LFS
    pointer stubs (the usual Streamlit Cloud failure), and rebuilds the
    collection from the CSV when no usable store is present.

    Returns (embed_model, collection, groq_client) or (None, None, None).
    """
    global CHROMA_PATH

    try:
        api_key = _get_api_key()
        if not api_key:
            raise ValueError(
                "GROQ_API_KEY not found. Add it to .env locally, or to "
                "Streamlit Cloud under Settings > Secrets."
            )
        print("[xpect] API key found.")

        chroma_path, status = resolve_chroma_path()
        CHROMA_PATH = chroma_path

        if status != "ok":
            if status == "lfs-pointers":
                print(
                    "[xpect] chroma_data contains Git LFS pointer stubs, not "
                    "real data. Streamlit Cloud does not run the LFS smudge "
                    "filter, so the embeddings were never downloaded."
                )
            if not allow_build:
                return None, None, None
            print("[xpect] Building the collection from the CSV instead...")
            try:
                from pipeline import build_chroma_collection
            except ImportError:
                from .pipeline import build_chroma_collection
            build_chroma_collection(chroma_path=chroma_path, progress=progress)

        print("[xpect] Loading embedding model...")
        embed = SentenceTransformer(EMBED_MODEL)

        print(f"[xpect] Connecting to ChromaDB at {chroma_path}")
        chroma_client = chromadb.PersistentClient(path=chroma_path)
        collection = chroma_client.get_collection(name=COLLECTION)
        count = collection.count()
        print(f"[xpect] Collection '{COLLECTION}' loaded with {count} chunks.")

        if count == 0:
            # An empty collection would silently return zero recommendations.
            if not allow_build:
                return None, None, None
            print("[xpect] Collection is empty. Rebuilding...")
            try:
                from pipeline import build_chroma_collection
            except ImportError:
                from .pipeline import build_chroma_collection
            build_chroma_collection(chroma_path=chroma_path, progress=progress)
            chroma_client = chromadb.PersistentClient(path=chroma_path)
            collection = chroma_client.get_collection(name=COLLECTION)
            print(f"[xpect] Rebuilt with {collection.count()} chunks.")

        client = Groq(api_key=api_key)
        print("[xpect] Engine loaded successfully.")
        return embed, collection, client

    except Exception as exc:
        print(f"[xpect] Error loading engine: {exc}")
        return None, None, None


def diagnostics():
    """Human-readable state of the vector store, for logs and the UI."""
    try:
        from paths import candidate_chroma_paths
    except ImportError:  # pragma: no cover
        from .paths import candidate_chroma_paths

    return [(p, store_status(p)) for p in candidate_chroma_paths()]


def get_answer(query, collection, embed_model, client):
    """Retrieve matching titles and generate recommendations with Groq."""
    try:
        print(f"[xpect] Searching for: {query}")
        query_embedding = embed_model.encode(query).tolist()
        results = collection.query(query_embeddings=[query_embedding], n_results=5)

        documents = (results or {}).get("documents") or []
        movies = documents[0] if documents else []
        if not movies:
            return "No results found - try a different query!"

        context = "\n\n".join(movies)
        prompt = f"""
You are Xpect AI (CineSense AI), a Netflix recommendation assistant.
Based on these movies that match the user's mood, recommend the best ones:
{context}
User's mood: {query}
Give 3 specific movie recommendations from the list above.
Explain why each movie fits the mood.
Be concise and helpful.
"""
        print("[xpect] Generating recommendations with Groq...")
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=500,
        )
        return response.choices[0].message.content

    except Exception as exc:
        return f"Error: {exc}"
