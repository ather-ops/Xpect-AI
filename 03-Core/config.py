import os
import chromadb
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
import streamlit as st
from dotenv import load_dotenv

# Load .env file
load_dotenv()

# Paths
CHROMA_PATH = "./chroma_data"
COLLECTION = "netflix_titles"
EMBED_MODEL = "all-MiniLM-L6-v2"
LLM_MODEL = "models/gemini-2.5-flash"

def load_engine():
    """Load the RAG engine"""
    try:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            try:
                api_key = st.secrets["GEMINI_API_KEY"]
            except Exception:
                pass

        if not api_key:
            raise ValueError("GEMINI_API_KEY not found in .env or secrets")

        # Configure Gemini
        genai.configure(api_key=api_key)

        # Load embedding model
        embed = SentenceTransformer(EMBED_MODEL)

        # Load ChromaDB
        client = chromadb.PersistentClient(path=CHROMA_PATH)

        try:
            coll = client.get_collection(name=COLLECTION)
        except Exception:
            coll = client.create_collection(name=COLLECTION)

        # Load Gemini model
        llm = genai.GenerativeModel(LLM_MODEL)

        return embed, coll, llm

    except Exception as e:
        print(f"Error loading engine: {e}")
        return None, None, None

def get_answer(query, collection, embed_model, llm):
    """Get answer from RAG pipeline"""
    try:
        # Get embedding
        query_embedding = embed_model.encode(query).tolist()

        # Search
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=5
        )

        if not results or not results['documents']:
            return "No results found. Try a different query!"

        # Build context
        context = "\n\n".join(results['documents'][0])

        # Build prompt
        prompt = f"""
        You are Xpect AI, a Netflix recommendation assistant.

        Context:
        {context}

        User: {query}

        Give specific recommendations from the context.
        """

        # Get response
        response = llm.generate_content(prompt)

        return response.text

    except Exception as e:
        return f"Error: {str(e)}"
