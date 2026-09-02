import os
from dotenv import load_dotenv
from groq import Groq
import chromadb
from sentence_transformers import SentenceTransformer
import streamlit as st

# Load .env file and configuration
load_dotenv()
CHROMA_PATH = "./chroma_data"
COLLECTION = "netflix_titles"
EMBED_MODEL = "all-MiniLM-L6-v2"
GROQ_MODEL = "openai/gpt-oss-120b"

# Enginee
def load_engine():
    """Load the RAG engine with Groq"""
    try:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            try:
                api_key = st.secrets["GROQ_API_KEY"]
            except Exception:
                pass
        if not api_key:
            raise ValueError("GROQ_API_KEY not found in .env or secrets")

        print("API key found!")
        print("Initializing Groq client...")
        client = Groq(api_key=api_key)
        print("Loading embedding model...")
        embed = SentenceTransformer(EMBED_MODEL)
        print("Connecting to ChromaDB...")
        chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
        try:
            coll = chroma_client.get_collection(name=COLLECTION)
            print(f"Collection '{COLLECTION}' found with {coll.count()} documents")
        except Exception:
            print(f"Collection '{COLLECTION}' not found. Creating empty collection...")
            coll = chroma_client.create_collection(name=COLLECTION)

        print("Engine loaded successfully!")
        return embed, coll, client

    except Exception as e:
        print(f"Error loading engine: {e}")
        return None, None, None

# Answer function
def get_answer(query, collection, embed_model, client):
    """Get answer from RAG pipeline using Groq"""
    try:
        print(f"Searching for: {query}")
        query_embedding = embed_model.encode(query).tolist()
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=5
        )
        if not results or not results['documents']:
            return "No results found try different query!"
        movies = results['documents'][0]
        context = "\n\n".join(movies)
        # Prompt for Groq
        prompt = f"""
You are Xpect AI (CineSense AI), a Netflix recommendation assistant.
Based on these movies that match the user's mood, recommend the best ones:
{context}
User's mood: {query}
Give 3 specific movie recommendations from the list above.
Explain why each movie fits the mood.
Be concise and helpful.
"""
        print("Generating recommendations with Groq...")
        response = client.chat.completions.create(
            model=GROQ_MODEL,  
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=500
        )
        return response.choices[0].message.content

    except Exception as e:
        return f"Error: {str(e)}"
