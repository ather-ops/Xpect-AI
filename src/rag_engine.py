import os
import chromadb
import google.generativeai as genai
from sentence_transformers import SentenceTransformer

CHROMA_PATH = "../chroma_data"
COLLECTION = "netflix_titles"
EMBED_MODEL = "all-MiniLM-L6-v2"
LLM_MODEL = "gemini-1.5-flash"

def load_resources():
    """
    Load embedding model, ChromaDB collection, and Gemini LLM.
    API key is read from Streamlit Secrets or environment variable.
    """
    # Get API key
    try:
        import streamlit as st
        api_key = st.secrets["GEMINI_API_KEY"]
    except:
        api_key = os.getenv("GEMINI_API_KEY", "")
    
    if not api_key:
        raise ValueError(
            "GEMINI_API_KEY not found.\n"
            "Add it in Streamlit Cloud: Settings > Secrets > GEMINI_API_KEY\n"
            "Or set environment variable locally"
        )
    
    # Load embedding model
    embed_model = SentenceTransformer(EMBED_MODEL)
    
    # Connect to ChromaDB
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = client.get_collection(name=COLLECTION)
    
    # Configure Gemini
    genai.configure(api_key=api_key)
    llm = genai.GenerativeModel(LLM_MODEL)
    
    print(f"CineSense AI loaded: {collection.count()} chunks in database")
    return embed_model, collection, llm


def retrieve(collection, embed_model, query, genre=None, min_year=None, 
             max_year=None, rating=None, movie_type=None, top_k=5):
    """
    Semantic search in ChromaDB with optional metadata filters.
    Returns ChromaDB query results.
    """
    query_emb = embed_model.encode([query])[0]
    conditions = []
    
    if genre:
        conditions.append({"listed_in": {"$contains": genre}})
    if min_year:
        conditions.append({"release_year": {"$gte": min_year}})
    if max_year:
        conditions.append({"release_year": {"$lte": max_year}})
    if rating:
        conditions.append({"rating": {"$eq": rating}})
    if movie_type:
        conditions.append({"type": {"$eq": movie_type}})
    
    where = None
    if len(conditions) > 1:
        where = {"$and": conditions}
    elif len(conditions) == 1:
        where = conditions[0]
    
    return collection.query(
        query_embeddings=[query_emb.tolist()],
        where=where,
        n_results=top_k,
        include=["metadatas", "distances"]
    )


def build_context(results):
    """Format ChromaDB results into readable text for LLM prompt."""
    context = ""
    seen = set()
    rank = 1
    
    for meta in results["metadatas"][0]:
        title = meta["title"]
        if title in seen:
            continue
        seen.add(title)
        context += (
            f"{rank}. {title} ({meta['release_year']})\n"
            f"   Genre: {meta['listed_in']}\n"
            f"   Rating: {meta['rating']} | Type: {meta['type']}\n\n"
        )
        rank += 1
    
    return context


def rag_answer(llm, query, results):
    """Generate answer using Gemini based on retrieved context."""
    context = build_context(results)
    
    if not context.strip():
        return "No matching titles found. Try adjusting your filters or query."
    
    prompt = (
        f"You are CineSense AI, a Netflix recommendation assistant powered by Gemini.\n\n"
        f"User query: '{query}'\n\n"
        f"Task: Recommend the top 3 most relevant titles from the list below.\n"
        f"For each title, write one sentence explaining why it matches the query.\n"
        f"Be conversational and helpful. Only use titles from this list.\n\n"
        f"AVAILABLE TITLES:\n{context}"
    )
    
    response = llm.generate_content(prompt)
    return response.text


def cinesense(query, collection, embed_model, llm, **filters):
    """
    Full RAG pipeline: retrieve relevant titles and generate answer.
    
    Args:
        query: User's natural language query
        collection: ChromaDB collection
        embed_model: SentenceTransformer model
        llm: Gemini model
        **filters: Optional metadata filters (genre, min_year, max_year, rating, movie_type)
    
    Returns:
        results: ChromaDB query results
        answer: Generated recommendation text
    """
    results = retrieve(collection, embed_model, query, **filters)
    answer = rag_answer(llm, query, results)
    return results, answer