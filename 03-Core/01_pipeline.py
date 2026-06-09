print("=" * 80)
print("Xpect AI - Full Pipeline")
print("=" * 80)

import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import chromadb
import warnings
warnings.filterwarnings('ignore')
import nltk
from nltk.tokenize import sent_tokenize

nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)

# Load data
try:
    df = pd.read_csv("../01-Data/netflix_titles.csv")
    print(f"File loaded successfully: {len(df)} rows")
except FileNotFoundError:
    print("Error: netflix_titles.csv not found in Data folder")
    exit()
except Exception as e:
    print(f"Error loading data: {e}")
    exit()

# Fill missing values
def analysis(df):
    for col in df.columns:
        if df[col].dtype in ['int64', 'float64']:
            if 'year' in col.lower():
                df[col] = df[col].fillna(df[col].median())
            else:
                df[col] = df[col].fillna(df[col].mean())
        else:
            df[col] = df[col].fillna("unknown")
    return df

df = analysis(df)
print("Missing values handled")

# Sentence chunking
def sentence_chunk(text, max_sentence=2):
    sentences = sent_tokenize(text)
    chunks = []
    for i in range(0, len(sentences), max_sentence):
        chunk = " ".join(sentences[i:i+max_sentence])
        chunks.append(chunk)
    return chunks

all_chunks = []
metadata_chunks = []

print("Creating text chunks...")
for idx, row in df.iterrows():
    combined_parts = [
        str(row["title"]) if pd.notnull(row["title"]) else "",
        str(row["director"]) if pd.notnull(row["director"]) else "",
        str(row["cast"]) if pd.notnull(row["cast"]) else "",
        str(row["listed_in"]) if pd.notnull(row["listed_in"]) else "",
        str(row["description"]) if pd.notnull(row["description"]) else ""
    ]
    combined = " ".join(combined_parts).strip()
    chunks = sentence_chunk(combined, max_sentence=2)
    
    for chunk_idx, chunk in enumerate(chunks):
        all_chunks.append(chunk)
        metadata_chunks.append({
            "show_id": str(row["show_id"]),
            "title": str(row["title"]),
            "type": str(row["type"]),
            "country": str(row["country"]),
            "release_year": int(row["release_year"]),
            "rating": str(row["rating"]),
            "listed_in": str(row["listed_in"]),
            "chunk_index": chunk_idx,
            "total_chunks": len(chunks)
        })

print(f"Created {len(all_chunks)} chunks from {len(df)} documents")

# Generate embeddings
print("Generating embeddings with all-MiniLM-L6-v2...")
model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(all_chunks, show_progress_bar=True)
print(f"Embeddings shape: {embeddings.shape}")

# ChromaDB setup
print("Initializing ChromaDB...")
client = chromadb.PersistentClient(path='../chroma_data')

try:
    client.delete_collection(name="netflix_titles")
    print("Deleted existing collection")
except: 
    pass

collection = client.create_collection(
    name="netflix_titles",
    metadata={"description": "Netflix movies and TV shows"}
)

# Batch insert
print("Inserting embeddings into ChromaDB...")
ids = [f"{meta['show_id']}_chunk_{meta['chunk_index']}" for meta in metadata_chunks]
batch_size = 100

for i in range(0, len(embeddings), batch_size):
    end = i + batch_size
    collection.add(
        ids=ids[i:end],
        embeddings=embeddings[i:end].tolist(),
        metadatas=metadata_chunks[i:end],
        documents=all_chunks[i:end]
    )
    print(f"Inserted batch {i//batch_size + 1}/{(len(embeddings)//batch_size) + 1}")

print("=" * 80)
print(f"Pipeline Complete: {collection.count()} chunks stored in ChromaDB")
print("Database saved to: ../chroma_data")
print("=" * 80)
