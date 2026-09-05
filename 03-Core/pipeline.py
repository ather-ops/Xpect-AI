"""Build the ChromaDB collection of Netflix titles from the source CSV."""

import os
import warnings

warnings.filterwarnings("ignore")

import pandas as pd

try:
    from paths import COLLECTION, EMBED_MODEL, resolve_csv_path, writable_build_path
except ImportError:
    from .paths import COLLECTION, EMBED_MODEL, resolve_csv_path, writable_build_path

BATCH_SIZE = 500


def _sentence_chunk(text, max_sentence=2):
    """Split text into chunks of max_sentence sentences."""
    try:
        import nltk
        from nltk.tokenize import sent_tokenize

        try:
            nltk.data.find("tokenizers/punkt")
        except LookupError:
            nltk.download("punkt", quiet=True)
            nltk.download("punkt_tab", quiet=True)
        sentences = sent_tokenize(text)
    except Exception:
        # NLTK data is often unavailable on hosted runtimes.
        import re

        sentences = [s for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]

    if not sentences:
        return [text] if text.strip() else []

    return [
        " ".join(sentences[i:i + max_sentence])
        for i in range(0, len(sentences), max_sentence)
    ]


def _fill_missing(df):
    for col in df.columns:
        if df[col].dtype in ["int64", "float64"]:
            fill = df[col].median() if "year" in col.lower() else df[col].mean()
            df[col] = df[col].fillna(fill)
        else:
            df[col] = df[col].fillna("unknown")
    return df


def _safe_year(value):
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return 0


def build_chroma_collection(chroma_path=None, csv_path=None, progress=None):
    """Build the netflix_titles collection and return the path it was written to."""
    import chromadb
    from sentence_transformers import SentenceTransformer

    def say(msg):
        print(f"[xpect] {msg}", flush=True)
        if progress:
            try:
                progress(msg)
            except Exception:
                pass

    chroma_path = chroma_path or writable_build_path()
    csv_path = csv_path or resolve_csv_path()
    os.makedirs(chroma_path, exist_ok=True)

    say(f"Loading data from {csv_path}")
    df = _fill_missing(pd.read_csv(csv_path))
    say(f"Loaded {len(df)} rows")

    say("Creating text chunks...")
    all_chunks, metadata_chunks = [], []
    for _, row in df.iterrows():
        combined = " ".join(
            str(row.get(field, ""))
            for field in ("title", "director", "cast", "listed_in", "description")
        ).strip()

        chunks = _sentence_chunk(combined)
        for chunk_idx, chunk in enumerate(chunks):
            all_chunks.append(chunk)
            metadata_chunks.append({
                "show_id": str(row.get("show_id", "")),
                "title": str(row.get("title", "")),
                "type": str(row.get("type", "")),
                "country": str(row.get("country", "")),
                "release_year": _safe_year(row.get("release_year")),
                "rating": str(row.get("rating", "")),
                "listed_in": str(row.get("listed_in", "")),
                "chunk_index": chunk_idx,
                "total_chunks": len(chunks),
            })
    say(f"Created {len(all_chunks)} chunks from {len(df)} titles")

    say(f"Loading embedding model {EMBED_MODEL}...")
    model = SentenceTransformer(EMBED_MODEL)

    say("Connecting to ChromaDB...")
    client = chromadb.PersistentClient(path=chroma_path)
    try:
        client.delete_collection(name=COLLECTION)
        say("Deleted stale collection")
    except Exception:
        pass

    collection = client.create_collection(
        name=COLLECTION,
        metadata={"description": "Netflix movies and TV shows"},
    )

    ids = [
        f"{meta['show_id']}_chunk_{meta['chunk_index']}"
        for meta in metadata_chunks
    ]
    total_batches = (len(all_chunks) + BATCH_SIZE - 1) // BATCH_SIZE

    # Embed in batches to keep peak memory low on small cloud instances.
    for batch_no, start in enumerate(range(0, len(all_chunks), BATCH_SIZE), 1):
        end = start + BATCH_SIZE
        batch_docs = all_chunks[start:end]
        embeddings = model.encode(batch_docs, show_progress_bar=False)
        collection.add(
            ids=ids[start:end],
            embeddings=[e.tolist() for e in embeddings],
            metadatas=metadata_chunks[start:end],
            documents=batch_docs,
        )
        say(f"Indexed batch {batch_no}/{total_batches}")

    say(f"Pipeline complete: {collection.count()} chunks stored at {chroma_path}")
    return chroma_path


if __name__ == "__main__":
    build_chroma_collection()
