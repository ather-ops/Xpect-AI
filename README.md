<div align="center">
<img src="https://github.com/ather-ops/Xpect-AI/blob/main/02-Assets/xepct%20ai%20cover.png?raw=true" alt="Xpect AI Cover" width="100%" />
</div>

# Xpect AI

A production-grade RAG system trained on the Netflix titles dataset (8,800+ real titles). You describe what you want to watch in plain language. The pipeline encodes your query into a vector, retrieves the most semantically relevant chunks from a persistent ChromaDB vector store, and sends them to Gemini 2.5 Flash — which returns ranked recommendations with per-title reasoning.

This is not a recommender system. It is a full retrieval-augmented generation pipeline built from scratch over 15 days and 125+ commits.

**Live app:** [xpect-ai-v1.streamlit.app](https://xpect-ai-v1.streamlit.app/)

**Learn RAG from scratch:** [Cortex_RAG](https://github.com/ather-ops/Cortex_RAG) — the foundation repository. One-hot encoding through SentenceTransformers, step by step, fully annotated.

---

## What It Does

No keyword matching. No collaborative filtering. Pure RAG.

You type: "something dark and psychological set in Europe." The pipeline embeds that sentence into a 384-dimensional vector, searches ChromaDB for the nearest title chunks, and sends the retrieved context to Gemini 2.5 Flash. The model returns ranked recommendations with a reason for each — and it can only recommend titles that were actually retrieved, which prevents hallucination.

---

## Pipeline Architecture

![Xpect AI Pipeline Architecture](https://github.com/ather-ops/CineSense-AI/blob/main/02-Assets/architecture1.svg?raw=true)

---

## Repository Structure

![Xpect AI Repository Structure](https://github.com/ather-ops/CineSense-AI/blob/main/02-Assets/repo-structure1.svg?raw=true)

---

## EDA Visuals

Five production charts generated from the raw Netflix dataset during ingestion. Saved to `04-Visuals/`.

![Xpect AI EDA Visuals](https://github.com/ather-ops/Xpect-AI/blob/main/04-Visuals/eda-visuals-combined.png?raw=true)

---

## It Actually Works — The Test Story

The last two days before launch were not smooth. Getting the full pipeline working end-to-end — Streamlit UI through ChromaDB through Gemini — produced errors that took 48 hours of continuous debugging and 40+ commits to resolve.

ChromaDB metadata type mismatches caused `$gte` filters to silently fail. The Gemini API rate limits hit mid-test. Streamlit session state was re-initializing the embedding model on every single query. None of these were obvious. All of them were solved.

**First successful end-to-end query**
![Xpect AI working — test 1](https://github.com/ather-ops/CineSense-AI/blob/main/02-Assets/1.png?raw=true)

**Filtered search with genre and year working**
![Xpect AI working — test 2](https://github.com/ather-ops/CineSense-AI/blob/main/02-Assets/2.png?raw=true)

125+ commits across 15 days. Some are one-line fixes. Some are complete rewrites. None are copy-paste. AI was used in specific places to debug errors — the architecture, the decisions, and the pipeline were built from understanding.

---

## Build Log

### Days 1–3 — Data, Embeddings, Retrieval

CSV ingestion, `fill_missing()` with median/mean strategy, 5-chart EDA dashboard. SentenceTransformer embeddings at 384 dimensions, ChromaDB collection, batch insert at size 100. `retrieve()` with compound `$and` filter support across genre, year, rating, and type.

### Days 4–5 — Chunking

Day 4 built experimental word-count splitting — produced incomplete mid-sentence cuts, degrading embedding quality. Day 5 replaced it with `sentence_chunk()` using NLTK `sent_tokenize` for true sentence-boundary detection. Full pipeline refactor followed: type hints on all functions, explicit metadata casting, named constants throughout.

| Aspect | Day 4 | Day 5 |
|--------|-------|-------|
| Split strategy | Fixed word-count | Sentence-boundary |
| Sentence integrity | Cuts mid-sentence | Always complete |
| Chunk IDs | `chunk_{i}` | `{show_id}_chunk_{N}` |
| Metadata typing | Implicit | Explicit cast per field |

### Day 6 — Gemini 2.5 Flash

`rag_engine.py` complete. `build_context()` deduplicates chunks by title before sending to the LLM. `rag_answer()` sends a grounded prompt — the model can only recommend titles that were retrieved, preventing hallucination. ChromaDB upgraded to `PersistentClient`. API key loaded from environment variable.

### Days 7–11 — Production Hardening and EDA

Pipeline hardening, EDA chart exports, code reviews, Streamlit UI scaffolding.

### Days 12–15 — 48 Hours of Debugging and MVP Launch

48 continuous hours. 40+ commits. ChromaDB type errors, Gemini rate limits, Streamlit session state issues — every layer broke at integration. Every layer was fixed. MVP launched Day 15.

---

## Phase 2 Roadmap

Phase 1 is the MVP. Phase 2 is the real product.

| Feature | Status |
|---------|--------|
| Streamlit MVP — live | Done |
| Full website with custom FastAPI backend, no third-party LLM keys | In progress |
| Chrome Extension — semantic search overlay inside Netflix | Planned |
| Telegram bot — query Xpect AI directly from Telegram | Planned |
| Amazon product search RAG (separate repo, next 5 days) | Planned |

The full website will not use the Gemini API. It will run self-hosted inference — no third-party key, no rate limits, full control over the model layer.

---

## Task Tracker

| Priority | Task | Status |
|----------|------|--------|
| 1 | `pipeline.py` — full ingestion pipeline to ChromaDB | Done |
| 2 | `rag_engine.py` — retrieval + Gemini 2.5 Flash | Done |
| 3 | EDA visuals — 5 charts to `04-Visuals/` | Done |
| 4 | `app.py` — Streamlit UI | Done |
| 5 | Unit tests — ingestion, chunking, retrieval | In progress |
| 6 | Custom FastAPI backend | In progress |
| 7 | Chrome Extension | Planned |
| 8 | Telegram bot | Planned |

---

## Testing

Tests in `03-Core/tests/` — written alongside the pipeline, not after.

Coverage: `fill_missing()` null handling, `sentence_chunk()` sentence integrity and chunk size, ChromaDB insert schema validation, `retrieve()` semantic correctness and filter behavior.

```bash
pip install pytest
pytest 03-Core/tests/ -v
```

---

## Getting Started

```bash
git clone https://github.com/ather-ops/Xpect-AI.git
cd Xpect-AI
pip install -r requirements.txt
export GEMINI_API_KEY=your_key_here
python 03-Core/pipeline.py
streamlit run 03-Core/app.py
```

Or skip all of this and use the live app: [xpect-ai-v1.streamlit.app](https://xpect-ai-v1.streamlit.app/)

---

## Tech Stack

| Layer | Technology |
|-------|------------|
| Language | Python 3.10+ |
| Data | pandas, numpy |
| Visualization | matplotlib, seaborn |
| Tokenization | NLTK `sent_tokenize` |
| Embeddings | SentenceTransformers `all-MiniLM-L6-v2` |
| Vector Store | ChromaDB `PersistentClient` |
| LLM | Gemini 2.5 Flash |
| UI | Streamlit |
| Testing | pytest |
| Phase 2 backend | FastAPI (custom, self-hosted) |
| Phase 2 extension | Chrome Extension Manifest V3 |

---

## Learn RAG From Scratch

[Cortex_RAG](https://github.com/ather-ops/Cortex_RAG) — the foundation repository. Covers the complete embedding pipeline from one-hot encoding through SentenceTransformers. Every concept powering Xpect AI is explained there first. If you want to understand what this project actually does, start there.

---

## Author

**Ather Assadullah** — Self-taught AI/ML engineer, Kashmir, India.

Completed linear and logistic regression, feature engineering. Skipped KNN and RNN. Went straight into RAG — starting from one-hot encoding and working up to LLM connections over the last month and a half. Built [Cortex_RAG](https://github.com/ather-ops/Cortex_RAG) as the learning foundation. Then built this.

One commit at a time. No shortcuts.

GitHub: [ather-ops](https://github.com/ather-ops)
LinkedIn: [ather-assadullah](https://linkedin.com/in/ather-assadullah-164492301)
Portfolio: [portofolio-eight-fawn.vercel.app](https://portofolio-eight-fawn.vercel.app)
Live App: [xpect-ai-v1.streamlit.app](https://xpect-ai-v1.streamlit.app/)

---

## License

MIT License

---

Phase 1 complete. Phase 2 in progress. Built from scratch in 15 days.
