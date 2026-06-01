<div align="center">

<img src="https://github.com/ather-ops/Xpect-AI/blob/main/02-Assets/xepct%20ai%20cover.png?raw=true" alt="CineSense AI Banner" width="100%" />

<br/>
<br/>

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat-square&logo=python&logoColor=white&labelColor=0d1117)
![SentenceTransformers](https://img.shields.io/badge/SentenceTransformers-MiniLM-2ECC71?style=flat-square&labelColor=0d1117)
![ChromaDB](https://img.shields.io/badge/ChromaDB-Persistent-E67E22?style=flat-square&labelColor=0d1117)
![Gemini](https://img.shields.io/badge/Gemini-2.5%20Flash-8957E5?style=flat-square&logo=google&logoColor=white&labelColor=0d1117)
![NLTK](https://img.shields.io/badge/NLTK-sent__tokenize-9B59B6?style=flat-square&labelColor=0d1117)
![Streamlit](https://img.shields.io/badge/Streamlit-MVP%20Live-FF4B4B?style=flat-square&logo=streamlit&logoColor=white&labelColor=0d1117)
![Commits](https://img.shields.io/badge/Commits-125%2B-27AE60?style=flat-square&labelColor=0d1117)
![Sprint](https://img.shields.io/badge/Sprint-Day%2015-8957E5?style=flat-square&labelColor=0d1117)
![Phase](https://img.shields.io/badge/Phase-1%20Complete-2ECC71?style=flat-square&labelColor=0d1117)
![License](https://img.shields.io/badge/License-MIT-F39C12?style=flat-square&labelColor=0d1117)

<br/>

### MVP is live

**[cinesense-ai-v1.streamlit.app](https://cinesense-ai-v1.streamlit.app/)**

> Describe what you want to watch. Get ranked Netflix recommendations powered by semantic search and Gemini 2.5 Flash. No keyword matching. No collaborative filtering. Pure RAG.

<br/>

> Phase 1 complete. Phase 2 — full website with own backend, Chrome Extension, Telegram bot — in progress.

</div>

---

## What is CineSense AI

CineSense AI is a production-grade RAG system trained on the Netflix titles dataset (8,800+ real titles). You describe what you want to watch in plain language. The pipeline encodes your query into a vector, retrieves the most semantically relevant chunks from a persistent ChromaDB vector store, and sends them to Gemini 2.5 Flash — which returns ranked recommendations with per-title reasoning.

This is not a recommender system. It is a full retrieval-augmented generation pipeline built from scratch over 15 days and 125+ commits.

**Learn RAG from scratch:** [Cortex\_RAG](https://github.com/ather-ops/Cortex_RAG) — the foundation repo. One-hot encoding through SentenceTransformers, step by step, fully annotated.

---

## Pipeline Architecture

<div align="center">
  <img src="https://github.com/ather-ops/CineSense-AI/blob/main/02-Assets/architecture1.svg?raw=true" alt="CineSense AI Pipeline Architecture" width="72%" />
</div>

---

## Repository Structure

<div align="center">
  <img src="https://github.com/ather-ops/CineSense-AI/blob/main/02-Assets/repo-structure1.svg?raw=true" alt="CineSense AI Repository Structure" width="72%" />
</div>

---

## EDA Visuals

Five production charts generated from the raw Netflix dataset during ingestion. Saved to `04-Visuals/`.

<div align="center">
<table>
<tr>
<td align="center" width="50%">
<img src="https://github.com/ather-ops/CineSense-AI/blob/main/04-Visuals/Content%20_Type%20_%20Movies_%20vs_Tv%20_shows.png?raw=true" width="100%" alt="Content Type Distribution"/>
<br/><sub>Content Type — Movies vs TV Shows</sub>
</td>
<td align="center" width="50%">
<img src="https://github.com/ather-ops/CineSense-AI/blob/main/04-Visuals/Content_Growth_over_the_Year.png?raw=true" width="100%" alt="Content Growth Over the Years"/>
<br/><sub>Content Growth Over the Years</sub>
</td>
</tr>
<tr>
<td align="center" width="50%">
<img src="https://github.com/ather-ops/CineSense-AI/blob/main/04-Visuals/Top_10_Countries.png?raw=true" width="100%" alt="Top 10 Content-Producing Countries"/>
<br/><sub>Top 10 Content-Producing Countries</sub>
</td>
<td align="center" width="50%">
<img src="https://github.com/ather-ops/CineSense-AI/blob/main/04-Visuals/Top_10_genres.png?raw=true" width="100%" alt="Top 10 Genres"/>
<br/><sub>Top 10 Most Popular Genres</sub>
</td>
</tr>
<tr>
<td align="center" colspan="2">
<img src="https://github.com/ather-ops/CineSense-AI/blob/main/04-Visuals/Segment_Distribution.png?raw=true" width="50%" alt="Audience Segment Distribution"/>
<br/><sub>Audience Segment Distribution by Rating</sub>
</td>
</tr>
</table>
</div>

---

## It Actually Works — The Test Story

The last two days before launch were not smooth. Getting the full pipeline working end-to-end — Streamlit UI through ChromaDB through Gemini — produced errors that took 48 hours of continuous debugging and 40+ commits to resolve.

ChromaDB metadata type mismatches caused `$gte` filters to silently fail. The Gemini API rate limits hit mid-test. Streamlit session state was re-initializing the embedding model on every single query. None of these were obvious. All of them were solved.

These are the screenshots from when it finally worked:

<div align="center">
<table>
<tr>
<td align="center" width="50%">
<img src="https://github.com/ather-ops/CineSense-AI/blob/main/02-Assets/1.png?raw=true" width="100%" alt="CineSense AI working — test 1"/>
<br/><sub>First successful end-to-end query</sub>
</td>
<td align="center" width="50%">
<img src="https://github.com/ather-ops/CineSense-AI/blob/main/02-Assets/2.png?raw=true" width="100%" alt="CineSense AI working — test 2"/>
<br/><sub>Filtered search with genre and year working</sub>
</td>
</tr>
</table>
</div>

125+ commits across 15 days. Some are one-line fixes. Some are complete rewrites. None are copy-paste. AI was used in specific places to debug errors — the architecture, the decisions, and the pipeline were built from understanding.

---

## Phase 2 Roadmap

Phase 1 is the MVP. Phase 2 is the real product.

| Feature | Status |
|---|---|
| Streamlit MVP — live | Done |
| Full website with custom FastAPI backend, no third-party LLM keys | In progress |
| Chrome Extension — semantic search overlay inside Netflix | Planned |
| Telegram bot — query CineSense directly from Telegram | Planned |
| Amazon product search RAG (separate repo, next 5 days) | Planned |

The full website will not use the Gemini API. It will run self-hosted inference — no third-party key, no rate limits, full control over the model layer.

---

## Roadmap and Priorities

| Priority | Task | Status |
|---|---|---|
| 1 | `pipeline.py` — full ingestion pipeline to ChromaDB | Done |
| 2 | `rag_engine.py` — retrieval + Gemini 2.5 Flash | Done |
| 3 | EDA visuals — 5 charts to `04-Visuals/` | Done |
| 4 | `app.py` — Streamlit UI | Done |
| 5 | Unit tests — ingestion, chunking, retrieval | In progress |
| 6 | Custom FastAPI backend | In progress |
| 7 | Chrome Extension | Planned |
| 8 | Telegram bot | Planned |

---

## Build Log

### Days 1–3 — Data, Embeddings, Retrieval

CSV ingestion, `fill_missing()` with median/mean strategy, 5-chart EDA dashboard. SentenceTransformer embeddings at 384 dimensions, ChromaDB collection, batch insert at size 100. `retrieve()` with compound `$and` filter support across genre, year, rating, and type.

### Days 4–5 — Chunking

Day 4 built experimental word-count splitting — produced incomplete mid-sentence cuts, degrading embedding quality. Day 5 replaced it with `sentence_chunk()` using NLTK `sent_tokenize` for true sentence-boundary detection. Full pipeline refactor followed: type hints on all functions, explicit metadata casting, named constants throughout.

| Aspect | Day 4 | Day 5 |
|---|---|---|
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
git clone https://github.com/ather-ops/CineSense-AI.git
cd CineSense-AI
pip install -r requirements.txt
export GEMINI_API_KEY=your_key_here
python 03-Core/pipeline.py
streamlit run 03-Core/app.py
```

Or skip all of this and use the live app: **[cinesense-ai-v1.streamlit.app](https://cinesense-ai-v1.streamlit.app/)**

---

## Tech Stack

| Layer | Technology |
|---|---|
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

**[Cortex\_RAG](https://github.com/ather-ops/Cortex_RAG)** — the foundation repo. Covers the complete embedding pipeline from one-hot encoding through SentenceTransformers. Every concept powering CineSense AI is explained there first. If you want to understand what this project actually does, start there.

---

## Author

**Ather Assadullah** — Self-taught AI/ML engineer, Kashmir, India.

Completed linear and logistic regression, feature engineering. Skipped KNN and RNN. Went straight into RAG — starting from one-hot encoding and working up to LLM connections over the last month and a half. Built [Cortex\_RAG](https://github.com/ather-ops/Cortex_RAG) as the learning foundation. Then built this.

One commit at a time. No shortcuts.

[![GitHub](https://img.shields.io/badge/GitHub-ather--ops-181717?style=flat-square&logo=github&labelColor=0d1117)](https://github.com/ather-ops)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-ather--assadullah-0A66C2?style=flat-square&logo=linkedin&labelColor=0d1117)](https://linkedin.com/in/ather-assadullah-164492301)
[![Portfolio](https://img.shields.io/badge/Portfolio-Live-27AE60?style=flat-square&labelColor=0d1117)](https://portofolio-eight-fawn.vercel.app)
[![Live App](https://img.shields.io/badge/CineSense%20AI-Try%20It-8957E5?style=flat-square&logo=streamlit&labelColor=0d1117)](https://cinesense-ai-v1.streamlit.app/)

---

## License

MIT License

---

<div align="center">

Phase 1 complete. Phase 2 in progress. Built from scratch in 15 days.

**[Try CineSense AI — cinesense-ai-v1.streamlit.app](https://cinesense-ai-v1.streamlit.app/)**

</div>
