# Xpect AI 

**AI-powered Netflix movie discovery using semantic search, RAG, and LLM generation.**

Describe what you want to watch in natural language. Xpect AI converts your query into an embedding, retrieves semantically relevant Netflix titles using **FAISS**, and sends the retrieved context to **Groq (`openai/gpt-oss-20b`)** to generate a grounded response.

###  Live App

**[Try Xpect AI →](https://x-ai-test.streamlit.app/)**



## What It Does

No keyword matching. No traditional recommendation engine.

```text
User Query
    ↓
SentenceTransformer
    ↓
384-dimensional embedding
    ↓
FAISS semantic search
    ↓
Relevant Netflix titles
    ↓
Context construction
    ↓
Groq LLM
    ↓
Natural-language answer
```

The system is built on **8,800+ Netflix movie and TV-show titles**.

---

## Tech Stack

| Layer         | Technology                              |
| ------------- | --------------------------------------- |
| Language      | Python 3.12                             |
| Data          | pandas, NumPy                           |
| Embeddings    | SentenceTransformers `all-MiniLM-L6-v2` |
| Vector Search | FAISS                                   |
| LLM           | Groq `openai/gpt-oss-20b`               |
| UI            | Streamlit                               |
| Environment   | Python virtual environment              |
| Deployment    | Streamlit Community Cloud               |



## Project Structure

```text
Xpect-AI/
├── app/
│   └── app.py
├── assets/
├── Data/
│   ├── netflix_cleaned.csv
│   └── netflix_titles.csv
├── notebooks/
│   ├── Data-Cleaning.ipynb
│   ├── RAG-Pipeline.ipynb
│   └── RAG-Generation.ipynb
├── src/
│   ├── config.py
│   ├── paths.py
│   ├── pipeline.py
│   ├── rag_engine.py
│   └── llm.py
├── vector_store/
│   ├── documents.pkl
│   ├── movies.pkl
│   └── netflix_index
├── requirements.txt
└── README.md
```



## Key Features

*  Semantic movie search
*  Retrieval-Augmented Generation
*  Netflix dataset with 8,800+ titles
*  FAISS vector similarity search
*  Groq LLM generation
*  Context-grounded responses
*  Live Streamlit deployment
*  Persistent vector store

---

## RAG Pipeline

The retrieval layer uses `all-MiniLM-L6-v2` to convert both movie documents and user queries into **384-dimensional embeddings**.

FAISS then performs similarity search against the stored movie embeddings and returns the most relevant documents.

The retrieved documents are passed as context to the Groq LLM, which generates the final response.

---

## Run Locally

```bash
git clone https://github.com/ather-ops/Xpect-AI.git
cd Xpect-AI

python -m venv .venv
```

Activate the environment and install dependencies:

```bash
pip install -r requirements.txt
```

Create a `.env` file:

```env
GROQ_API_KEY=your_api_key_here
```

Run the application:

```bash
streamlit run app/app.py
```

Or use the deployed version:

### 👉 [x-ai-test.streamlit.app](https://x-ai-test.streamlit.app/)



## Roadmap

| Feature                 | Status         |
| ----------------------- | -------------- |
| Data cleaning & EDA     | ✅ Complete     |
| Semantic retrieval      | ✅ Complete     |
| FAISS vector store      | ✅ Complete     |
| LLM generation          | ✅ Complete     |
| Streamlit application   | ✅ Complete     |
| Cloud deployment        | ✅ Complete     |
| Production improvements | 🚧 In progress |
| Custom FastAPI backend  | 🔜 Planned     |
| Chrome Extension        | 🔜 Planned     |
| Telegram bot            | 🔜 Planned     |



## License

MIT License

---

**Phase 1 complete. Built from scratch, deployed, and running live. **
