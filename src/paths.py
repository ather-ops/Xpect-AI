from pathlib import Path

# Project root directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Vector store directory
VECTOR_STORE_DIR = PROJECT_ROOT / "vector_store"

# Vector store files
FAISS_INDEX_PATH = VECTOR_STORE_DIR / "netflix_index"
DOCUMENTS_PATH = VECTOR_STORE_DIR / "documents.pkl"
MOVIES_PATH = VECTOR_STORE_DIR / "movies.pkl"
