# Step 1: Imports
import faiss
import pandas as pd
import pickle
from sentence_transformers import SentenceTransformer
from src.llm  import generate_answer
from src.paths import(
    FAISS_INDEX_PATH,
    DOCUMENTS_PATH,
    MOVIES_PATH
)
# Step 2: Loads
# index
index = faiss.read_index(str(FAISS_INDEX_PATH))
# documents pkl
with open(DOCUMENTS_PATH,"rb") as f:
    documents=pickle.load(f)

# movies pkl
movies=pd.read_pickle(MOVIES_PATH)

# embedding model
model=SentenceTransformer("all-MiniLM-L6-v2")

# Test
print(index.ntotal)
print(len(documents))
print(len(movies))

# retrival function
def retrive_quries(query,top_k=5):
    query_embd=model.encode([query]).astype("float32")
    distances,indices=index.search(
        query_embd,
        top_k
    )
    result=[]
    for idx in indices[0]:
        result.append(documents[idx])

    return result

# ask xpect function
def ask_xpect(query):
    result=retrive_quries(query)
    context="\n\n -- \n\n".join(result)
    answer=generate_answer(query,context)
    return answer

# test query
query="Something old and scary"
print(ask_xpect(query))
