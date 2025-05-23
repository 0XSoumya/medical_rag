from langchain_community.vectorstores import FAISS
from langchain_mistralai import MistralAIEmbeddings
import os

VECTOR_STORE_PATH = "vector_store.faiss"

def create_vector_store(documents):
    embeddings = MistralAIEmbeddings()
    vector_store = FAISS.from_documents(documents, embeddings)
    vector_store.save_local(VECTOR_STORE_PATH)
    return vector_store

def load_vector_store():
    if os.path.exists(VECTOR_STORE_PATH):
        embeddings = MistralAIEmbeddings()
        return FAISS.load_local(VECTOR_STORE_PATH, embeddings, allow_dangerous_deserialization=True)
    return None

def update_vector_store(new_chunks, path: str = VECTOR_STORE_PATH):
    """
    Incrementally add new document chunks to an existing FAISS store,
    or create it if it does not exist.
    """
    embeddings = MistralAIEmbeddings()
    # 1. Load or create
    if os.path.exists(path):
        store = FAISS.load_local(path, embeddings, allow_dangerous_deserialization=True)
        # 2. Add only new chunks
        store.add_documents(new_chunks, embeddings=embeddings)
    else:
        store = FAISS.from_documents(new_chunks, embeddings)
    # 3. Persist
    store.save_local(path)
    return store
