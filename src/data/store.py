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
