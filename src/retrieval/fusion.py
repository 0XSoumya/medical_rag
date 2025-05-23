from langchain.schema import BaseRetriever
from langchain.retrievers import EnsembleRetriever

from src.retrieval.dense import get_retriever as get_dense_retriever
from src.retrieval.sparse import get_sparse_retriever

def get_fusion_retriever(vector_store, documents, k: int = 3, weights=(0.7, 0.3)) -> BaseRetriever:
    """
    Combines:
      - Dense retriever (FAISS)
      - Sparse retriever (BM25)
    using weighted rank fusion via LangChain's EnsembleRetriever.
    """
    dense_retriever = get_dense_retriever(vector_store, k=k)
    sparse_retriever = get_sparse_retriever(documents, k=k)

    return EnsembleRetriever(
        retrievers=[dense_retriever, sparse_retriever],
        weights=list(weights),
    )
