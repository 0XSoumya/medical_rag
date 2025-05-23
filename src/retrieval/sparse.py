from langchain_community.retrievers import BM25Retriever

def get_sparse_retriever(documents, k: int = 3):
    # BM25Retriever.from_documents takes a list of Document objects
    retriever = BM25Retriever.from_documents(documents, k=k)
    return retriever
