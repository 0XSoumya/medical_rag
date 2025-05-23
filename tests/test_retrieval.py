from src.data.loader import load_and_split_docs
from src.data.store import create_vector_store
from src.retrieval.dense import get_retriever
from src.retrieval.sparse import get_sparse_retriever
from src.retrieval.fusion import get_fusion_retriever

def test_dense_retriever():
    docs = load_and_split_docs("docs")
    store = create_vector_store(docs)
    retriever = get_retriever(store, k=2)
    results = retriever.invoke("What is diabetes?")
    assert isinstance(results, list)

def test_sparse_retriever():
    docs = load_and_split_docs("docs")
    retriever = get_sparse_retriever(docs, k=2)
    results = retriever.invoke("What is hypertension?")
    assert isinstance(results, list)

def test_fusion_retriever():
    docs = load_and_split_docs("docs")
    store = create_vector_store(docs)
    retriever = get_fusion_retriever(store, docs, k=2)
    results = retriever.invoke("How do you treat asthma?")
    assert isinstance(results, list)
