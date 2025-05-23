import os
from src.data.loader import load_and_split_all_docs
from src.data.store  import create_vector_store, load_vector_store

def test_load_and_split_all_docs():
    docs = load_and_split_all_docs("docs")
    # 1) We expect a list of “chunks” (Document objects)
    assert isinstance(docs, list)
    # 2) There should be at least one chunk
    assert len(docs) > 0
    # 3) Each chunk must have the text attribute page_content
    first = docs[0]
    assert hasattr(first, "page_content")
    assert isinstance(first.page_content, str)
