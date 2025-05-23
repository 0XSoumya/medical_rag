import os
from glob import glob
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


def load_and_split_all_docs(
    docs_dir: str = "docs",
    chunk_size: int = 1000,
    chunk_overlap: int = 200
) -> list:
    """
    Load all supported documents from the given directory, split them into
    text chunks, and preserve metadata.

    Args:
        docs_dir: Path to the folder containing PDF/TXT files.
        chunk_size: Maximum characters per chunk.
        chunk_overlap: Number of characters to overlap between chunks.

    Returns:
        List of langchain.schema.Document objects with page_content and metadata.
    """
    # 1) Gather all files
    patterns = ["*.pdf", "*.txt"]
    paths = []
    for pattern in patterns:
        paths.extend(glob(os.path.join(docs_dir, pattern)))

    # 2) Load documents
    docs = []
    for p in paths:
        if p.lower().endswith(".pdf"):
            docs.extend(PyPDFLoader(p).load())
        elif p.lower().endswith(".txt"):
            docs.extend(TextLoader(p).load())

    # 3) Split into chunks (preserves metadata)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    return splitter.split_documents(docs)