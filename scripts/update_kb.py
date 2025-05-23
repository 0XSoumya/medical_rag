#!/usr/bin/env python3

import os
import argparse
from src.data.loader import load_and_split_all_docs
from src.data.store import update_vector_store, VECTOR_STORE_PATH

def main(docs_dir: str):
    if not os.path.isdir(docs_dir):
        raise FileNotFoundError(f"Docs folder not found: {docs_dir}")

    print(f"⏳ Loading and splitting documents from '{docs_dir}'...")
    chunks = load_and_split_all_docs(docs_dir)
    print(f"  • {len(chunks)} chunks created.")

    print(f"🔁 Updating vector store at '{VECTOR_STORE_PATH}' with these chunks...")
    vs = update_vector_store(chunks, path=VECTOR_STORE_PATH)
    print(f"✅ Vector store updated successfully. Total indexed chunks: {vs.index.ntotal}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Incrementally update FAISS vector store from docs folder.")
    parser.add_argument(
        "--docs-dir", "-d",
        default="docs",
        help="Path to the folder containing documents to ingest (PDFs, TXTs, etc.)"
    )
    args = parser.parse_args()
    main(args.docs_dir)
