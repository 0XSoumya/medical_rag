#!/usr/bin/env python3
from src.data.loader import load_and_split_all_docs
from src.data.store  import load_vector_store, create_vector_store, update_vector_store


if __name__ == "__main__":
    print("⏳ Bootstrapping KB…")
    chunks = load_and_split_all_docs("docs")

    vs = load_vector_store()
    if vs is None:
        vs = create_vector_store(chunks)
        print("✅ Vector store created and saved.")
    else:
        print("✅ Vector store already exists; loaded from disk.")
