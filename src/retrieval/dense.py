def get_retriever(vector_store, k=3):
    return vector_store.as_retriever(search_type="similarity", search_kwargs={"k": k})

def format_docs(retrieved_docs):
    return "\n\n".join(doc.page_content for doc in retrieved_docs)
