import os
from dotenv import load_dotenv
import gradio as gr

from langchain_mistralai import ChatMistralAI, MistralAIEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

# Load env
load_dotenv()

# Paths
PDF_PATH = "docs/1bit.pdf"
VECTOR_PATH = "vectorstores/faiss_index"

# Load or create vector store
if os.path.exists(VECTOR_PATH):
    print("✅ Loading vector store from disk...")
    embeddings = MistralAIEmbeddings()
    vectorstore = FAISS.load_local(VECTOR_PATH, embeddings, allow_dangerous_deserialization=True)
else:
    print("📄 Loading and chunking PDF...")
    loader = PyPDFLoader(PDF_PATH)
    docs = loader.load()
    raw_texts = [doc.page_content for doc in docs]
    metadatas = [doc.metadata for doc in docs]

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.create_documents(texts=raw_texts, metadatas=metadatas)

    print(f"🔍 Creating embeddings for {len(chunks)} chunks...")
    embeddings = MistralAIEmbeddings()
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(VECTOR_PATH)
    print("💾 Vector store saved.")

# Set up retriever and model
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})
model = ChatMistralAI(model="mistral-small")

prompt = PromptTemplate(
    template="""You are a skilled doctor. Based only on the following context, answer the question.
If you don't know the answer, say "I don't know".

Context:
{context}

Question:
{question}
""",
    input_variables=["context", "question"]
)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# RAG chain
parallel_chain = RunnableParallel({
    "context": retriever | RunnableLambda(format_docs),
    "question": RunnablePassthrough()
})
main_chain = parallel_chain | prompt | model | StrOutputParser()

# Gradio UI
def answer_question(user_question):
    try:
        return main_chain.invoke(user_question)
    except Exception as e:
        return f"⚠️ Error: {str(e)}"

demo = gr.Interface(
    fn=answer_question,
    inputs=gr.Textbox(lines=2, placeholder="Enter your medical question...", label="Patient Query"),
    outputs=gr.Textbox(label="Doctor's Answer"),
    title="🧠 Medical Diagnostic Assistant",
    description="Ask a question based on the medical PDF. The AI answers only from the content."
)

if __name__ == "__main__":
    demo.launch()
