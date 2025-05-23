from src.data.store import load_vector_store
from src.retrieval.fusion import get_fusion_retriever
from src.data.loader import load_and_split_all_docs
from src.utils.pipeline import build_chain
import gradio as gr

# 1. Load (or re-use) the vector store
vector_store = load_vector_store()
chunks = load_and_split_all_docs("docs")
retriever = get_fusion_retriever(vector_store, chunks, k=5, weights=(0.7, 0.3))  # adjust as needed

main_chain   = build_chain(retriever)       # matches pipeline.py

# 2. Wrap it in a Gradio interface
def answer_question(user_question):
    try:
        return main_chain.invoke(user_question)
    except Exception as e:
        return f"⚠️ Error: {e}"

demo = gr.Interface(
    fn=answer_question,
    inputs=gr.Textbox(lines=2, placeholder="Enter your medical question...", label="Patient Query"),
    outputs=gr.Textbox(label="Doctor's Answer"),
    title="🧠 Medical Diagnostic Assistant",
    description="Ask a medical question based on the provided knowledge base."
)

if __name__ == "__main__":
    demo.launch()
