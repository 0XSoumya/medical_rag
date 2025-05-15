# 🩺 Medical-RAG

**Medical-RAG** is an AI-powered chatbot that performs medical diagnostics by combining patient data with external medical knowledge using a Retrieval-Augmented Generation (RAG) pipeline.

## 🚀 Project Overview

This project aims to build a reliable, low-hallucination chatbot that can:
- Ingest patient symptoms and records
- Retrieve relevant documents from a medical knowledge base
- Generate clinically relevant, explainable diagnostic responses

## 📁 Folder Structure

```plaintext
medical-rag/
│
├── src/                  # Main source code
│   ├── data/             # Load/store patient data
│   ├── retrieval/        # Sparse, dense, hybrid search
│   ├── llm/              # LLM prompts and API clients
│   ├── utils/            # Logging, helpers, etc.
│   └── pipeline.py       # RAG orchestration
│
├── scripts/              # KB bootstrapper and other setup scripts
├── tests/                # Unit tests
├── docs/                 # Architecture diagrams, notes
├── .env                  # API keys and environment settings
├── .gitignore
├── requirements.txt
├── setup.sh              # Project setup script
└── README.md
```

## 🧠 Features

- ✅ Multi-vector retrieval (sparse, dense, fusion)
- ✅ Patient-aware document routing
- ✅ Streamlit/Gradio interface (planned)
- ✅ Post-RAG fact-checking (planned)
- ✅ Medical model integration (e.g. ClinicalCamel) (planned)

## ⚙️ Getting Started

1. **Clone the repo**
   ```bash
   git clone https://github.com/your-username/medical-rag.git
   cd medical-rag
   ```

2. **Setup environment**
   ```bash
   bash setup.sh
   source venv/bin/activate
   pip install -r requirements.txt
   ```

3. **Run the pipeline**
   ```bash
   python src/pipeline.py
   ```

## 🏗️ Technologies

- Python, LangChain, OpenAI/Mistral APIs
- FAISS / Elasticsearch
- Streamlit or Gradio
- Clinical/NLP models (Med-Alpaca, ClinicalCamel)

## 🤖 Future Enhancements

- Hybrid retrieval rank fusion
- UMLS/SNOMED integration
- Severity-based triage output
- Voice input interface
- Docker deployment

## 📄 License

This project is open source and available under the [MIT License](LICENSE).
