# GDPR Advisor - Local RAG with Ollama

A specialized legal assistant using Local AI to query the GDPR (General Data Protection Regulation). This project ensures 100% data privacy by processing all documents offline.

## 🛠️ Technical Stack
- **LLM:** Qwen2.5-7B (via Ollama)
- **Embeddings:** BGE-M3 (High-performance multilingual embeddings)
- **Framework:** LangChain
- **Vector Database:** ChromaDB
- **PDF Processing:** PyMuPDF4LLM (Markdown-optimized conversion)

## 🚀 Getting Started

### Prerequisites
- Python 3.10+
- Ollama installed and running

## 🚀 Quick Start

Follow these steps to set up and run the GDPR Advisor locally:

### 1. Environment Setup
```bash
python -m venv venv
source venv/bin/activate  # Windows: .\venv\Scripts\activate
pip install -r requirements.txt

2. Model Preparation
Ensure Ollama is installed and running, then execute:

Bash
ollama pull qwen2.5:7b
ollama pull bge-m3
3. Running the App
Once the models are downloaded, start the web interface:

Bash
streamlit run app.py

### **💡 Pro-tip for your "Setup" phase:**
Since you mentioned having the official PDF in the `data/` folder, make sure your **`app.py`** points to that exact path. 

**Are you ready for me to give you the full `app.py` code that matches these commands and uses your official PDF?**