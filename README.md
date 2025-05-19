# LEXA: Local Retrieval-Augmented Generation System

LEXA is a lightweight, fully local Retrieval-Augmented Generation (RAG) system that uses a small language model (TinyLlama 1.1B) to answer user queries based on relevant documents stored locally.

---

## 🔧 Features

- 🔍 Semantic search using SentenceTransformers + FAISS
- 🧠 Local language generation using TinyLlama (HuggingFace)
- ⚡ FastAPI backend for API exposure
- 💬 Supports both document-grounded answers and freeform LLM prompts

---

## 🏗️ Project Structure

LEXA/
├── backend/
│ ├── app.py # FastAPI app with endpoints
│ ├── generation.py # TinyLlama-based response generation
│ ├── retrieval.py # Semantic document retrieval logic
├── data/
│ └── documents/ # Local .txt files used for context
├── requirements.txt

yaml
Copy
Edit

---

## 🚀 Quickstart

### 1. Clone the repository
```bash
git clone https://github.com/your-username/LEXA.git
cd LEXA
2. Set up the virtual environment

python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

3. Add your documents
Place .txt files in:

data/documents/
4. Start the API server
bash
Copy
Edit
uvicorn backend.app:app --reload --port 8000
5. Test the endpoints
Go to: http://127.0.0.1:8000/docs
/ask → for document-grounded queries

/generate → for pure LLM generation (no documents)

🧠 Example Use Cases
Ask LEXA: “What is reinforcement learning?”

Or generate: “Write a Python function to compute the Fibonacci series.”

📚 Future Work
Add early-exit logic to reduce inference time

Extend support for PDF/Markdown document ingestion

Build a simple frontend interface using React or Streamlit

Add logging and benchmarking features

📦 Requirements
Python 3.10+

torch, transformers, fastapi, faiss-cpu, sentence-transformers, uvicorn

Install via:

bash
Copy
Edit
pip install -r requirements.txt
📬 Contact
Sri Harsha Mudumba
📫 srim@iastate.edu
🔗 LinkedIn
