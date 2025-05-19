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

