# LEXA: Lightweight Local Retrieval-Augmented Generation System

**LEXA** is a lightweight, fully local **Retrieval-Augmented Generation (RAG)** framework designed for offline use.  
It integrates **TinyLlama 1.1B** for language generation and **Sentence Transformers** for semantic document retrieval.  
The system enables efficient, privacy-preserving question answering directly over local text files—no external API calls or internet access required.

---

## Overview

LEXA is built for developers, researchers, and students who need a self-contained RAG setup that runs entirely on local hardware.  
By combining a compact language model with fast semantic search, it allows users to query documents, generate summaries, and retrieve relevant context without relying on cloud-based inference services.

The architecture emphasizes:
- **Lightweight deployment** (runs comfortably on laptops or edge devices)
- **Full data privacy** (documents never leave your local system)
- **Ease of setup** (minimal dependencies, modular backend)

---

## Getting Started

Follow these steps to install and run LEXA locally.

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/LEXA.git
cd LEXA
```

### 2. Set Up the Virtual Environment
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 3. Add Your Documents
Place your text files in the following directory:
```
data/documents/
```
LEXA uses these documents for semantic retrieval and context generation.

### 4. Start the API Server
```bash
uvicorn backend.app:app --reload --port 8000
```

### 5. Access the API Documentation
Open your browser and navigate to:
```
http://127.0.0.1:8000/docs
```
From there, you can test endpoints interactively.

---

## Example Usage

Use the `/ask` endpoint to perform question answering or summarization.  
Example JSON payload:

```json
{
  "task": "question answering",
  "question": "What is deep learning?"
}
```

LEXA retrieves the most relevant text chunks using **Sentence Transformers**, feeds them into **TinyLlama 1.1B**, and returns a concise, context-aware response.

---

## Architecture

1. **Document Embedding:**  
   Text files are split into manageable chunks and embedded using a Sentence Transformer model.

2. **Semantic Retrieval:**  
   When a query is received, LEXA searches for the most relevant chunks using cosine similarity.

3. **Context-Aware Generation:**  
   Retrieved text is passed to TinyLlama 1.1B for final answer generation or summarization.

---

## Technical Highlights

- **Language Model:** TinyLlama 1.1B (optimized for CPU and GPU)
- **Embedding Model:** Sentence Transformers (e.g., all-MiniLM-L6-v2)
- **Frameworks:** FastAPI, Uvicorn, PyTorch, Transformers
- **Indexing:** FAISS or in-memory vector search for speed

---

## Example Applications

- Local academic research assistant  
- Secure enterprise knowledge retrieval  
- Offline Q&A over proprietary documentation  
- Lightweight chatbot for internal data  

---

## Future Enhancements

- Web-based front-end interface (Streamlit or React)  
- Multi-format document ingestion (PDF, DOCX)  
- Caching and incremental vector updates  
- Optional local fine-tuning of TinyLlama  

---

## Maintainer

**Sri Harsha Mudumba**  
M.S. Computer Engineering, Iowa State University  
Focus: AI Inference Optimization, RAG Systems, and Edge Intelligence  

---

**LEXA — a compact and private Retrieval-Augmented Generation system designed for local deployment and research exploration.**
