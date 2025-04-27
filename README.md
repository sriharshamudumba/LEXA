# LEXA
LEXA: Layer-Efficient eXit Agent with RAG implementation




## Overview

LEXA is an advanced Retrieval-Augmented Generation (RAG) system optimized with Early Exit logic inside the LLM transformer architecture. 
It retrieves documents based on user queries, generates high-quality responses using an LLM, and dynamically decides when to exit the model early to save computation and memory.

---

## Key Features

- Retrieval-Augmented Generation (RAG) with FAISS and Sentence Transformers
- Early Exit at transformer layers based on entropy/confidence
- Support for code generation, question answering, and document clarification tasks
- Memory-optimized retrieval with FAISS mmap indexing
- Quantized LLM inference (4-bit) for low-memory operation
- Modular FastAPI backend and Next.js frontend
- Deployable locally or to cloud services

---

## System Architecture

User Input → [FAISS Retriever] → Retrieved Context → [LEXA Early-Exit LLM] → Response

---

## Project Structure

```
LEXA/
├── backend/
│   ├── app.py              # FastAPI server
│   ├── retrieval.py        # FAISS document retriever
│   ├── generation.py       # LLM generation using OpenAI or local model
│   ├── exit_transformer.py # Early-exit transformer model (later phase)
│   ├── utils.py            # Helper functions
├── frontend/
│   ├── pages/
│   │   └── index.js        # Main chat page (Next.js)
│   ├── components/
│   │   └── ChatBox.js      # Chat UI
│   ├── package.json
├── data/
│   └── documents/          # Text files or data to be indexed
├── models/
│   └── sentence_transformer/ # Local embedding models
├── requirements.txt
├── README.md
├── .env
└── .gitignore
```

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/LEXA.git
cd LEXA
```

### 2. Backend Setup (FastAPI)

Install backend dependencies:

```bash
pip install -r requirements.txt
```

Create a `.env` file:

```
OPENAI_API_KEY=your-openai-api-key-here
```

Run FastAPI backend:

```bash
cd backend
uvicorn app:app --reload --port 8000
```

### 3. Frontend Setup (Next.js)

Install frontend dependencies:

```bash
cd frontend
npm install
npm run dev
```

Frontend will be available at `http://localhost:3000`.

---

## Usage Flow

1. User enters a query via frontend
2. FastAPI server retrieves top-k documents using FAISS
3. LEXA (with early exit optimization) generates an answer using the context
4. Answer is displayed back in the frontend

---

## Implementation Details

### RAG System

- SentenceTransformer `MiniLM-L6` or smaller model for embedding
- FAISS with memory-mapped index for efficient retrieval
- Top-2 document retrieval with optional early-exit based on retrieval confidence

### Early-Exit LLM (LEXA Core)

- Modified HuggingFace transformer model
- Classifiers added at intermediate layers
- Monitor entropy of output logits
- Dynamically exit if confidence threshold is reached during forward pass
- Quantized weights (4-bit) to reduce model memory

### Frontend

- Next.js powered chat interface
- Axios to communicate with FastAPI backend
- Dynamic loading indicators and user-friendly design

---

## Future Improvements

- Reinforcement Learning to adjust early-exit thresholds
- Streamlit dashboard for monitoring inference efficiency
- Support for additional tasks like translation, summarization
- Dockerized backend + frontend for deployment
- Advanced Retrieval Re-ranking (ColBERT, cross-encoder)

---

## Benchmarks (Planned)

| Metric | Baseline | After Early Exit |
|:---|:---|:---|
| Avg Latency (ms) | 800 | 450 |
| Avg Memory Usage (MB) | 3200 | 1400 |
| Accuracy Drop | N/A | <2% |

---


## Acknowledgements

- Hugging Face (Transformers, Sentence Transformers)
- OpenAI (GPT-3.5-turbo API)
- Facebook AI Research (FAISS)
- Vercel (Next.js Hosting)
- FastAPI Team

