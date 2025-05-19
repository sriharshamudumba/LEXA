# LEXA: Lightweight Local RAG System

LEXA is a lightweight, fully local Retrieval-Augmented Generation (RAG) system using TinyLlama 1.1B and semantic retrieval with Sentence Transformers.

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/your-username/LEXA.git
cd LEXA
```

### 2. Set up the virtual environment

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 3. Add your documents

Place `.txt` files in the directory:

```
data/documents/
```

### 4. Start the API server

```bash
uvicorn backend.app:app --reload --port 8000
```

### 5. Test the endpoints

Go to your browser and open:

```
http://127.0.0.1:8000/docs
```

Use the `/ask` endpoint with a payload like:

```json
{
  "task": "question answering",
  "question": "What is deep learning?"
}
```
