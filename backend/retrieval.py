from sentence_transformers import SentenceTransformer
import os
import faiss
import numpy as np

class DocumentRetriever:
    def __init__(self, data_dir='data/documents', embedding_model_name='all-MiniLM-L6-v2'):
        self.data_dir = data_dir
        self.embedding_model = SentenceTransformer(embedding_model_name)
        self.documents = []
        self.embeddings = None
        self.index = None

    def load_documents(self):
        """
        Load all text documents from the data directory.
        """
        docs = []
        for filename in os.listdir(self.data_dir):
            if filename.endswith('.txt'):
                filepath = os.path.join(self.data_dir, filename)
                with open(filepath, 'r', encoding='utf-8') as file:
                    docs.append(file.read())
        self.documents = docs

    def embed_documents(self):
        """
        Create embeddings for all loaded documents.
        """
        self.embeddings = self.embedding_model.encode(self.documents, convert_to_numpy=True)

    def build_faiss_index(self):
        """
        Build a FAISS index from document embeddings.
        """
        dimension = self.embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)  # Using L2 distance
        self.index.add(self.embeddings)

    def retrieve(self, query: str, top_k: int = 2):
        """
        Retrieve top-k most relevant documents for a given query.
        """
        query_embedding = self.embedding_model.encode([query], convert_to_numpy=True)
        distances, indices = self.index.search(query_embedding, top_k)
        retrieved_docs = [self.documents[i] for i in indices[0]]
        return "\n\n".join(retrieved_docs)


# Initialize once at server start
retriever = DocumentRetriever()
retriever.load_documents()
retriever.embed_documents()
retriever.build_faiss_index()

# This function is imported in app.py
def retrieve_documents(query: str) -> str:
    return retriever.retrieve(query)
