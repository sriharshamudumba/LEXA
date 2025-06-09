from fastapi import FastAPI
from pydantic import BaseModel
from backend.retrieval import retrieve_documents
from backend.generation import generate_response


app = FastAPI()

# Define what a query request should look like
class QueryRequest(BaseModel):
    task: str
    question: str

# Define an endpoint to handle user queries
@app.post("/ask")
def ask_query(request: QueryRequest):
    # Step 1: Retrieve documents based on the question
    retrieved_results = retrieve_documents(request.question)  # Now a list of (doc, score)

    EARLY_EXIT_THRESHOLD = 0.2

    # Check for early exit condition
    if retrieved_results and retrieved_results[0][1] < EARLY_EXIT_THRESHOLD:
        top_document_text = retrieved_results[0][0]
        return {"answer": top_document_text, "source": "retrieval (early exit)"}
    else:
        # Extract document texts to form context
        document_texts = [result[0] for result in retrieved_results]
        context = "\n\n".join(document_texts)

        # Step 2: Generate an answer using the LLM and the context
        answer = generate_response(request.task, request.question, context)

        return {"answer": answer, "source": "generation"}
