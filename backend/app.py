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
    context = retrieve_documents(request.question)

    # Step 2: Generate an answer using the LLM and the context
    answer = generate_response(request.task, request.question, context)

    return {"answer": answer}
