import os
import torch
from fastapi import FastAPI
from pydantic import BaseModel
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from pinecone import Pinecone

# Initialize FastAPI
app = FastAPI()

# Ensure GPU usage
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load embedding model correctly
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Set Pinecone API Key
PINECONE_API_KEY = "pcsk_6hfbZM_PyDhaPVXZs1H5ddoLnsEWx7iNai3SZR8Mfbgsf59yEKteEmXsiatUEu5q2RPML2"
if not PINECONE_API_KEY:
    raise ValueError("Pinecone API Key is missing. Set it in the environment or in the script.")

# Initialize Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)

index_name = "medbot"

# Check if Pinecone index exists (Fix: Extract index names correctly)
existing_indexes = [index["name"] for index in pc.list_indexes()]
if index_name not in existing_indexes:
    raise ValueError(f"Pinecone index '{index_name}' does not exist. Create it in Pinecone.")

# Initialize Pinecone Vector Store
vector_store = PineconeVectorStore(
    index_name=index_name,
    embedding=embedding_model,
    pinecone_api_key=PINECONE_API_KEY
)

# Initialize Groq LLM
GROQ_API_KEY = "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"
groq_llm = ChatGroq(model_name="mixtral-8x7b-32768", api_key=GROQ_API_KEY)

# Retrieval Chain
retriever = vector_store.as_retriever()
qa_chain = RetrievalQA.from_chain_type(llm=groq_llm, retriever=retriever)

# Request Model
class QueryRequest(BaseModel):
    question: str

# Define API Endpoint
@app.post("/query")
async def query_model(request: QueryRequest):
    try:
        response = qa_chain.invoke({"query": request.question})  # Fix: Use .invoke()
        return {"answer": response}
    except Exception as e:
        return {"error": str(e)}

# Run FastAPI
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000, reload=True)
