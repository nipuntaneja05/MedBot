import os
import torch
from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from langchain_pinecone import PineconeVectorStore
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from pinecone import Pinecone  # Correct import

# Initialize FastAPI
app = FastAPI()

# Ensure GPU usage
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load embedding model and move to GPU
embedding_model = SentenceTransformer("all-MiniLM-L6-v2", device="cuda")

# Pinecone setup (Correct way)
pc = Pinecone(api_key="YOUR_PINECONE_API_KEY")

index_name = "medbot"
vector_store = PineconeVectorStore(index_name=index_name, embedding=embedding_model.encode)

# LLM (Groq API)
groq_llm = ChatGroq(model_name="mixtral-8x7b-32768", api_key="YOUR_GROQ_API_KEY")

# Retrieval Chain
retriever = vector_store.as_retriever()
qa_chain = RetrievalQA.from_chain_type(llm=groq_llm, retriever=retriever)

# Request Model
class QueryRequest(BaseModel):
    question: str

# Define API Endpoint
@app.post("/query")
async def query_model(request: QueryRequest):
    response = qa_chain.run(request.question)
    return {"answer": response}
