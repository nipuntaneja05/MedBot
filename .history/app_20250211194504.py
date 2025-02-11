import os
import torch
from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from langchain_pinecone import PineconeVectorStore
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from pinecone import Pinecone  

# ✅ Load API keys securely from environment variables
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "ypcsk_6hfbZM_PyDhaPVXZs1H5ddoLnsEWx7iNai3SZR8Mfbgsf59yEKteEmXsiatUEu5q2RPML2")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "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")
INDEX_NAME = "medbot"

# ✅ Initialize FastAPI
app = FastAPI()

# ✅ Ensure GPU usage
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ✅ Load embedding model on GPU
embedding_model = SentenceTransformer("all-MiniLM-L6-v2", device=device)

# ✅ Initialize Pinecone correctly
pc = Pinecone(api_key=PINECONE_API_KEY)

# ✅ Define embedding function explicitly
def embed_function(texts):
    return embedding_model.encode(texts, convert_to_numpy=True).tolist()

# ✅ Setup Pinecone Vector Store
vector_store = PineconeVectorStore(index_name=INDEX_NAME, embedding=embed_function)

# ✅ Initialize LLM (Groq API)
groq_llm = ChatGroq(model_name="mixtral-8x7b-32768", api_key=GROQ_API_KEY)

# ✅ Define RetrievalQA Chain
retriever = vector_store.as_retriever()
qa_chain = RetrievalQA.from_chain_type(llm=groq_llm, retriever=retriever)

# ✅ Define Request Model
class QueryRequest(BaseModel):
    question: str

# ✅ Define API Endpoint
@app.post("/query")
async def query_model(request: QueryRequest):
    response = qa_chain.run(request.question)
    return {"answer": response}
