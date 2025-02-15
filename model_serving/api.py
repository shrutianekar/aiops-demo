import logging
from fastapi import FastAPI, HTTPException, Query
from prometheus_fastapi_instrumentator import Instrumentator
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import chromadb
from loguru import logger

# Initialize FastAPI app
app = FastAPI(
    title="AI Model Serving",
    description="Serving AI model and vector-based retrieval using ChromaDB",
    version="2.0.0"
)
# Enable Prometheus metrics
Instrumentator().instrument(app).expose(app, endpoint="/metrics")

# Configure logging
logger.add("model_serving.log", rotation="10MB", level="INFO")

# Load the embedding model
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
try:
    logger.info(f"Loading embedding model: {MODEL_NAME}...")
    embedding_model = SentenceTransformer(MODEL_NAME)
    logger.info("Model loaded successfully!")
except Exception as e:
    logger.error(f"Failed to load model: {str(e)}")
    raise RuntimeError(f"Failed to load model: {str(e)}")

# Connect to ChromaDB
CHROMA_DB_PATH = "./chroma_db"
try:
    logger.info(f"Connecting to ChromaDB at {CHROMA_DB_PATH}...")
    chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    collection = chroma_client.get_or_create_collection(name="documents")
    logger.info("ChromaDB initialized successfully!")
except Exception as e:
    logger.error(f"ChromaDB initialization failed: {str(e)}")
    raise RuntimeError(f"ChromaDB initialization failed: {str(e)}")

# Request model for querying
class QueryRequest(BaseModel):
    query: str
    top_k: int = 5

@app.get("/health", tags=["Health"])
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "model": MODEL_NAME}

@app.post("/retrieve", tags=["Retrieval"])
async def retrieve_documents(request: QueryRequest):
    """Retrieves most relevant documents from ChromaDB based on query"""
    if not request.query:
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    try:
        # Generate embedding for the query
        query_embedding = embedding_model.encode(request.query).tolist()

        # Retrieve similar documents from ChromaDB
        results = collection.query(query_embeddings=[query_embedding], n_results=request.top_k)

        return {
            "query": request.query,
            "results": results.get("documents", []),
            "metadata": results.get("metadatas", [])
        }
    except Exception as e:
        logger.error(f"Query failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))