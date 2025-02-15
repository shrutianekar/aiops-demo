import logging
import uuid
import time
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field
from typing import List, Optional
from sentence_transformers import SentenceTransformer
import chromadb
from loguru import logger
from prometheus_fastapi_instrumentator import Instrumentator
from transformers import pipeline

# Initialize FastAPI app
app = FastAPI(
    title="ChromaDB Vector Service",
    description="A scalable service for vector-based similarity search using ChromaDB.",
    version="2.0.0"
)
# Enable Prometheus metrics
Instrumentator().instrument(app).expose(app, endpoint="/metrics")

# Configure logging
logger.add("chroma_service.log", rotation="10MB", level="INFO")

# Load the embedding model
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
try:
    logger.info(f"Loading embedding model: {MODEL_NAME}...")
    embedding_model = SentenceTransformer(MODEL_NAME)
    logger.info("Model loaded successfully!")
except Exception as e:
    logger.error(f"Failed to load model: {str(e)}")
    raise RuntimeError(f"Failed to load model: {str(e)}")

# Load text generation model for RAG
try:
    logger.info("Loading text generation model...")
    rag_generator = pipeline("text-generation", model="facebook/opt-1.3b")
    logger.info("Text generation model loaded successfully!")
except Exception as e:
    logger.error(f"Failed to load text generation model: {str(e)}")
    raise RuntimeError(f"Failed to load text generation model: {str(e)}")

# Initialize ChromaDB client
CHROMA_DB_PATH = "./chroma_db"
try:
    logger.info(f"Connecting to ChromaDB at {CHROMA_DB_PATH}...")
    chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    collection = chroma_client.get_or_create_collection(name="documents")
    logger.info("ChromaDB initialized successfully!")
except Exception as e:
    logger.error(f"ChromaDB initialization failed: {str(e)}")
    raise RuntimeError(f"ChromaDB initialization failed: {str(e)}")

# Define request models
class Document(BaseModel):
    text: str = Field(..., example="This is a sample document.")
    metadata: Optional[dict] = Field(default_factory=dict)

class BulkDocuments(BaseModel):
    documents: List[Document]

class QueryRequest(BaseModel):
    query: str = Field(..., example="What is AI?")
    top_k: int = Field(5, gt=0, le=50, example=5)

@app.get("/", tags=["Health"])
async def root():
    """Basic health check endpoint."""
    return {"message": "ChromaDB Vector Service is running!"}

@app.get("/health", tags=["Health"])
async def health_check():
    """Liveness & Readiness probe for Kubernetes."""
    return {"status": "healthy", "chroma_db_path": CHROMA_DB_PATH}

@app.post("/add_document/", tags=["Data Ingestion"])
async def add_document(doc: Document):
    """Ingests a single document into ChromaDB after encoding it into an embedding."""
    if not doc.text:
        raise HTTPException(status_code=400, detail="Text cannot be empty")

    try:
        embedding = embedding_model.encode(doc.text).tolist()
        doc_id = str(uuid.uuid4())
        timestamp = time.time()

        collection.add(
            ids=[doc_id],
            embeddings=[embedding],
            metadatas=[{**doc.metadata, "timestamp": timestamp}],
            documents=[doc.text]
        )
        logger.info(f"Added document {doc_id} to ChromaDB")
        return {"message": "Document added successfully", "doc_id": doc_id}
    except Exception as e:
        logger.error(f"Failed to add document: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/add_bulk_documents/", tags=["Data Ingestion"])
async def add_bulk_documents(docs: BulkDocuments):
    """Ingests multiple documents in batch mode for better efficiency."""
    if not docs.documents:
        raise HTTPException(status_code=400, detail="Document list cannot be empty")

    try:
        doc_texts = [doc.text for doc in docs.documents]
        doc_ids = [str(uuid.uuid4()) for _ in docs.documents]
        embeddings = embedding_model.encode(doc_texts).tolist()
        timestamps = [time.time()] * len(docs.documents)
        metadatas = [{**doc.metadata, "timestamp": ts} for doc, ts in zip(docs.documents, timestamps)]

        collection.add(
            ids=doc_ids,
            embeddings=embeddings,
            metadatas=metadatas,
            documents=doc_texts
        )
        logger.info(f"Added {len(doc_texts)} documents to ChromaDB")
        return {"message": "Documents added successfully", "doc_ids": doc_ids}
    except Exception as e:
        logger.error(f"Failed to add bulk documents: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query/", tags=["Search"])
async def query_documents(request: QueryRequest):
    """Performs similarity search using query embeddings."""
    if not request.query:
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    try:
        query_embedding = embedding_model.encode(request.query).tolist()
        results = collection.query(query_embeddings=[query_embedding], n_results=request.top_k)

        return {
            "query": request.query,
            "results": results.get("documents", [])
        }
    except Exception as e:
        logger.error(f"Query failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/query/", tags=["Search"])
async def query_documents_get(query: str = Query(..., description="Search text"), top_k: int = 5):
    """GET method for querying (same functionality as the POST endpoint)."""
    return await query_documents(QueryRequest(query=query, top_k=top_k))

@app.post("/generate_answer/", tags=["RAG"])
async def generate_answer(request: QueryRequest):
    """Retrieves relevant documents and generates an answer using a language model."""
    if not request.query:
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    try:
        # Retrieve similar documents from ChromaDB
        query_embedding = embedding_model.encode(request.query).tolist()
        results = collection.query(query_embeddings=[query_embedding], n_results=request.top_k)
        retrieved_docs = results.get("documents", [])
        
        if not retrieved_docs:
            return {"query": request.query, "answer": "No relevant information found."}

        # Combine retrieved documents as context
        context = "\n".join(retrieved_docs[:3])  # Limit to top 3 documents for context
        prompt = f"Context: {context}\n\nQuestion: {request.query}\nAnswer:"
        
        # Generate response
        generated_answer = rag_generator(prompt, max_length=200, num_return_sequences=1)[0]['generated_text']
        
        return {
            "query": request.query,
            "retrieved_documents": retrieved_docs,
            "generated_answer": generated_answer
        }
    except Exception as e:
        logger.error(f"RAG generation failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
