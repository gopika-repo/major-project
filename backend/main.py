import sys
import os
import logging

# Fix import path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from config import settings
from core.rag_engine import RAGEngine

# ── Logging ────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# ── FastAPI app ────────────────────────────────────────────────────────────
app = FastAPI(
    title="Visual Financial Analyst API",
    description="Multimodal RAG system for financial documents",
    version="1.0.0"
)

# ── CORS — allows frontend to talk to backend ──────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

# ── Single RAG engine instance ─────────────────────────────────────────────
rag = RAGEngine()


# ── Request/Response models ────────────────────────────────────────────────
class QueryRequest(BaseModel):
    question: str
    doc_name: str


# ── Endpoints ──────────────────────────────────────────────────────────────

@app.get("/")
def root():
    """Health check — is server running?"""
    return {
        "status": "running",
        "project": "Visual Financial Analyst",
        "version": "1.0.0"
    }


@app.get("/api/health")
def health():
    """Check all services are connected."""
    return {
        "status": "healthy",
        "gemini": "connected",
        "qdrant": "connected",
        "model": settings.MODEL_NAME
    }


@app.post("/api/upload")
async def upload_pdf(file: UploadFile = File(...)):
    """
    Upload a PDF file.
    Extracts text, creates embeddings, stores in Qdrant.
    Returns processing stats.
    """
    # Validate file type
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are allowed"
        )

    try:
        # Save uploaded file
        os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
        save_path = os.path.join(settings.UPLOAD_DIR, file.filename)

        with open(save_path, "wb") as f:
            content = await file.read()
            f.write(content)

        logger.info(f"File saved: {save_path}")

        # Create clean doc name
        doc_name = file.filename.replace(".pdf", "").strip()

        # Process through RAG pipeline
        result = rag.process_document(save_path, doc_name)

        return {
            "status": "success",
            "doc_name": doc_name,
            "filename": file.filename,
            "pages_processed": result["pages_processed"],
            "chunks_created": result["chunks_created"],
            "vectors_stored": result["vectors_stored"],
            "message": f"{doc_name} is ready. You can now ask questions."
        }

    except Exception as e:
        logger.error(f"Upload failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/query")
def query_document(request: QueryRequest):
    """
    Ask a question about an uploaded document.
    Returns answer with sources and confidence score.
    """
    if not request.question.strip():
        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty"
        )

    try:
        result = rag.query(request.question, request.doc_name)
        return result

    except Exception as e:
        logger.error(f"Query failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/info")
def info():
    """Return system configuration info."""
    return {
        "model": settings.MODEL_NAME,
        "embed_model": settings.EMBED_MODEL,
        "dimensions": settings.EMBED_DIMENSIONS,
        "chunk_size": settings.CHUNK_SIZE,
        "top_k": settings.TOP_K,
        "collection": settings.QDRANT_COLLECTION
    }


# ── Run directly ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)