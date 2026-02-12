"""
FastAPI Application — Mysoft AI Chatbot.

Endpoints:
  POST /chat             — Send a question, get a RAG-powered answer.
  POST /rebuild-index    — Rebuild the FAISS index for a company.
  GET  /health           — Health check.
  GET  /                 — Serve frontend (redirects to static files).
"""

import logging
import uuid
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from typing import Optional, List

from config import ALLOWED_ORIGINS, DEFAULT_COMPANY_ID
from rag import answer_query, initialize_company_index

# ── Logging Setup ────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(levelname)-7s │ %(name)s │ %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ── Lifespan (startup/shutdown) ──────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize the default company index on startup."""
    logger.info("━━━ Starting Mysoft AI Chatbot ━━━")
    success = initialize_company_index(DEFAULT_COMPANY_ID)
    if success:
        logger.info("Default company index ready: '%s'", DEFAULT_COMPANY_ID)
    else:
        logger.warning(
            "Could not initialize index for '%s'. "
            "Add documents to data/%s/ and call POST /rebuild-index.",
            DEFAULT_COMPANY_ID, DEFAULT_COMPANY_ID,
        )
    yield
    logger.info("━━━ Shutting down Mysoft AI Chatbot ━━━")


# ── App ──────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Mysoft AI Chatbot",
    description="RAG-powered chatbot for Mysoft Heaven (BD) Ltd.",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request / Response Models ────────────────────────────────────────────

class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000, description="User question")
    company_id: str = Field(default=DEFAULT_COMPANY_ID, description="Company identifier")
    session_id: Optional[str] = Field(default=None, description="Session ID for conversation memory")


class SourceInfo(BaseModel):
    filename: str
    chunk_index: int
    score: float


class ChatResponse(BaseModel):
    answer: str
    confidence: str
    sources: List[SourceInfo]
    fallback: bool
    session_id: str


class RebuildRequest(BaseModel):
    company_id: str = Field(default=DEFAULT_COMPANY_ID, description="Company identifier")


class RebuildResponse(BaseModel):
    status: str
    company_id: str
    message: str


# ── Endpoints ────────────────────────────────────────────────────────────

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Main chat endpoint. Sends a question through the RAG pipeline
    and returns an answer grounded in company documents.
    """
    session_id = request.session_id or str(uuid.uuid4())

    logger.info(
        "Chat request — company='%s', session='%s', question='%s'",
        request.company_id, session_id, request.question[:80],
    )

    result = answer_query(
        query=request.question,
        company_id=request.company_id,
        session_id=session_id,
    )

    return ChatResponse(
        answer=result["answer"],
        confidence=result["confidence"],
        sources=result["sources"],
        fallback=result["fallback"],
        session_id=session_id,
    )


@app.post("/rebuild-index", response_model=RebuildResponse)
async def rebuild_index(request: RebuildRequest):
    """Rebuild the FAISS index for a specific company from its documents."""
    logger.info("Rebuilding index for company: '%s'", request.company_id)

    success = initialize_company_index(request.company_id, force_rebuild=True)

    if success:
        return RebuildResponse(
            status="success",
            company_id=request.company_id,
            message=f"Index rebuilt successfully for '{request.company_id}'.",
        )
    else:
        raise HTTPException(
            status_code=404,
            detail=f"No documents found for company '{request.company_id}'. "
                   f"Add documents to data/{request.company_id}/ directory.",
        )


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy", "service": "Mysoft AI Chatbot"}


# ── Serve Frontend ───────────────────────────────────────────────────────
FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"

if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")

    @app.get("/")
    async def serve_frontend():
        """Serve the frontend index.html."""
        index_file = FRONTEND_DIR / "index.html"
        if index_file.exists():
            return FileResponse(str(index_file))
        return {"message": "Frontend not found. Place index.html in ../frontend/"}
