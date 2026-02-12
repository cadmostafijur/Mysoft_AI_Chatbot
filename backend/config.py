"""
Configuration module for the Mysoft AI Chatbot RAG system.
Centralizes all settings and supports multi-company scalability.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ── Paths ────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

# ── Chunking Strategy ────────────────────────────────────────────────────
CHUNK_SIZE = 400          # characters per chunk (300-500 range)
CHUNK_OVERLAP = 80        # overlap between chunks (50-100 range)

# ── Embedding Model ─────────────────────────────────────────────────────
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# ── FAISS ────────────────────────────────────────────────────────────────
FAISS_INDEX_DIR = BASE_DIR / "faiss_indexes"

# ── Retrieval ────────────────────────────────────────────────────────────
TOP_K = 5                           # number of chunks to retrieve
SIMILARITY_THRESHOLD = 0.25         # minimum cosine similarity score
HIGH_CONFIDENCE_THRESHOLD = 0.55    # above this → high confidence answer

# ── LLM Provider ─────────────────────────────────────────────────────────
# Supported: "openai", "gemini", "local"
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "gemini")

# ── LLM Reliability ──────────────────────────────────────────────────────
# If the LLM provider is rate-limited (429) or unavailable, we can still return
# a strict, document-grounded extractive answer from the retrieved chunks.
ALLOW_EXTRACTIVE_FALLBACK_ON_LLM_ERROR = True
EXTRACTIVE_FALLBACK_MAX_SNIPPETS = 3

# OpenAI
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")

# Google Gemini
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")

# Local / Ollama
LOCAL_MODEL_URL = os.getenv("LOCAL_MODEL_URL", "http://localhost:11434/api/generate")
LOCAL_MODEL_NAME = os.getenv("LOCAL_MODEL_NAME", "mistral")

# ── Conversation Memory ─────────────────────────────────────────────────
MAX_CONVERSATION_TURNS = 5          # remember last N turns

# ── Default Company ──────────────────────────────────────────────────────
DEFAULT_COMPANY_ID = "mysoft_heaven"

# ── CORS ─────────────────────────────────────────────────────────────────
ALLOWED_ORIGINS = ["*"]

# ── Fallback Message ─────────────────────────────────────────────────────
FALLBACK_MESSAGE = (
    "I'm unable to answer that as it's outside "
    "Mysoft Heaven (BD) Ltd.'s provided information. "
    "Please ask questions related to our company, services, or products."
)

# ── System Prompt ────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are an AI assistant for {company_name}. You MUST follow these rules strictly:

1. Answer questions ONLY using the provided context below. Do NOT use any external or prior knowledge.
2. If the context does not contain enough information to answer, say exactly: "This information is not found in the provided documents."
3. Do NOT hallucinate, guess, or make up any information.
4. Be professional, helpful, and concise.
5. If the user greets you, respond politely and briefly introduce yourself as the AI assistant for {company_name}.
6. Format your answers clearly. Use bullet points or numbered lists when appropriate.

CONTEXT:
{context}

CONVERSATION HISTORY:
{history}
"""
