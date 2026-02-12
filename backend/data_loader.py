"""
Document Loading & Chunking Module.

Handles:
  - Loading .txt and .pdf files from per-company data directories.
  - Splitting documents into overlapping chunks for optimal retrieval.

CHUNKING STRATEGY EXPLANATION:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
We use overlapping fixed-size character chunks (300-500 chars, 50-100 overlap):

  ┌────── Chunk 1 ──────┐
  │  ...text content...  │
  │         ┌────── Chunk 2 ──────┐
  │         │  overlap   │        │
  └─────────│────────────┘        │
            │   ...text content...│
            └─────────────────────┘

WHY overlapping chunks?
  1. Prevents information loss at chunk boundaries — if a sentence is split
     between two chunks, the overlap ensures both chunks capture the full idea.
  2. Improves retrieval recall — a query may match context that spans two
     consecutive chunks; overlap keeps that context in at least one chunk.
  3. 400-char chunks (with 80-char overlap) balance granularity vs. context:
     - Too small → fragments lose meaning.
     - Too large → dilutes relevance signal in embedding space.
"""

import os
import logging
from pathlib import Path
from typing import List, Dict

from config import CHUNK_SIZE, CHUNK_OVERLAP, DATA_DIR

logger = logging.getLogger(__name__)


# ── PDF text extraction ──────────────────────────────────────────────────
def extract_text_from_pdf(filepath: str) -> str:
    """Extract text from a PDF file using PyPDF2."""
    try:
        from PyPDF2 import PdfReader
        reader = PdfReader(filepath)
        text_parts = []
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text_parts.append(page_text)
        return "\n".join(text_parts)
    except ImportError:
        logger.warning("PyPDF2 not installed. Skipping PDF: %s", filepath)
        return ""
    except Exception as e:
        logger.error("Error reading PDF %s: %s", filepath, e)
        return ""


# ── Load documents for a company ─────────────────────────────────────────
def load_documents(company_id: str) -> List[Dict[str, str]]:
    """
    Load all .txt and .pdf documents from  data/<company_id>/  directory.

    Returns a list of dicts:  [{"filename": "...", "content": "..."}, ...]
    """
    company_dir = DATA_DIR / company_id
    if not company_dir.exists():
        logger.error("Data directory not found: %s", company_dir)
        return []

    documents: List[Dict[str, str]] = []

    for filepath in sorted(company_dir.iterdir()):
        if filepath.suffix.lower() == ".txt":
            content = filepath.read_text(encoding="utf-8", errors="ignore")
            documents.append({"filename": filepath.name, "content": content})
            logger.info("Loaded TXT: %s (%d chars)", filepath.name, len(content))

        elif filepath.suffix.lower() == ".pdf":
            content = extract_text_from_pdf(str(filepath))
            if content:
                documents.append({"filename": filepath.name, "content": content})
                logger.info("Loaded PDF: %s (%d chars)", filepath.name, len(content))

    logger.info("Total documents loaded for '%s': %d", company_id, len(documents))
    return documents


# ── Chunk documents ──────────────────────────────────────────────────────
def chunk_text(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> List[str]:
    """
    Split text into overlapping chunks of approximately `chunk_size` characters.
    Tries to break at sentence boundaries when possible.
    """
    if not text or not text.strip():
        return []

    # Normalize whitespace
    text = " ".join(text.split())
    chunks: List[str] = []
    start = 0

    while start < len(text):
        end = start + chunk_size

        # If we're not at the end of text, try to break at a sentence boundary
        if end < len(text):
            # Look for sentence-ending punctuation near the end of the chunk
            best_break = -1
            search_start = max(start + chunk_size - 100, start)
            for i in range(min(end, len(text)) - 1, search_start - 1, -1):
                if text[i] in ".!?\n" and i + 1 < len(text) and text[i + 1] == " ":
                    best_break = i + 1
                    break

            if best_break > start:
                end = best_break

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        # Move start forward by (chunk_size - overlap)
        start += chunk_size - overlap
        if start >= end and end < len(text):
            start = end  # Avoid infinite loop

    return chunks


def load_and_chunk_documents(company_id: str) -> List[Dict[str, str]]:
    """
    Load all documents for a company, chunk them, and return a flat list of:
      [{"filename": "...", "chunk_index": 0, "text": "..."}, ...]
    """
    documents = load_documents(company_id)
    all_chunks: List[Dict[str, str]] = []

    for doc in documents:
        chunks = chunk_text(doc["content"])
        for i, chunk in enumerate(chunks):
            all_chunks.append({
                "filename": doc["filename"],
                "chunk_index": i,
                "text": chunk,
            })

    logger.info(
        "Company '%s': %d documents → %d chunks",
        company_id, len(documents), len(all_chunks),
    )
    return all_chunks
