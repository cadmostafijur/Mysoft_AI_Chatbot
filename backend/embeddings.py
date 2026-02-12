"""
Embedding & FAISS Index Module.

Handles:
  - Generating sentence-level embeddings via sentence-transformers.
  - Building, saving, and loading FAISS vector indexes (per company).
  - Similarity search with cosine distance scoring.

EMBEDDING MODEL CHOICE — all-MiniLM-L6-v2:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
We chose  all-MiniLM-L6-v2  for the following reasons:

  1. Compact & Fast — only 80 MB / 22M parameters. Runs efficiently on
     CPU, making it ideal for local deployments without a GPU.
  2. High Quality — trained on 1B+ sentence pairs; produces 384-dim
     embeddings that capture rich semantic meaning.
  3. Cosine Similarity Optimized — the model is specifically fine-tuned
     for cosine similarity, which pairs perfectly with FAISS IndexFlatIP
     (inner-product on L2-normalized vectors ≡ cosine similarity).
  4. Widely Adopted — battle-tested in production RAG systems; excellent
     balance of speed vs. accuracy for retrieval tasks.
  5. No API Dependency — runs 100% offline; no external API calls needed
     for embeddings, reducing latency and cost.
"""

import logging
import json
import numpy as np
import faiss
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from sentence_transformers import SentenceTransformer

from config import EMBEDDING_MODEL_NAME, FAISS_INDEX_DIR, TOP_K

logger = logging.getLogger(__name__)

# ── Global model cache ───────────────────────────────────────────────────
_model: Optional[SentenceTransformer] = None


def get_embedding_model() -> SentenceTransformer:
    """Lazy-load and cache the sentence-transformer model."""
    global _model
    if _model is None:
        logger.info("Loading embedding model: %s ...", EMBEDDING_MODEL_NAME)
        _model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        logger.info("Embedding model loaded. Dimension: %d", _model.get_sentence_embedding_dimension())
    return _model


def embed_texts(texts: List[str]) -> np.ndarray:
    """Embed a list of texts and return L2-normalized vectors."""
    model = get_embedding_model()
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
    # L2-normalize so inner product == cosine similarity
    faiss.normalize_L2(embeddings)
    return embeddings


# ── FAISS Index Management ───────────────────────────────────────────────

class CompanyIndex:
    """
    Manages the FAISS index and chunk metadata for a single company.
    Each company gets its own index — enabling multi-company scalability.
    """

    def __init__(self, company_id: str):
        self.company_id = company_id
        self.index: Optional[faiss.IndexFlatIP] = None
        self.chunks: List[Dict[str, str]] = []
        self._index_dir = Path(FAISS_INDEX_DIR) / company_id

    @property
    def index_path(self) -> Path:
        return self._index_dir / "index.faiss"

    @property
    def meta_path(self) -> Path:
        return self._index_dir / "chunks_meta.json"

    def build_index(self, chunks: List[Dict[str, str]]) -> None:
        """Build a FAISS index from a list of chunk dicts (must have 'text' key)."""
        if not chunks:
            logger.warning("No chunks to index for company '%s'.", self.company_id)
            return

        texts = [c["text"] for c in chunks]
        logger.info("Embedding %d chunks for '%s'...", len(texts), self.company_id)
        embeddings = embed_texts(texts)

        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)  # inner product (cosine on normalized vecs)
        self.index.add(embeddings)
        self.chunks = chunks

        logger.info(
            "FAISS index built for '%s': %d vectors, dim=%d",
            self.company_id, self.index.ntotal, dim,
        )
        self.save()

    def save(self) -> None:
        """Persist the FAISS index and metadata to disk."""
        self._index_dir.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(self.index_path))
        with open(self.meta_path, "w", encoding="utf-8") as f:
            json.dump(self.chunks, f, ensure_ascii=False, indent=2)
        logger.info("Index saved to %s", self._index_dir)

    def load(self) -> bool:
        """Load a previously saved FAISS index from disk. Returns True if successful."""
        if not self.index_path.exists() or not self.meta_path.exists():
            logger.info("No existing index found for '%s'.", self.company_id)
            return False

        self.index = faiss.read_index(str(self.index_path))
        with open(self.meta_path, "r", encoding="utf-8") as f:
            self.chunks = json.load(f)

        logger.info(
            "Loaded existing index for '%s': %d vectors",
            self.company_id, self.index.ntotal,
        )
        return True

    def search(self, query: str, top_k: int = TOP_K) -> List[Tuple[Dict[str, str], float]]:
        """
        Search the index for chunks most similar to the query.

        Returns list of (chunk_dict, similarity_score) tuples,
        sorted by descending similarity.
        """
        if self.index is None or self.index.ntotal == 0:
            logger.warning("Index is empty for '%s'.", self.company_id)
            return []

        query_vec = embed_texts([query])
        scores, indices = self.index.search(query_vec, min(top_k, self.index.ntotal))

        results: List[Tuple[Dict[str, str], float]] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            results.append((self.chunks[idx], float(score)))

        return results


# ── Index Registry (multi-company) ──────────────────────────────────────
_index_cache: Dict[str, CompanyIndex] = {}


def get_company_index(company_id: str) -> CompanyIndex:
    """Get or create a CompanyIndex instance (cached)."""
    if company_id not in _index_cache:
        _index_cache[company_id] = CompanyIndex(company_id)
    return _index_cache[company_id]
