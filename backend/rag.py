"""
RAG (Retrieval-Augmented Generation) Pipeline Module.

Orchestrates the full flow:
  Query → Retrieve relevant chunks → Build prompt → Call LLM → Return answer.

GUARDRAILS:
━━━━━━━━━━
- Similarity threshold filtering: chunks below SIMILARITY_THRESHOLD are discarded.
- If NO chunks pass the threshold, a fallback message is returned — no LLM call.
- Confidence scoring: average similarity of retrieved chunks determines confidence.
- System prompt strictly forbids the LLM from using external knowledge.
"""

import logging
import os
from typing import List, Dict, Optional, Tuple

from config import (
    SIMILARITY_THRESHOLD,
    HIGH_CONFIDENCE_THRESHOLD,
    TOP_K,
    FALLBACK_MESSAGE,
    SYSTEM_PROMPT,
    LLM_PROVIDER,
    OPENAI_API_KEY,
    OPENAI_MODEL,
    GEMINI_API_KEY,
    GEMINI_MODEL,
    LOCAL_MODEL_URL,
    LOCAL_MODEL_NAME,
    MAX_CONVERSATION_TURNS,
    DEFAULT_COMPANY_ID,
    ALLOW_EXTRACTIVE_FALLBACK_ON_LLM_ERROR,
    EXTRACTIVE_FALLBACK_MAX_SNIPPETS,
)
from embeddings import get_company_index
from data_loader import load_and_chunk_documents

logger = logging.getLogger(__name__)


# ── Conversation Memory ──────────────────────────────────────────────────

class ConversationMemory:
    """Simple in-memory conversation buffer per session."""

    def __init__(self, max_turns: int = MAX_CONVERSATION_TURNS):
        self.max_turns = max_turns
        self.history: List[Dict[str, str]] = []

    def add(self, role: str, content: str) -> None:
        self.history.append({"role": role, "content": content})
        # Keep only last N turns (each turn = user + assistant = 2 entries)
        max_entries = self.max_turns * 2
        if len(self.history) > max_entries:
            self.history = self.history[-max_entries:]

    def format_history(self) -> str:
        if not self.history:
            return "(No prior conversation)"
        lines = []
        for msg in self.history:
            prefix = "User" if msg["role"] == "user" else "Assistant"
            lines.append(f"{prefix}: {msg['content']}")
        return "\n".join(lines)

    def clear(self) -> None:
        self.history.clear()


# ── Session store (keyed by session_id) ─────────────────────────────────
_sessions: Dict[str, ConversationMemory] = {}


def get_memory(session_id: str) -> ConversationMemory:
    if session_id not in _sessions:
        _sessions[session_id] = ConversationMemory()
    return _sessions[session_id]


# ── LLM Callers ─────────────────────────────────────────────────────────

def call_openai(prompt: str, system: str) -> str:
    """Call OpenAI Chat Completions API."""
    try:
        from openai import OpenAI
        client = OpenAI(api_key=OPENAI_API_KEY)
        response = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
            max_tokens=1024,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logger.error("OpenAI API error: %s", e)
        return f"Error communicating with OpenAI: {e}"


def call_gemini(prompt: str, system: str) -> str:
    """Call Google Gemini API."""
    try:
        import google.generativeai as genai
        genai.configure(api_key=GEMINI_API_KEY)
        model = genai.GenerativeModel(
            GEMINI_MODEL,
            system_instruction=system,
        )
        response = model.generate_content(
            prompt,
            generation_config=genai.GenerationConfig(
                temperature=0.2,
                max_output_tokens=1024,
            ),
        )
        return response.text.strip()
    except Exception as e:
        logger.error("Gemini API error: %s", e)
        return f"Error communicating with Gemini: {e}"


def call_local_llm(prompt: str, system: str) -> str:
    """Call a local LLM (Ollama-compatible API)."""
    try:
        import requests
        payload = {
            "model": LOCAL_MODEL_NAME,
            "prompt": prompt,
            "system": system,
            "stream": False,
            "options": {"temperature": 0.2},
        }
        resp = requests.post(LOCAL_MODEL_URL, json=payload, timeout=120)
        resp.raise_for_status()
        return resp.json().get("response", "").strip()
    except Exception as e:
        logger.error("Local LLM error: %s", e)
        return f"Error communicating with local LLM: {e}"


def call_llm(prompt: str, system: str) -> str:
    """Route to the configured LLM provider."""
    provider = LLM_PROVIDER.lower()
    if provider == "openai":
        return call_openai(prompt, system)
    elif provider == "gemini":
        return call_gemini(prompt, system)
    elif provider == "local":
        return call_local_llm(prompt, system)
    else:
        return f"Unknown LLM provider: {provider}. Set LLM_PROVIDER to openai, gemini, or local."


def _looks_like_llm_error(text: str) -> bool:
    """Detect our standardized LLM error strings."""
    if not text or not text.strip():
        return True
    prefixes = (
        "Error communicating with Gemini:",
        "Error communicating with OpenAI:",
        "Error communicating with local LLM:",
        "Unknown LLM provider:",
    )
    return text.strip().startswith(prefixes)


def _build_extractive_answer(
    filtered_chunks: List[Tuple[Dict[str, str], float]],
    max_snippets: int,
) -> str:
    """
    Produce a strictly document-grounded response without any LLM generation.
    Used when Gemini/OpenAI is unavailable or rate-limited.
    """
    snippets = filtered_chunks[: max(1, max_snippets)]
    lines: List[str] = [
        "The language model is temporarily unavailable (rate limit/quota).",
        "Here is the most relevant information found in the provided documents:",
        "",
    ]
    for chunk, score in snippets:
        text = (chunk.get("text") or "").strip()
        if len(text) > 500:
            text = text[:500].rstrip() + "..."
        lines.append(f"- Source: {chunk.get('filename','unknown')} (score={score:.3f})")
        lines.append(f"  {text}")
        lines.append("")
    return "\n".join(lines).strip()


# ── Index Initialization ────────────────────────────────────────────────

def initialize_company_index(company_id: str, force_rebuild: bool = False) -> bool:
    """
    Ensure the FAISS index for a company is loaded (or built from documents).
    Returns True if the index is ready.
    """
    idx = get_company_index(company_id)

    # Try loading from disk first
    if not force_rebuild and idx.load():
        return True

    # Build from documents
    logger.info("Building index for company '%s'...", company_id)
    chunks = load_and_chunk_documents(company_id)
    if not chunks:
        logger.error("No documents found for company '%s'.", company_id)
        return False

    idx.build_index(chunks)
    return True


# ── Core RAG Function ───────────────────────────────────────────────────

def answer_query(
    query: str,
    company_id: str = DEFAULT_COMPANY_ID,
    session_id: str = "default",
    company_name: str = "Mysoft Heaven (BD) Ltd.",
) -> Dict:
    """
    Full RAG pipeline:
      1. Retrieve relevant chunks from FAISS
      2. Apply similarity threshold guardrail
      3. Build context-aware prompt
      4. Call LLM
      5. Return structured response with confidence

    Returns:
      {
        "answer": str,
        "confidence": "high" | "low" | "none",
        "sources": [ {"filename": ..., "chunk_index": ..., "score": ...} ],
        "fallback": bool,
      }
    """
    idx = get_company_index(company_id)

    # Ensure index is loaded
    if idx.index is None or idx.index.ntotal == 0:
        if not initialize_company_index(company_id):
            return {
                "answer": "No documents are loaded for this company. Please add documents and rebuild the index.",
                "confidence": "none",
                "sources": [],
                "fallback": True,
            }

    # Step 1: Retrieve
    results = idx.search(query, top_k=TOP_K)

    # Step 2: Filter by similarity threshold
    filtered = [(chunk, score) for chunk, score in results if score >= SIMILARITY_THRESHOLD]

    if not filtered:
        logger.info("No chunks passed threshold (%.2f) for query: %s", SIMILARITY_THRESHOLD, query)
        memory = get_memory(session_id)
        memory.add("user", query)
        memory.add("assistant", FALLBACK_MESSAGE)
        return {
            "answer": FALLBACK_MESSAGE,
            "confidence": "none",
            "sources": [],
            "fallback": True,
        }

    # Step 3: Determine confidence
    avg_score = sum(s for _, s in filtered) / len(filtered)
    confidence = "high" if avg_score >= HIGH_CONFIDENCE_THRESHOLD else "low"

    # Build context from retrieved chunks
    context_parts = []
    sources = []
    for chunk, score in filtered:
        context_parts.append(chunk["text"])
        sources.append({
            "filename": chunk["filename"],
            "chunk_index": chunk.get("chunk_index", 0),
            "score": round(score, 4),
        })

    context = "\n\n---\n\n".join(context_parts)

    # Step 4: Build prompt with conversation history
    memory = get_memory(session_id)
    history_str = memory.format_history()

    system_prompt = SYSTEM_PROMPT.format(
        company_name=company_name,
        context=context,
        history=history_str,
    )

    # If low confidence, add extra caution to the prompt
    user_prompt = query
    if confidence == "low":
        user_prompt += (
            "\n\n[NOTE: The retrieved context may not be highly relevant. "
            "If you cannot find a clear answer in the context above, "
            "say so honestly rather than guessing.]"
        )

    # Step 5: Call LLM
    answer = call_llm(user_prompt, system_prompt)

    # If Gemini/OpenAI/local is quota-limited or unavailable, return an extractive answer (still grounded in docs)
    fallback_flag = False
    confidence_out = confidence
    if _looks_like_llm_error(answer) and ALLOW_EXTRACTIVE_FALLBACK_ON_LLM_ERROR:
        logger.warning("LLM unavailable; using extractive fallback. Raw error: %s", answer[:200])
        answer = _build_extractive_answer(filtered, max_snippets=EXTRACTIVE_FALLBACK_MAX_SNIPPETS)
        fallback_flag = True
        confidence_out = "low"

    # Update conversation memory
    memory.add("user", query)
    memory.add("assistant", answer)

    return {
        "answer": answer,
        "confidence": confidence_out,
        "sources": sources,
        "fallback": fallback_flag,
    }
