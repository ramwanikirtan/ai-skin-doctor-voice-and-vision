"""Standalone retrieval: question -> top-k chunks with metadata + scores.

Embeddings are reused from processed/chunks.json (created by rag/ingest.py).
Only the short question is embedded per query — never the whole knowledge base.
"""

import os

from dotenv import load_dotenv
from openai import OpenAI

from rag.ingest import EMBEDDING_MODEL
from rag.vector_store import load_store, search

load_dotenv()

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STORE_PATH = os.path.join(PROJECT_ROOT, "knowledge_base", "processed", "chunks.json")

# Cached singletons: reused across requests (no behavior change).
_client = None
_chunks_cache = None


def _get_client():
    """Return a shared OpenAI client instead of creating one per request."""
    global _client
    if _client is None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY is not set (needed to embed the question).")
        _client = OpenAI(api_key=api_key)
    return _client


def _get_chunks():
    """Load chunks.json once and reuse it instead of parsing it per request."""
    global _chunks_cache
    if _chunks_cache is None:
        _chunks_cache = load_store(STORE_PATH)
    return _chunks_cache


def retrieve(question, k=3):
    """Embed one question, return top-k chunk dicts (text + metadata + score)."""
    if not question or not question.strip():
        raise ValueError("retrieve() needs a non-empty question.")

    client = _get_client()
    response = client.embeddings.create(
        model=EMBEDDING_MODEL, input=question.strip()
    )
    query_embedding = response.data[0].embedding

    chunks = _get_chunks()
    return search(query_embedding, chunks, k=k)


if __name__ == "__main__":
    demo_questions = [
        "Itchy red patch on my forearm, what could it be?",
        "Spreading rash with pus and fever, should I see a doctor urgently?",
    ]
    for question in demo_questions:
        print(f"\nQ: {question}")
        for hit in retrieve(question, k=3):
            print(
                f"  [{hit['score']:.3f}] {hit['document_id']} / {hit['section']} "
                f"({hit['source_org']})"
            )
