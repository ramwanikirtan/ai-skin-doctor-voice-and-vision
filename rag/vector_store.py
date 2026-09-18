"""Local JSON vector store: save/load chunks + embeddings, cosine-similarity search.

Beginner-friendly and dependency-light (numpy only). The search interface
(search(query_embedding, k)) stays the same if we later swap in ChromaDB.
"""

import json
import os

import numpy as np


def save_store(chunks, path):
    """Write list of chunk dicts (with 'embedding') to JSON file."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=1)
    print(f"Saved {len(chunks)} chunks to {path}")


def load_store(path):
    """Load chunk dicts from JSON file. Raises FileNotFoundError if not ingested."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Vector store not found: {path}. Run rag/ingest.py first."
        )
    with open(path, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    print(f"Loaded {len(chunks)} chunks from {path}")
    return chunks


def search(query_embedding, chunks, k=3):
    """Return top-k chunks ranked by cosine similarity (embeddings reused, not regenerated)."""
    q = np.array(query_embedding, dtype=float)
    q_norm = np.linalg.norm(q)
    if q_norm == 0:
        raise ValueError("Query embedding is a zero vector.")

    scored = []
    for chunk in chunks:
        v = np.array(chunk["embedding"], dtype=float)
        denom = q_norm * np.linalg.norm(v)
        score = float(np.dot(q, v) / denom) if denom > 0 else 0.0
        scored.append((score, chunk))

    scored.sort(key=lambda item: item[0], reverse=True)
    return [{"score": score, **chunk} for score, chunk in scored[:k]]
