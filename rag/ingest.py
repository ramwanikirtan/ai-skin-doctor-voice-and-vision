"""Standalone ingestion pipeline: documents/*.md -> processed/chunks.json.

Steps: load -> parse header metadata -> clean -> split by ## sections ->
word-chunk oversized sections -> embed once (OpenAI) -> store.

Run:  uv run python rag/ingest.py
Rerun only when documents change. Retrieval reuses stored embeddings.
"""

import os
import re
import sys

from dotenv import load_dotenv
from openai import OpenAI

from rag.vector_store import save_store

load_dotenv()

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOCS_DIR = os.path.join(PROJECT_ROOT, "knowledge_base", "documents")
OUTPUT_PATH = os.path.join(PROJECT_ROOT, "knowledge_base", "processed", "chunks.json")

EMBEDDING_MODEL = "text-embedding-3-small"
MAX_WORDS_PER_CHUNK = 400


def parse_document(path):
    """Split a markdown file into (metadata, sections). Metadata from --- header."""
    with open(path, "r", encoding="utf-8") as f:
        raw = f.read()

    metadata = {"document_id": os.path.splitext(os.path.basename(path))[0]}
    body = raw
    header_match = re.match(r"^---\s*\n(.*?)\n---\s*\n(.*)$", raw, re.DOTALL)
    if header_match:
        for line in header_match.group(1).splitlines():
            if ":" in line:
                key, value = line.split(":", 1)
                metadata[key.strip()] = value.strip()
        body = header_match.group(2)

    # Split on ## headings; skip the H1 title and any blockquote source note.
    sections = []
    current_heading, current_lines = None, []
    for line in body.splitlines():
        heading = re.match(r"^##\s+(.*)", line)
        if heading:
            if current_heading and current_lines:
                sections.append((current_heading, "\n".join(current_lines).strip()))
            current_heading, current_lines = heading.group(1).strip(), []
        elif current_heading is not None and not line.strip().startswith(">"):
            current_lines.append(line)
    if current_heading and current_lines:
        sections.append((current_heading, "\n".join(current_lines).strip()))
    return metadata, sections


def clean_text(text):
    """Collapse whitespace; drop empty fragments."""
    text = re.sub(r"\s+", " ", text).strip()
    return text


def chunk_section(text, max_words=MAX_WORDS_PER_CHUNK):
    """One chunk per section; word-split only if oversized (keeps meaning intact)."""
    words = text.split()
    if len(words) <= max_words:
        return [text]
    return [" ".join(words[i:i + max_words]) for i in range(0, len(words), max_words)]


def build_chunks():
    """Load all documents -> chunk dicts with metadata (no embeddings yet)."""
    chunks = []
    for filename in sorted(os.listdir(DOCS_DIR)):
        if not filename.endswith(".md"):
            continue
        metadata, sections = parse_document(os.path.join(DOCS_DIR, filename))
        for heading, section_text in sections:
            cleaned = clean_text(section_text)
            if not cleaned:
                continue
            for part in chunk_section(cleaned):
                chunks.append({
                    "document_id": metadata.get("document_id", ""),
                    "title": metadata.get("title", ""),
                    "source_org": metadata.get("source_org", ""),
                    "source_url": metadata.get("source_url", ""),
                    "section": heading,
                    "text": part,
                })
    return chunks


def embed_texts(client, texts):
    """One API call for all chunk texts -> list of embedding vectors."""
    response = client.embeddings.create(model=EMBEDDING_MODEL, input=texts)
    return [item.embedding for item in response.data]


def main():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY is not set (needed for embeddings).")

    chunks = build_chunks()
    if not chunks:
        raise ValueError(f"No sections found in {DOCS_DIR}.")
    print(f"Built {len(chunks)} chunks from {DOCS_DIR}")

    client = OpenAI(api_key=api_key)
    embeddings = embed_texts(client, [c["text"] for c in chunks])
    for chunk, embedding in zip(chunks, embeddings):
        chunk["embedding"] = embedding

    save_store(chunks, OUTPUT_PATH)
    print("Ingestion complete. Retrieval will reuse these stored embeddings.")


if __name__ == "__main__":
    sys.exit(main())
