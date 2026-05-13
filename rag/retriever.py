"""
Retriever module — semantic search + metadata filtering over ChromaDB.

Provides the main retrieve() function used by the agent pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from rag.embeddings import get_single_embedding
from rag.vector_store import get_or_create_collection, query_collection


@dataclass
class RetrievedChunk:
    """A retrieved chunk with its similarity score and metadata."""

    text: str
    score: float  # cosine distance (lower = more similar)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def source_id(self) -> str:
        return self.metadata.get("source_id", "unknown")

    @property
    def title(self) -> str:
        return self.metadata.get("title", "Unknown")

    @property
    def chunk_id(self) -> str:
        return self.metadata.get("chunk_id", "unknown")


def retrieve(
    query: str,
    *,
    top_k: int = 5,
    filters: dict[str, Any] | None = None,
) -> list[RetrievedChunk]:
    """
    Retrieve the top-k most relevant chunks for a query.

    Args:
        query: User query or rewritten retrieval query.
        top_k: Number of chunks to return.
        filters: Optional ChromaDB where-clause for metadata filtering.
                 Example: {"doc_type": "policy"} or
                          {"department": {"$in": ["cardiology", "general"]}}

    Returns:
        List of RetrievedChunk sorted by relevance (best first).
    """
    query_emb = get_single_embedding(query)
    collection = get_or_create_collection()
    results = query_collection(
        query_emb,
        top_k=top_k,
        where=filters,
        collection=collection,
    )

    chunks: list[RetrievedChunk] = []
    if results and results.get("documents"):
        docs = results["documents"][0]
        metas = results["metadatas"][0] if results.get("metadatas") else [{}] * len(docs)
        dists = results["distances"][0] if results.get("distances") else [0.0] * len(docs)

        for doc, meta, dist in zip(docs, metas, dists):
            chunks.append(RetrievedChunk(text=doc, score=dist, metadata=meta))

    return chunks


def format_context(chunks: list[RetrievedChunk]) -> str:
    """
    Format retrieved chunks into a context string for the LLM prompt.

    Each chunk is labelled with its source for citation.
    """
    if not chunks:
        return "No relevant documents were found."

    parts: list[str] = []
    for i, chunk in enumerate(chunks, 1):
        source = chunk.title
        section = chunk.metadata.get("section_heading", "")
        label = f"{source} — {section}" if section else source
        parts.append(f"[Source {i}: {label}]\n{chunk.text}")

    return "\n\n---\n\n".join(parts)
