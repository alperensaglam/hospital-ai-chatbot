"""
ChromaDB vector store management.

Handles collection creation, upserting chunks with embeddings,
and low-level query access.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import chromadb
from dotenv import load_dotenv

from rag.chunking import Chunk
from rag.embeddings import get_embeddings

load_dotenv()

CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "data/chroma")
COLLECTION_NAME = "hospital_docs"


def get_chroma_client() -> chromadb.ClientAPI:
    """Return a persistent ChromaDB client."""
    persist_path = Path(CHROMA_PERSIST_DIR)
    persist_path.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(persist_path))


def get_or_create_collection(
    client: chromadb.ClientAPI | None = None,
    collection_name: str = COLLECTION_NAME,
) -> chromadb.Collection:
    """Get or create the main document collection."""
    client = client or get_chroma_client()
    return client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )


def upsert_chunks(
    chunks: list[Chunk],
    collection: chromadb.Collection | None = None,
    batch_size: int = 50,
) -> int:
    """
    Embed and upsert chunks into ChromaDB.

    Returns the number of chunks upserted.
    """
    collection = collection or get_or_create_collection()

    total = 0
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        texts = [c.text for c in batch]
        ids = [c.metadata["chunk_id"] for c in batch]
        metadatas = [c.metadata for c in batch]

        embeddings = get_embeddings(texts)

        collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
        )
        total += len(batch)

    return total


def query_collection(
    query_embedding: list[float],
    *,
    top_k: int = 5,
    where: dict[str, Any] | None = None,
    collection: chromadb.Collection | None = None,
) -> dict:
    """
    Query the collection by embedding vector.

    Returns the raw ChromaDB query result dict with keys:
      ids, embeddings, documents, metadatas, distances
    """
    collection = collection or get_or_create_collection()
    kwargs: dict[str, Any] = {
        "query_embeddings": [query_embedding],
        "n_results": top_k,
        "include": ["documents", "metadatas", "distances"],
    }
    if where:
        kwargs["where"] = where
    return collection.query(**kwargs)


def collection_count(collection: chromadb.Collection | None = None) -> int:
    """Return the number of items in the collection."""
    collection = collection or get_or_create_collection()
    return collection.count()


def delete_collection(
    collection_name: str = COLLECTION_NAME,
    client: chromadb.ClientAPI | None = None,
) -> None:
    """Delete a collection entirely (useful for re-ingestion)."""
    client = client or get_chroma_client()
    try:
        client.delete_collection(collection_name)
    except chromadb.errors.NotFoundError:
        pass  # Collection doesn't exist
