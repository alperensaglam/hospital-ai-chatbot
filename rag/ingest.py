"""
Document ingestion pipeline.

Orchestrates: read raw docs → chunk → embed → upsert to ChromaDB.

Usage:
    python -m rag.ingest                   # ingest all docs in data/raw/
    python -m rag.ingest --reset           # delete collection first, then ingest
    python -m rag.ingest --raw-dir path/   # custom raw directory
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from rag.chunking import Chunk, chunk_all_documents
from rag.vector_store import (
    collection_count,
    delete_collection,
    get_or_create_collection,
    upsert_chunks,
)

DEFAULT_RAW_DIR = Path("data/raw")
PROCESSED_OUTPUT = Path("data/processed/chunks.jsonl")


def save_chunks_jsonl(chunks: list[Chunk], output_path: Path) -> None:
    """Persist chunks as JSONL for inspection and reproducibility."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for chunk in chunks:
            record = {"text": chunk.text, "metadata": chunk.metadata}
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"  ✓ Saved {len(chunks)} chunks to {output_path}")


def run_ingestion(raw_dir: Path = DEFAULT_RAW_DIR, *, reset: bool = False) -> int:
    """
    Full ingestion pipeline.

    Returns the number of chunks ingested.
    """
    print("=" * 60)
    print("  Hospital AI Chatbot — Document Ingestion Pipeline")
    print("=" * 60)

    # 1. Reset if requested
    if reset:
        print("\n⚠  Resetting collection...")
        delete_collection()

    # 2. Chunk documents
    md_files = sorted(raw_dir.glob("*.md"))
    print(f"\n📂 Found {len(md_files)} documents in {raw_dir}/")
    for f in md_files:
        print(f"   • {f.name}")

    chunks = chunk_all_documents(raw_dir)
    print(f"\n✂  Created {len(chunks)} chunks")

    # 3. Save JSONL
    save_chunks_jsonl(chunks, PROCESSED_OUTPUT)

    # 4. Embed and upsert
    print(f"\n🔄 Embedding and upserting to ChromaDB...")
    collection = get_or_create_collection()
    count = upsert_chunks(chunks, collection=collection)
    print(f"  ✓ Upserted {count} chunks")

    # 5. Verify
    total = collection_count(collection)
    print(f"\n✅ Collection now contains {total} chunks total")
    print("=" * 60)

    return count


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest hospital documents into ChromaDB")
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=DEFAULT_RAW_DIR,
        help="Directory containing raw .md documents",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Delete existing collection before ingesting",
    )
    args = parser.parse_args()

    if not args.raw_dir.exists():
        print(f"Error: {args.raw_dir} does not exist", file=sys.stderr)
        sys.exit(1)

    run_ingestion(args.raw_dir, reset=args.reset)


if __name__ == "__main__":
    main()
