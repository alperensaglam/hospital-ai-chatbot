"""
Chunking module for splitting raw documents into retrieval-friendly chunks.

Follows the roadmap specification:
  - Chunk size: 300-800 tokens (approximated via character count)
  - Overlap: 50-150 tokens
  - Prefer heading / paragraph boundaries
  - Attach metadata for filtering and citation
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


# ── Defaults ────────────────────────────────────────────────────────────────
# 1 token ≈ 4 chars on average for English text
DEFAULT_CHUNK_SIZE_CHARS = 1600  # ~400 tokens
DEFAULT_OVERLAP_CHARS = 400  # ~100 tokens


@dataclass
class Chunk:
    """A single chunk ready for embedding and storage."""

    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


# ── Helpers ─────────────────────────────────────────────────────────────────

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)


def _detect_doc_metadata(filepath: Path) -> dict[str, Any]:
    """Derive base metadata from the file path and name."""
    stem = filepath.stem  # e.g. "hospital_faq"
    # Map filename stems to doc_type and department
    type_map: dict[str, str] = {
        "hospital_faq": "faq",
        "departments": "department_info",
        "doctor_biographies": "doctor_bio",
        "surgery_preparation": "policy",
        "appointment_policy": "policy",
        "pricing_policy": "pricing",
        "insurance_policy": "policy",
        "patient_rights": "policy",
        "emergency_services": "service_info",
        "telehealth_services": "service_info",
        "visitor_policies": "policy",
    }
    return {
        "source_id": stem,
        "title": stem.replace("_", " ").title(),
        "doc_type": type_map.get(stem, "general"),
        "department": "general",
        "language": "en",
        "access_level": "public",
        "version": "1.0",
    }


def _make_chunk_id(source_id: str, index: int, text: str) -> str:
    """Deterministic chunk id."""
    digest = hashlib.md5(text.encode()).hexdigest()[:8]
    return f"{source_id}_chunk_{index:03d}_{digest}"


# ── Section-aware splitting ─────────────────────────────────────────────────


def _split_by_headings(text: str) -> list[tuple[str, str]]:
    """
    Split markdown text into (heading, body) pairs.
    If a section body is empty the section is merged with the next one.
    """
    matches = list(_HEADING_RE.finditer(text))
    if not matches:
        return [("", text)]

    sections: list[tuple[str, str]] = []
    # Text before first heading
    preamble = text[: matches[0].start()].strip()
    if preamble:
        sections.append(("", preamble))

    for i, m in enumerate(matches):
        heading = m.group(2).strip()
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[start:end].strip()
        sections.append((heading, body))

    return sections


def _split_long_text(
    text: str,
    max_chars: int = DEFAULT_CHUNK_SIZE_CHARS,
    overlap_chars: int = DEFAULT_OVERLAP_CHARS,
) -> list[str]:
    """
    Split a long text blob into overlapping chunks, preferring paragraph
    boundaries.
    """
    if len(text) <= max_chars:
        return [text]

    paragraphs = re.split(r"\n{2,}", text)
    chunks: list[str] = []
    current = ""

    for para in paragraphs:
        candidate = (current + "\n\n" + para).strip() if current else para
        if len(candidate) > max_chars and current:
            chunks.append(current.strip())
            # Overlap: keep tail of previous chunk
            overlap_text = current[-overlap_chars:] if len(current) > overlap_chars else current
            current = (overlap_text + "\n\n" + para).strip()
        else:
            current = candidate

    if current.strip():
        chunks.append(current.strip())

    return chunks


# ── Public API ──────────────────────────────────────────────────────────────


def chunk_document(
    filepath: Path,
    *,
    chunk_size_chars: int = DEFAULT_CHUNK_SIZE_CHARS,
    overlap_chars: int = DEFAULT_OVERLAP_CHARS,
) -> list[Chunk]:
    """
    Read a markdown document and return a list of Chunk objects.

    Strategy:
      1. Split by headings to respect document structure.
      2. If a section exceeds *chunk_size_chars*, split further at paragraph
         boundaries with overlap.
      3. Attach metadata derived from the filename + section heading.
    """
    text = filepath.read_text(encoding="utf-8")
    base_meta = _detect_doc_metadata(filepath)
    sections = _split_by_headings(text)

    chunks: list[Chunk] = []
    idx = 0

    for heading, body in sections:
        # Prefix the heading into the chunk text for better retrieval context
        section_text = f"## {heading}\n\n{body}" if heading else body

        sub_chunks = _split_long_text(section_text, chunk_size_chars, overlap_chars)
        for sub in sub_chunks:
            meta = {
                **base_meta,
                "section_heading": heading,
                "chunk_id": _make_chunk_id(base_meta["source_id"], idx, sub),
            }
            # Attempt to detect department from heading
            heading_lower = heading.lower()
            for dept in (
                "cardiology",
                "orthopedics",
                "neurology",
                "pediatrics",
                "surgery",
                "emergency",
                "radiology",
                "internal medicine",
            ):
                if dept in heading_lower or dept in body[:200].lower():
                    meta["department"] = dept
                    break

            chunks.append(Chunk(text=sub, metadata=meta))
            idx += 1

    return chunks


def chunk_all_documents(
    raw_dir: Path,
    *,
    chunk_size_chars: int = DEFAULT_CHUNK_SIZE_CHARS,
    overlap_chars: int = DEFAULT_OVERLAP_CHARS,
) -> list[Chunk]:
    """Chunk every .md file in *raw_dir* and return a flat list."""
    all_chunks: list[Chunk] = []
    for md_file in sorted(raw_dir.glob("*.md")):
        all_chunks.extend(
            chunk_document(md_file, chunk_size_chars=chunk_size_chars, overlap_chars=overlap_chars)
        )
    return all_chunks
