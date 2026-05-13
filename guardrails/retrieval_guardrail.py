"""
Retrieval guardrail — access-level filtering on retrieved documents.

From roadmap §8:
  - Public user can only access access_level: "public" documents
  - Filter out documents that exceed the user's access level
"""

from __future__ import annotations

from typing import Any

from guardrails.audit_logger import log_event
from rag.retriever import RetrievedChunk

# Access level hierarchy (higher number = more restrictive)
ACCESS_LEVELS = {
    "public": 0,
    "registered": 1,
    "staff": 2,
    "admin": 3,
}


def filter_by_access_level(
    chunks: list[RetrievedChunk],
    user_access_level: str = "public",
    user_id: str = "demo_user",
) -> list[RetrievedChunk]:
    """
    Filter retrieved chunks based on user access level.

    Chunks with an access_level higher than the user's level are removed.

    Args:
        chunks: Retrieved chunks to filter.
        user_access_level: The user's access level.
        user_id: For audit logging.

    Returns:
        Filtered list of chunks.
    """
    user_level = ACCESS_LEVELS.get(user_access_level, 0)
    allowed: list[RetrievedChunk] = []
    blocked_count = 0

    for chunk in chunks:
        doc_level_str = chunk.metadata.get("access_level", "public")
        doc_level = ACCESS_LEVELS.get(doc_level_str, 0)

        if doc_level <= user_level:
            allowed.append(chunk)
        else:
            blocked_count += 1

    if blocked_count > 0:
        log_event(
            "retrieval_guardrail",
            user_id=user_id,
            outcome="modified",
            reason=f"Filtered {blocked_count} chunks above user access level '{user_access_level}'",
            details={
                "user_access_level": user_access_level,
                "total_chunks": len(chunks),
                "allowed_chunks": len(allowed),
                "blocked_chunks": blocked_count,
            },
        )
    else:
        log_event(
            "retrieval_guardrail",
            user_id=user_id,
            outcome="allowed",
            reason="All chunks within user access level",
            details={"total_chunks": len(chunks)},
        )

    return allowed
