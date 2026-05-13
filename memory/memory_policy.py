"""
Memory policy — rules governing what can and cannot be stored in memory.

From roadmap §7:
  ✅ Store: language preference, answer length preference, selected department
  ❌ Block: diagnosis, symptoms, identity numbers, phone numbers, appointment details
  - Prefer summarized memory over verbatim
  - Timestamps mandatory on all writes
"""

from __future__ import annotations

import re

# ── Sensitivity patterns ────────────────────────────────────────────────────

# Patterns that indicate sensitive content that should NOT be persisted
SENSITIVE_PATTERNS = [
    # Personal identifiers
    re.compile(r"\b\d{3}[-.\s]?\d{2}[-.\s]?\d{4}\b"),  # SSN-like
    re.compile(r"\b\d{9,11}\b"),  # Long number sequences (ID numbers)
    re.compile(r"\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"),  # Phone numbers
    re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"),  # Email addresses
]

# Keywords that indicate medical sensitive content
SENSITIVE_KEYWORDS = [
    "diagnosis",
    "diagnosed",
    "symptom",
    "symptoms",
    "prescription",
    "medication",
    "dosage",
    "blood test",
    "test result",
    "lab result",
    "medical record",
    "health record",
    "patient record",
    "treatment plan",
    "surgery date",
    "biopsy",
    "condition",
    "disease",
    "illness",
    "prognosis",
]

# Types of memory that are allowed to be stored long-term
ALLOWED_PREFERENCE_TYPES = {
    "language_preference",
    "answer_length_preference",
    "preferred_department",
    "communication_style",
    "general_preference",
}

# Types that must remain short-term only
SHORT_TERM_ONLY_TYPES = {
    "conversation",
    "session_state",
    "task_progress",
}


def contains_sensitive_data(text: str) -> tuple[bool, str]:
    """
    Check whether text contains sensitive data that should not be persisted.

    Returns:
        (is_sensitive, reason)
    """
    text_lower = text.lower()

    # Check regex patterns
    for pattern in SENSITIVE_PATTERNS:
        if pattern.search(text):
            return True, f"Contains pattern matching sensitive identifier: {pattern.pattern}"

    # Check sensitive keywords
    for keyword in SENSITIVE_KEYWORDS:
        if keyword in text_lower:
            return True, f"Contains sensitive medical keyword: '{keyword}'"

    return False, ""


def can_store_long_term(memory_type: str, content: str) -> tuple[bool, str]:
    """
    Determine if a memory entry can be stored as long-term memory.

    Returns:
        (allowed, reason)
    """
    # Check type
    if memory_type not in ALLOWED_PREFERENCE_TYPES:
        return False, f"Memory type '{memory_type}' is not allowed for long-term storage"

    # Check content
    is_sensitive, reason = contains_sensitive_data(content)
    if is_sensitive:
        return False, f"Content blocked: {reason}"

    return True, "Allowed"


def classify_sensitivity(content: str) -> str:
    """
    Classify the sensitivity level of content.

    Returns: "low", "medium", or "high"
    """
    is_sensitive, _ = contains_sensitive_data(content)
    if is_sensitive:
        return "high"

    # Medium-sensitivity heuristics
    medium_keywords = ["appointment", "schedule", "doctor name", "visit"]
    for kw in medium_keywords:
        if kw in content.lower():
            return "medium"

    return "low"
