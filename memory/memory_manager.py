"""
Memory manager — unified interface for reading and writing memory.

Coordinates between short-term store, long-term store, and memory policy.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from agents.llm import llm_call_json
from memory.memory_policy import (
    can_store_long_term,
    classify_sensitivity,
    contains_sensitive_data,
)
from memory.memory_store import LongTermStore, ShortTermStore, make_memory_entry

logger = logging.getLogger(__name__)

# ── Preference extraction prompt ────────────────────────────────────────────

PREFERENCE_EXTRACTION_PROMPT = """\
Analyze the user message and determine if it contains a user PREFERENCE that 
should be remembered for future interactions.

Types of preferences to detect:
- language_preference: user wants responses in a specific language
- answer_length_preference: user wants concise or detailed answers
- preferred_department: user has mentioned a department they're interested in
- communication_style: user has indicated a communication preference
- general_preference: any other safe, non-sensitive preference

If a preference is detected, extract it. If no preference is found, return null.

Respond with JSON:
{
  "has_preference": true/false,
  "preference_type": "<type or null>",
  "preference_content": "<summarized preference or null>"
}
"""


class MemoryManager:
    """Unified memory interface for the agent orchestrator."""

    def __init__(
        self,
        db_path: str = "data/memory.db",
        default_user_id: str = "demo_user",
    ) -> None:
        self.short_term = ShortTermStore()
        self.long_term = LongTermStore(db_path=db_path)
        self.default_user_id = default_user_id

    # ── Read operations ─────────────────────────────────────────────────

    def get_conversation_context(
        self, user_id: str | None = None, last_n: int = 6
    ) -> str:
        """
        Get formatted conversation history for context injection.

        Returns a string suitable for including in the LLM prompt.
        """
        uid = user_id or self.default_user_id
        messages = self.short_term.get_conversation(uid, last_n=last_n)
        if not messages:
            return ""

        lines = []
        for msg in messages:
            role = msg["role"].capitalize()
            lines.append(f"{role}: {msg['content']}")
        return "\n".join(lines)

    def get_user_preferences(self, user_id: str | None = None) -> list[dict[str, Any]]:
        """Get all long-term preferences for a user."""
        uid = user_id or self.default_user_id
        return self.long_term.get_by_user(uid)

    def get_preference_summary(self, user_id: str | None = None) -> str:
        """Get a formatted summary of user preferences."""
        prefs = self.get_user_preferences(user_id)
        if not prefs:
            return ""

        lines = [f"- {p['content']} (type: {p['type']})" for p in prefs]
        return "User preferences:\n" + "\n".join(lines)

    def prefers_concise(self, user_id: str | None = None) -> bool:
        """Check if the user has a concise answer preference."""
        prefs = self.get_user_preferences(user_id)
        for p in prefs:
            if p["type"] == "answer_length_preference" and "concise" in p["content"].lower():
                return True
            if p["type"] == "answer_length_preference" and "short" in p["content"].lower():
                return True
        return False

    def get_session_state(
        self, key: str, user_id: str | None = None, default: Any = None
    ) -> Any:
        """Get a session state variable."""
        uid = user_id or self.default_user_id
        return self.short_term.get_session_state(uid, key, default)

    def get_full_context(self, user_id: str | None = None) -> str:
        """Get combined conversation + preference context."""
        conv = self.get_conversation_context(user_id)
        prefs = self.get_preference_summary(user_id)
        parts = [p for p in [conv, prefs] if p]
        return "\n\n".join(parts)

    # ── Write operations ────────────────────────────────────────────────

    def add_message(self, role: str, content: str, user_id: str | None = None) -> None:
        """Record a conversation message in short-term memory."""
        uid = user_id or self.default_user_id
        self.short_term.add_message(uid, role, content)

    def set_session_state(
        self, key: str, value: Any, user_id: str | None = None
    ) -> None:
        """Set a session state variable."""
        uid = user_id or self.default_user_id
        self.short_term.set_session_state(uid, key, value)

    def try_store_preference(
        self, memory_type: str, content: str, user_id: str | None = None
    ) -> tuple[bool, str]:
        """
        Attempt to store a long-term preference, applying policy checks.

        Returns:
            (stored, reason)
        """
        uid = user_id or self.default_user_id

        # Deduplication check
        existing_prefs = self.get_user_preferences(uid)
        for p in existing_prefs:
            if p["type"] == memory_type and p["content"].lower().strip() == content.lower().strip():
                logger.info(f"Memory skipped: already exists ({memory_type}: {content})")
                return False, "Already stored"

        # Policy check
        allowed, reason = can_store_long_term(memory_type, content)
        if not allowed:
            logger.info(f"Memory write blocked: {reason}")
            return False, reason

        sensitivity = classify_sensitivity(content)
        entry = make_memory_entry(uid, memory_type, content, sensitivity)
        self.long_term.store(entry)
        logger.info(f"Memory stored: {entry['memory_id']} ({memory_type})")
        return True, f"Stored as {entry['memory_id']}"

    def extract_and_store_preferences(
        self, user_message: str, user_id: str | None = None
    ) -> dict | None:
        """
        Use LLM to detect preferences in a user message and store them.

        Returns the extracted preference dict, or None if none found.
        """
        messages = [
            {"role": "system", "content": PREFERENCE_EXTRACTION_PROMPT},
            {"role": "user", "content": user_message},
        ]

        try:
            raw = llm_call_json(messages, temperature=0.0)
            result = json.loads(raw)
        except Exception:
            return None

        if result.get("has_preference") and result.get("preference_type"):
            stored, reason = self.try_store_preference(
                result["preference_type"],
                result["preference_content"],
                user_id,
            )
            result["stored"] = stored
            result["reason"] = reason
            return result

        return None

    # ── Delete / reset operations ───────────────────────────────────────

    def reset_session(self, user_id: str | None = None) -> None:
        """Clear short-term conversation and session state."""
        uid = user_id or self.default_user_id
        self.short_term.clear_session(uid)
        logger.info(f"Session reset for user: {uid}")

    def reset_all_memory(self, user_id: str | None = None) -> int:
        """Clear all memory (short-term + long-term) for a user."""
        uid = user_id or self.default_user_id
        self.short_term.clear_session(uid)
        count = self.long_term.delete_by_user(uid)
        logger.info(f"All memory reset for user: {uid} ({count} long-term entries deleted)")
        return count

    def get_memory_stats(self, user_id: str | None = None) -> dict[str, Any]:
        """Get memory statistics for debug display."""
        uid = user_id or self.default_user_id
        conv = self.short_term.get_conversation(uid)
        session = self.short_term.get_all_session_state(uid)
        long_term = self.long_term.get_by_user(uid)

        return {
            "user_id": uid,
            "conversation_messages": len(conv),
            "session_state_keys": list(session.keys()),
            "long_term_memories": len(long_term),
            "long_term_details": long_term,
        }
