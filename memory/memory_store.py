"""
Memory store — persistence backends for short-term and long-term memory.

Short-term: in-memory dict (lives for the session).
Long-term: SQLite database for safe preferences.
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


# ── Memory entry data structure ─────────────────────────────────────────────

def make_memory_entry(
    user_id: str,
    memory_type: str,
    content: str,
    sensitivity: str = "low",
    expires_at: str | None = None,
) -> dict[str, Any]:
    """Create a standardized memory entry dict (roadmap §7 schema)."""
    return {
        "memory_id": f"mem_{uuid.uuid4().hex[:8]}",
        "user_id": user_id,
        "type": memory_type,
        "content": content,
        "sensitivity": sensitivity,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "expires_at": expires_at,
    }


# ── Short-term store (in-memory) ────────────────────────────────────────────

class ShortTermStore:
    """In-memory store for conversation history and session state."""

    def __init__(self) -> None:
        self._conversations: dict[str, list[dict[str, str]]] = {}
        self._session_state: dict[str, dict[str, Any]] = {}

    def add_message(self, user_id: str, role: str, content: str) -> None:
        """Append a message to the conversation history."""
        if user_id not in self._conversations:
            self._conversations[user_id] = []
        self._conversations[user_id].append({
            "role": role,
            "content": content,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })

    def get_conversation(self, user_id: str, last_n: int = 10) -> list[dict[str, str]]:
        """Get the last N messages for a user."""
        msgs = self._conversations.get(user_id, [])
        return msgs[-last_n:]

    def set_session_state(self, user_id: str, key: str, value: Any) -> None:
        """Set a session state variable."""
        if user_id not in self._session_state:
            self._session_state[user_id] = {}
        self._session_state[user_id][key] = value

    def get_session_state(self, user_id: str, key: str, default: Any = None) -> Any:
        """Get a session state variable."""
        return self._session_state.get(user_id, {}).get(key, default)

    def get_all_session_state(self, user_id: str) -> dict[str, Any]:
        """Get all session state for a user."""
        return self._session_state.get(user_id, {})

    def clear_session(self, user_id: str) -> None:
        """Clear conversation and session state for a user."""
        self._conversations.pop(user_id, None)
        self._session_state.pop(user_id, None)

    def clear_all(self) -> None:
        """Clear everything."""
        self._conversations.clear()
        self._session_state.clear()


# ── Long-term store (SQLite) ────────────────────────────────────────────────

class LongTermStore:
    """SQLite-backed store for persistent safe preferences."""

    def __init__(self, db_path: str | Path = "data/memory.db") -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS memories (
                    memory_id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    type TEXT NOT NULL,
                    content TEXT NOT NULL,
                    sensitivity TEXT DEFAULT 'low',
                    created_at TEXT NOT NULL,
                    expires_at TEXT
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_memories_user
                ON memories(user_id, type)
            """)

    def store(self, entry: dict[str, Any]) -> None:
        """Store a memory entry."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO memories
                (memory_id, user_id, type, content, sensitivity, created_at, expires_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    entry["memory_id"],
                    entry["user_id"],
                    entry["type"],
                    entry["content"],
                    entry.get("sensitivity", "low"),
                    entry["created_at"],
                    entry.get("expires_at"),
                ),
            )

    def get_by_user(self, user_id: str, memory_type: str | None = None) -> list[dict[str, Any]]:
        """Retrieve all memories for a user, optionally filtered by type."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            if memory_type:
                rows = conn.execute(
                    "SELECT * FROM memories WHERE user_id = ? AND type = ? ORDER BY created_at DESC",
                    (user_id, memory_type),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM memories WHERE user_id = ? ORDER BY created_at DESC",
                    (user_id,),
                ).fetchall()
            return [dict(row) for row in rows]

    def delete_by_user(self, user_id: str) -> int:
        """Delete all memories for a user. Returns count deleted."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("DELETE FROM memories WHERE user_id = ?", (user_id,))
            return cursor.rowcount

    def delete_by_id(self, memory_id: str) -> bool:
        """Delete a specific memory entry."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("DELETE FROM memories WHERE memory_id = ?", (memory_id,))
            return cursor.rowcount > 0

    def count(self, user_id: str | None = None) -> int:
        """Count memories, optionally for a specific user."""
        with sqlite3.connect(self.db_path) as conn:
            if user_id:
                row = conn.execute(
                    "SELECT COUNT(*) FROM memories WHERE user_id = ?", (user_id,)
                ).fetchone()
            else:
                row = conn.execute("SELECT COUNT(*) FROM memories").fetchone()
            return row[0] if row else 0
