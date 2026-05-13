"""
Audit logger — structured logging for all guardrail decisions.

From roadmap §8:
  Record blocked prompts, tool calls, memory writes, and output modifications.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

LOG_DIR = Path("data/audit_logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)

logger = logging.getLogger("audit")


def _get_log_path() -> Path:
    """One log file per day."""
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    return LOG_DIR / f"audit_{today}.jsonl"


def log_event(
    event_type: str,
    *,
    user_id: str = "demo_user",
    details: dict[str, Any] | None = None,
    outcome: str = "allowed",
    reason: str = "",
) -> dict[str, Any]:
    """
    Log an audit event.

    Args:
        event_type: Type of event (e.g., "input_guardrail", "tool_call", "memory_write").
        user_id: The user involved.
        details: Additional structured details.
        outcome: "allowed" | "blocked" | "modified"
        reason: Human-readable reason.

    Returns:
        The log entry dict.
    """
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "event_type": event_type,
        "user_id": user_id,
        "outcome": outcome,
        "reason": reason,
        "details": details or {},
    }

    # Write to file
    log_path = _get_log_path()
    with log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    # Also log via Python logging
    log_msg = f"[{event_type}] {outcome}: {reason}"
    if outcome == "blocked":
        logger.warning(log_msg)
    else:
        logger.info(log_msg)

    return entry


def get_recent_events(
    n: int = 20,
    event_type: str | None = None,
) -> list[dict[str, Any]]:
    """Read the most recent audit events from today's log."""
    log_path = _get_log_path()
    if not log_path.exists():
        return []

    events = []
    with log_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
                if event_type and event.get("event_type") != event_type:
                    continue
                events.append(event)
            except json.JSONDecodeError:
                continue

    return events[-n:]
