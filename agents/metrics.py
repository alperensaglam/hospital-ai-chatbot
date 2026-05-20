"""
Runtime metrics tracker for LLM calls.

Required metrics (from course specification §5.1):
  - total-llm-calls
  - e2e-response-time
  - mean-response-time
  - total-tokens
  - mean-token-per-call

These metrics are emitted to the console on every run.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class CallRecord:
    """Record of a single LLM call."""

    duration_s: float
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    model: str


class MetricsTracker:
    """Thread-safe metrics tracker for LLM calls."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._calls: list[CallRecord] = []
        self._e2e_start: float | None = None
        self._e2e_end: float | None = None

    def reset(self) -> None:
        """Reset all metrics for a new query."""
        with self._lock:
            self._calls.clear()
            self._e2e_start = None
            self._e2e_end = None

    def start_e2e(self) -> None:
        """Mark the start of end-to-end processing."""
        self._e2e_start = time.perf_counter()

    def end_e2e(self) -> None:
        """Mark the end of end-to-end processing."""
        self._e2e_end = time.perf_counter()

    def record_call(
        self,
        duration_s: float,
        prompt_tokens: int,
        completion_tokens: int,
        total_tokens: int,
        model: str = "",
    ) -> None:
        """Record a completed LLM call."""
        with self._lock:
            self._calls.append(
                CallRecord(
                    duration_s=duration_s,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=total_tokens,
                    model=model,
                )
            )

    # ── Computed metrics ────────────────────────────────────────────────

    @property
    def total_llm_calls(self) -> int:
        return len(self._calls)

    @property
    def e2e_response_time(self) -> float:
        if self._e2e_start is not None and self._e2e_end is not None:
            return self._e2e_end - self._e2e_start
        return 0.0

    @property
    def mean_response_time(self) -> float:
        if not self._calls:
            return 0.0
        return sum(c.duration_s for c in self._calls) / len(self._calls)

    @property
    def total_tokens(self) -> int:
        return sum(c.total_tokens for c in self._calls)

    @property
    def mean_token_per_call(self) -> float:
        if not self._calls:
            return 0.0
        return self.total_tokens / len(self._calls)

    def summary(self) -> dict[str, Any]:
        """Return a dict of the 5 required metrics."""
        return {
            "total-llm-calls": self.total_llm_calls,
            "e2e-response-time": round(self.e2e_response_time, 2),
            "mean-response-time": round(self.mean_response_time, 2),
            "total-tokens": self.total_tokens,
            "mean-token-per-call": round(self.mean_token_per_call, 1),
        }

    def print_summary(self) -> None:
        """Print the 5 required metrics to console."""
        s = self.summary()
        print()
        print("═" * 45)
        print("  Runtime Metrics")
        print("═" * 45)
        print(f"  total-llm-calls:     {s['total-llm-calls']}")
        print(f"  e2e-response-time:   {s['e2e-response-time']}s")
        print(f"  mean-response-time:  {s['mean-response-time']}s")
        print(f"  total-tokens:        {s['total-tokens']}")
        print(f"  mean-token-per-call: {s['mean-token-per-call']}")
        print("═" * 45)


# Global singleton — shared across all modules
metrics = MetricsTracker()
