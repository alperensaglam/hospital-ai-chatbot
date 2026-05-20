"""
LLM helper — thin wrapper around LiteLLM for consistent LLM access.

All agent modules use this instead of calling LiteLLM directly,
so we have a single place to configure model, temperature, and retry logic.
"""

from __future__ import annotations

import os
import time

import litellm
from dotenv import load_dotenv

load_dotenv()

LLM_MODEL = os.getenv("LLM_MODEL", "openai/gpt-4o-mini")


def llm_call(
    messages: list[dict[str, str]],
    *,
    model: str | None = None,
    temperature: float = 0.2,
    max_tokens: int = 1024,
    response_format: dict | None = None,
) -> str:
    """
    Make a single LLM call and return the assistant's text response.

    Args:
        messages: OpenAI-style message list.
        model: LiteLLM model string. Defaults to LLM_MODEL env var.
        temperature: Sampling temperature.
        max_tokens: Maximum response tokens.
        response_format: Optional response format (e.g. {"type": "json_object"}).
    """
    from agents.metrics import metrics

    model = model or LLM_MODEL

    kwargs: dict = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if response_format:
        kwargs["response_format"] = response_format

    start = time.perf_counter()
    response = litellm.completion(**kwargs)
    duration = time.perf_counter() - start

    # Record runtime metrics
    usage = response.usage
    if usage:
        metrics.record_call(
            duration_s=duration,
            prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
            completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
            total_tokens=getattr(usage, "total_tokens", 0) or 0,
            model=model,
        )

    return response.choices[0].message.content.strip()


def llm_call_json(
    messages: list[dict[str, str]],
    *,
    model: str | None = None,
    temperature: float = 0.1,
    max_tokens: int = 1024,
) -> str:
    """
    LLM call that requests JSON output.

    The caller is responsible for parsing the returned JSON string.
    """
    return llm_call(
        messages,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        response_format={"type": "json_object"},
    )
