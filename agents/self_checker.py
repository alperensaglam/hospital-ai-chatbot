"""
Self-checker — verifies that the draft answer is supported by retrieved context.

From roadmap §6:
  - Check whether the answer includes unsupported claims
  - If support is weak, answer with uncertainty instead of hallucinating
"""

from __future__ import annotations

import json

from agents.llm import llm_call_json

SYSTEM_PROMPT = """\
You are a verification assistant. Your job is to check whether a draft answer
is fully supported by the provided context.

Evaluate the draft answer against the context and determine:
1. Is every factual claim in the answer supported by the context?
2. Does the answer contain any information NOT found in the context?
3. Does the answer make any medical claims that could be harmful if incorrect?

Respond ONLY with a JSON object:
{
  "verdict": "supported" | "partially_supported" | "unsupported",
  "confidence": <0.0-1.0>,
  "unsupported_claims": ["claim1", "claim2", ...],
  "reasoning": "<brief explanation>"
}
"""


def check_answer(
    draft_answer: str,
    context: str,
    user_question: str,
) -> dict:
    """
    Verify whether the draft answer is grounded in the retrieved context.

    Args:
        draft_answer: The generated answer to verify.
        context: The retrieved context that should support the answer.
        user_question: The original user question (for context).

    Returns:
        dict with keys: verdict, confidence, unsupported_claims, reasoning
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"User question: {user_question}\n\n"
                f"Retrieved context:\n{context}\n\n"
                f"Draft answer:\n{draft_answer}"
            ),
        },
    ]

    raw = llm_call_json(messages, temperature=0.0)

    try:
        result = json.loads(raw)
    except json.JSONDecodeError:
        result = {
            "verdict": "unsupported",
            "confidence": 0.0,
            "unsupported_claims": [],
            "reasoning": "Failed to parse verification response",
        }

    # Normalize verdict
    valid_verdicts = {"supported", "partially_supported", "unsupported"}
    if result.get("verdict") not in valid_verdicts:
        result["verdict"] = "unsupported"

    return result


def add_uncertainty_markers(answer: str, unsupported_claims: list[str]) -> str:
    """
    If the answer has unsupported claims, add a disclaimer.
    """
    if not unsupported_claims:
        return answer

    disclaimer = (
        "\n\n⚠️ **Note:** Some details in this response may not be fully verified "
        "against our records. Please contact the hospital at (555) 100-2000 "
        "for confirmation."
    )
    return answer + disclaimer
