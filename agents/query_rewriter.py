"""
Query rewriter — transforms vague user queries into retrieval-friendly queries.

Examples (from roadmap §6):
  "Can I see a heart doctor?" →
    ["cardiology department appointment policy", "cardiology doctors availability"]

  "What do I need before my operation?" →
    ["surgery preparation instructions", "pre-operative requirements"]
"""

from __future__ import annotations

import json

from agents.llm import llm_call_json

SYSTEM_PROMPT = """\
You are a query rewriting assistant for a hospital information system.

Your job is to take a user's natural language question and rewrite it into 1-3 
clear, specific queries optimized for searching a hospital document database.

The database contains documents about:
- Hospital FAQ, visiting hours, general info
- Department descriptions (Cardiology, Orthopedics, Neurology, Pediatrics, Surgery, Emergency, Radiology, Internal Medicine)
- Doctor biographies and specialties
- Surgery preparation instructions
- Appointment policies
- Pricing and payment policies
- Insurance coverage
- Patient rights
- Emergency services
- Telehealth services
- Visitor policies

Guidelines:
- Replace colloquial terms with medical/formal terms (e.g., "heart doctor" → "cardiology")
- Expand ambiguous questions into specific sub-queries
- Keep each query concise (5-15 words)
- If the question is already clear, return it as-is

Respond ONLY with a JSON object:
{"queries": ["query1", "query2", ...], "reasoning": "<brief explanation of rewrites>"}
"""


def rewrite_query(user_message: str, intent: str = "", memory_context: str = "") -> dict:
    """
    Rewrite a user message into retrieval-friendly queries.

    Args:
        user_message: The original user question.
        intent: Classified intent (for context).
        memory_context: Relevant memory context (e.g., user preferences).

    Returns:
        dict with keys: queries (list[str]), reasoning (str)
    """
    user_content = f"User question: {user_message}"
    if intent:
        user_content += f"\nClassified intent: {intent}"
    if memory_context:
        user_content += f"\nConversation context: {memory_context}"

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]

    raw = llm_call_json(messages, temperature=0.1)

    try:
        result = json.loads(raw)
    except json.JSONDecodeError:
        result = {"queries": [user_message], "reasoning": "Failed to parse, using original query"}

    if not result.get("queries"):
        result["queries"] = [user_message]

    return result
