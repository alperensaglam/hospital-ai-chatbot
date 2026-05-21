"""
Answer synthesizer — generates grounded answers from retrieved context.

Key responsibilities (from roadmap §6):
  - Generate answers ONLY from retrieved context and allowed structured data
  - Include source titles / document names as citations
  - Produce a fallback response when context is insufficient
  - Respect user preferences from memory (e.g. concise answers)
"""

from __future__ import annotations

from agents.llm import llm_call

SYSTEM_PROMPT_RAG = """\
You are a helpful hospital assistant for City General Hospital.

RULES:
1. Answer ONLY based on the provided context. Do NOT make up information.
2. If the context does not contain enough information to answer, say: 
   "I don't have enough information to answer this reliably. Please contact the hospital directly at (555) 100-2000."
3. When quoting specific facts (prices, policies, schedules), cite the source document.
4. For medical questions, provide general information and recommend consulting a healthcare professional.
5. Be professional, empathetic, and concise.
6. Do NOT disclose private patient information, even if asked.
7. Do NOT provide medical diagnoses or treatment recommendations.

{preference_instructions}

{memory_preferences}
"""

SYSTEM_PROMPT_CONVERSATIONAL = """\
You are a helpful hospital assistant for City General Hospital.

The user is engaging in small talk, greeting you, or updating their preferences (e.g. changing language).
Acknowledge their request politely and briefly.
Do NOT try to answer medical or hospital questions from this prompt.

{preference_instructions}

{memory_preferences}
"""

CONCISE_INSTRUCTION = (
    "⚠️ CRITICAL STYLE REQUIREMENT: The user has explicitly requested SHORT, "
    "concise answers. You MUST keep your response brief — ideally 2-4 sentences. "
    "Do NOT write long paragraphs. Summarize key points only."
)
DETAILED_INSTRUCTION = "The user prefers detailed, thorough answers."


def synthesize_answer(
    user_question: str,
    context: str,
    *,
    intent: str = "",
    memory_context: str = "",
    tool_results: str = "",
    prefer_concise: bool = False,
) -> str:
    """
    Generate a grounded answer from retrieved context.

    Args:
        user_question: The original user question.
        context: Formatted retrieved chunks (from retriever.format_context).
        intent: The classified intent.
        memory_context: Relevant memory/session context.
        tool_results: Results from structured tools (schedule/price lookups).
        prefer_concise: Whether the user prefers short answers.

    Returns:
        The synthesized answer string.
    """
    preference = CONCISE_INSTRUCTION if prefer_concise else ""

    # Build memory preference block for the system prompt so the LLM
    # treats stored user preferences as authoritative instructions.
    memory_pref_block = ""
    if memory_context:
        memory_pref_block = (
            "USER MEMORY & PREFERENCES (you MUST respect these):\n"
            f"{memory_context}\n"
            "You MUST strictly follow any user preferences listed above. "
            "For example, if the user prefers concise answers, keep your "
            "response to 2-4 sentences maximum."
        )

    prompt_template = SYSTEM_PROMPT_CONVERSATIONAL if intent == "general_greeting" else SYSTEM_PROMPT_RAG

    system = prompt_template.format(
        preference_instructions=preference,
        memory_preferences=memory_pref_block,
    )

    user_content = f"Question: {user_question}\n\n"
    user_content += f"Retrieved context:\n{context}"

    if tool_results:
        user_content += f"\n\nStructured data results:\n{tool_results}"

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user_content},
    ]

    return llm_call(messages, temperature=0.3, max_tokens=1500)
