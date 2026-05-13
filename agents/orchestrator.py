"""
Agent orchestrator — LangGraph StateGraph implementing the full agentic RAG pipeline.

Architecture (from roadmap §5):
  User → Input Guardrail → Agent Orchestrator → Agentic RAG Pipeline
       → Optional Structured Tools → Output Guardrail → Final Answer + Memory Update

LangGraph nodes:
  1. input_guard      — input guardrail check
  2. read_memory      — load conversation context + preferences
  3. classify_intent  — determine what the user wants
  4. rewrite_query    — transform vague queries
  5. retrieve         — vector DB search
  6. check_sufficiency — are the results good enough?
  7. route_tools      — optional structured data lookup
  8. synthesize       — generate grounded answer
  9. self_check       — verify answer is supported
  10. output_guard    — validate and sanitize output
  11. update_memory   — save conversation and preferences
"""

from __future__ import annotations

import json
import logging
from typing import Any, TypedDict

from langgraph.graph import END, StateGraph

from agents.answer_synthesizer import synthesize_answer
from agents.intent_classifier import classify_intent
from agents.query_rewriter import rewrite_query
from agents.self_checker import add_uncertainty_markers, check_answer
from guardrails.input_guardrail import check_input
from guardrails.output_guardrail import check_output
from guardrails.retrieval_guardrail import filter_by_access_level
from guardrails.tool_guardrail import check_tool_action
from memory.memory_manager import MemoryManager
from rag.retriever import RetrievedChunk, format_context, retrieve
from tools.hospital_service_db import (
    format_tool_results,
    get_doctor_schedule,
    get_pricing,
    search_doctors,
)

logger = logging.getLogger(__name__)


# ── Agent state ─────────────────────────────────────────────────────────────


class AgentState(TypedDict, total=False):
    """State passed between LangGraph nodes."""

    # Input
    user_message: str
    user_id: str

    # Guardrail
    input_guard_result: dict
    is_blocked: bool

    # Memory
    memory_context: str
    conversation_context: str
    prefer_concise: bool

    # Intent
    intent: str
    intent_confidence: float
    intent_reasoning: str

    # Query rewriting
    rewritten_queries: list[str]
    rewrite_reasoning: str

    # Retrieval
    retrieved_chunks: list[RetrievedChunk]
    formatted_context: str
    is_sufficient: bool
    retrieval_attempt: int

    # Tools
    tool_results: str
    needs_tool_confirmation: bool
    tool_confirmation_message: str

    # Answer
    draft_answer: str
    self_check_result: dict
    final_answer: str

    # Debug / trace
    trace: list[dict[str, Any]]


# ── Node functions ──────────────────────────────────────────────────────────

# Global memory manager instance (initialized in build_graph)
_memory_manager: MemoryManager | None = None


def _get_memory_manager() -> MemoryManager:
    global _memory_manager
    if _memory_manager is None:
        _memory_manager = MemoryManager()
    return _memory_manager


def set_memory_manager(manager: MemoryManager) -> None:
    """Allow external code (e.g. UI) to inject a shared memory manager."""
    global _memory_manager
    _memory_manager = manager


def node_input_guard(state: AgentState) -> dict:
    """Check input against safety rules."""
    result = check_input(state["user_message"], state.get("user_id", "demo_user"))
    trace_entry = {"node": "input_guard", "result": result}

    if result.get("blocked"):
        return {
            "input_guard_result": result,
            "is_blocked": True,
            "final_answer": result.get("safe_response", "I cannot process this request."),
            "trace": state.get("trace", []) + [trace_entry],
        }

    return {
        "input_guard_result": result,
        "is_blocked": False,
        "trace": state.get("trace", []) + [trace_entry],
    }


def node_read_memory(state: AgentState) -> dict:
    """Load conversation context and user preferences."""
    mm = _get_memory_manager()
    user_id = state.get("user_id", "demo_user")

    conv_ctx = mm.get_conversation_context(user_id)
    pref_summary = mm.get_preference_summary(user_id)
    full_ctx = mm.get_full_context(user_id)
    concise = mm.prefers_concise(user_id)

    trace_entry = {
        "node": "read_memory",
        "has_conversation": bool(conv_ctx),
        "has_preferences": bool(pref_summary),
        "prefers_concise": concise,
    }

    return {
        "memory_context": full_ctx,
        "conversation_context": conv_ctx,
        "prefer_concise": concise,
        "trace": state.get("trace", []) + [trace_entry],
    }


def node_classify_intent(state: AgentState) -> dict:
    """Classify the user's intent."""
    result = classify_intent(
        state["user_message"],
        conversation_context=state.get("conversation_context", ""),
    )

    trace_entry = {"node": "classify_intent", **result}

    return {
        "intent": result["intent"],
        "intent_confidence": result.get("confidence", 0.0),
        "intent_reasoning": result.get("reasoning", ""),
        "trace": state.get("trace", []) + [trace_entry],
    }


def node_rewrite_query(state: AgentState) -> dict:
    """Rewrite the user query for better retrieval."""
    result = rewrite_query(
        state["user_message"],
        intent=state.get("intent", ""),
        memory_context=state.get("memory_context", ""),
    )

    trace_entry = {"node": "rewrite_query", **result}

    return {
        "rewritten_queries": result.get("queries", [state["user_message"]]),
        "rewrite_reasoning": result.get("reasoning", ""),
        "trace": state.get("trace", []) + [trace_entry],
    }


def node_retrieve(state: AgentState) -> dict:
    """Retrieve relevant chunks from ChromaDB."""
    queries = state.get("rewritten_queries", [state["user_message"]])
    attempt = state.get("retrieval_attempt", 0) + 1
    user_id = state.get("user_id", "demo_user")

    all_chunks: list[RetrievedChunk] = []
    seen_ids: set[str] = set()

    for query in queries:
        chunks = retrieve(query, top_k=4)
        for chunk in chunks:
            cid = chunk.chunk_id
            if cid not in seen_ids:
                seen_ids.add(cid)
                all_chunks.append(chunk)

    # Apply retrieval guardrail
    all_chunks = filter_by_access_level(all_chunks, user_access_level="public", user_id=user_id)

    # Sort by score (lower = better for cosine distance)
    all_chunks.sort(key=lambda c: c.score)

    # Keep top results
    all_chunks = all_chunks[:6]

    context = format_context(all_chunks)

    trace_entry = {
        "node": "retrieve",
        "attempt": attempt,
        "queries_used": queries,
        "chunks_found": len(all_chunks),
        "top_scores": [c.score for c in all_chunks[:3]],
    }

    return {
        "retrieved_chunks": all_chunks,
        "formatted_context": context,
        "retrieval_attempt": attempt,
        "trace": state.get("trace", []) + [trace_entry],
    }


def node_check_sufficiency(state: AgentState) -> dict:
    """Check if retrieval results are sufficient to answer the question."""
    chunks = state.get("retrieved_chunks", [])

    # Heuristics for sufficiency
    if not chunks:
        is_sufficient = False
    elif chunks[0].score > 0.8:  # All results have high distance = low relevance
        is_sufficient = False
    elif len(chunks) < 2 and chunks[0].score > 0.5:
        is_sufficient = False
    else:
        is_sufficient = True

    trace_entry = {
        "node": "check_sufficiency",
        "is_sufficient": is_sufficient,
        "num_chunks": len(chunks),
        "best_score": chunks[0].score if chunks else None,
    }

    return {
        "is_sufficient": is_sufficient,
        "trace": state.get("trace", []) + [trace_entry],
    }


def node_route_tools(state: AgentState) -> dict:
    """Route to structured tools if needed (schedule, pricing)."""
    intent = state.get("intent", "")
    user_msg = state["user_message"].lower()
    tool_results = ""

    if intent == "doctor_info" or "schedule" in user_msg or "available" in user_msg:
        # Try to find doctor schedule
        check = check_tool_action("search_schedule", user_id=state.get("user_id", "demo_user"))
        if check["allowed"]:
            # Extract department hints
            for dept in ["cardiology", "orthopedics", "neurology", "pediatrics", "surgery", "internal medicine"]:
                if dept in user_msg:
                    results = search_doctors(department=dept)
                    if results:
                        schedules = []
                        for doc in results:
                            sched = get_doctor_schedule(doctor_id=doc["doctor_id"])
                            schedules.extend(sched)
                        tool_results = format_tool_results(schedules, "Doctor Schedule Search")
                    break
            else:
                # General doctor search
                results = search_doctors()
                tool_results = format_tool_results(results, "Doctor Search")

    elif intent == "pricing" or "price" in user_msg or "cost" in user_msg or "how much" in user_msg:
        check = check_tool_action("get_pricing", user_id=state.get("user_id", "demo_user"))
        if check["allowed"]:
            results = get_pricing()
            tool_results = format_tool_results(results, "Pricing Lookup")

    trace_entry = {
        "node": "route_tools",
        "intent": intent,
        "has_tool_results": bool(tool_results),
    }

    return {
        "tool_results": tool_results,
        "trace": state.get("trace", []) + [trace_entry],
    }


def node_synthesize(state: AgentState) -> dict:
    """Generate a grounded answer."""
    answer = synthesize_answer(
        user_question=state["user_message"],
        context=state.get("formatted_context", ""),
        memory_context=state.get("memory_context", ""),
        tool_results=state.get("tool_results", ""),
        prefer_concise=state.get("prefer_concise", False),
    )

    trace_entry = {"node": "synthesize", "answer_length": len(answer)}

    return {
        "draft_answer": answer,
        "trace": state.get("trace", []) + [trace_entry],
    }


def node_self_check(state: AgentState) -> dict:
    """Verify the answer is supported by context."""
    result = check_answer(
        draft_answer=state.get("draft_answer", ""),
        context=state.get("formatted_context", ""),
        user_question=state["user_message"],
    )

    answer = state.get("draft_answer", "")
    if result.get("verdict") == "unsupported":
        answer = (
            "I don't have enough verified information to fully answer this question. "
            "Please contact the hospital directly at (555) 100-2000 for accurate information."
        )
    elif result.get("verdict") == "partially_supported":
        answer = add_uncertainty_markers(answer, result.get("unsupported_claims", []))

    trace_entry = {"node": "self_check", **result}

    return {
        "self_check_result": result,
        "draft_answer": answer,
        "trace": state.get("trace", []) + [trace_entry],
    }


def node_output_guard(state: AgentState) -> dict:
    """Validate and sanitize the final output."""
    result = check_output(
        state.get("draft_answer", ""),
        user_id=state.get("user_id", "demo_user"),
    )

    trace_entry = {
        "node": "output_guard",
        "modifications": result.get("modifications", []),
    }

    return {
        "final_answer": result["modified_answer"],
        "trace": state.get("trace", []) + [trace_entry],
    }


def node_update_memory(state: AgentState) -> dict:
    """Save conversation turn and check for preference updates."""
    mm = _get_memory_manager()
    user_id = state.get("user_id", "demo_user")

    # Save the conversation turn
    mm.add_message("user", state["user_message"], user_id)
    mm.add_message("assistant", state.get("final_answer", ""), user_id)

    # Try to extract and store preferences
    pref = mm.extract_and_store_preferences(state["user_message"], user_id)

    trace_entry = {
        "node": "update_memory",
        "preference_detected": pref is not None,
        "preference_details": pref,
    }

    return {
        "trace": state.get("trace", []) + [trace_entry],
    }


# ── Routing functions ───────────────────────────────────────────────────────


def route_after_input_guard(state: AgentState) -> str:
    """Route based on input guardrail result."""
    if state.get("is_blocked"):
        return "update_memory"  # Still log the blocked interaction
    return "read_memory"


def route_after_intent(state: AgentState) -> str:
    """Route based on classified intent."""
    intent = state.get("intent", "")

    if intent == "general_greeting":
        # Skip retrieval for greetings
        return "synthesize"

    if intent == "out_of_scope":
        return "synthesize"

    if intent in ("appointment_action",):
        return "rewrite_query"

    # All other intents: go through RAG pipeline
    return "rewrite_query"


def route_after_sufficiency(state: AgentState) -> str:
    """Route based on retrieval sufficiency."""
    attempt = state.get("retrieval_attempt", 1)
    is_sufficient = state.get("is_sufficient", False)

    if is_sufficient:
        # Check if we also need tools
        intent = state.get("intent", "")
        if intent in ("doctor_info", "pricing", "appointment_action"):
            return "route_tools"
        return "synthesize"

    # Retry once with different queries
    if attempt < 2:
        return "rewrite_query"

    # After 2 attempts, proceed anyway (with weak context)
    intent = state.get("intent", "")
    if intent in ("doctor_info", "pricing", "appointment_action"):
        return "route_tools"
    return "synthesize"


# ── Graph builder ───────────────────────────────────────────────────────────


def build_graph(memory_manager: MemoryManager | None = None) -> StateGraph:
    """
    Build the LangGraph agent.

    Returns a compiled StateGraph ready for invocation.
    """
    if memory_manager:
        set_memory_manager(memory_manager)

    graph = StateGraph(AgentState)

    # Add nodes
    graph.add_node("input_guard", node_input_guard)
    graph.add_node("read_memory", node_read_memory)
    graph.add_node("classify_intent", node_classify_intent)
    graph.add_node("rewrite_query", node_rewrite_query)
    graph.add_node("retrieve", node_retrieve)
    graph.add_node("check_sufficiency", node_check_sufficiency)
    graph.add_node("route_tools", node_route_tools)
    graph.add_node("synthesize", node_synthesize)
    graph.add_node("self_check", node_self_check)
    graph.add_node("output_guard", node_output_guard)
    graph.add_node("update_memory", node_update_memory)

    # Set entry point
    graph.set_entry_point("input_guard")

    # Add edges
    graph.add_conditional_edges("input_guard", route_after_input_guard)
    graph.add_edge("read_memory", "classify_intent")
    graph.add_conditional_edges("classify_intent", route_after_intent)
    graph.add_edge("rewrite_query", "retrieve")
    graph.add_edge("retrieve", "check_sufficiency")
    graph.add_conditional_edges("check_sufficiency", route_after_sufficiency)
    graph.add_edge("route_tools", "synthesize")
    graph.add_edge("synthesize", "self_check")
    graph.add_edge("self_check", "output_guard")
    graph.add_edge("output_guard", "update_memory")
    graph.add_edge("update_memory", END)

    return graph.compile()


# ── Convenience function ────────────────────────────────────────────────────


def run_agent(
    user_message: str,
    user_id: str = "demo_user",
    memory_manager: MemoryManager | None = None,
) -> dict:
    """
    Run the full agent pipeline for a user message.

    Returns the final AgentState dict.
    """
    graph = build_graph(memory_manager)
    initial_state: AgentState = {
        "user_message": user_message,
        "user_id": user_id,
        "trace": [],
    }
    result = graph.invoke(initial_state)
    return result
