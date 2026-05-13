"""
Application entrypoint — wires everything together.

Provides the main chat interface logic used by both CLI and Streamlit UI.
"""

from __future__ import annotations

import logging
import os
import sys

from dotenv import load_dotenv

load_dotenv()

# Configure logging
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)


def create_app():
    """Initialize and return the agent pipeline components."""
    from agents.orchestrator import build_graph, set_memory_manager
    from memory.memory_manager import MemoryManager

    # Initialize memory manager
    memory_manager = MemoryManager(
        db_path="data/memory.db",
        default_user_id="demo_user",
    )

    # Build the agent graph
    graph = build_graph(memory_manager)

    return graph, memory_manager


def chat(
    user_message: str,
    user_id: str = "demo_user",
    graph=None,
    memory_manager=None,
) -> dict:
    """
    Process a user message through the full agent pipeline.

    Returns:
        dict with keys: answer, trace, intent, is_blocked
    """
    if graph is None:
        graph, memory_manager = create_app()

    from agents.orchestrator import set_memory_manager
    if memory_manager:
        set_memory_manager(memory_manager)

    initial_state = {
        "user_message": user_message,
        "user_id": user_id,
        "trace": [],
    }

    result = graph.invoke(initial_state)

    return {
        "answer": result.get("final_answer", "I'm sorry, something went wrong."),
        "trace": result.get("trace", []),
        "intent": result.get("intent", "unknown"),
        "is_blocked": result.get("is_blocked", False),
        "self_check": result.get("self_check_result", {}),
        "retrieved_chunks": result.get("retrieved_chunks", []),
    }


def cli_mode():
    """Simple CLI interface for testing."""
    print("=" * 60)
    print("  City General Hospital AI Assistant")
    print("  Type 'quit' to exit, 'reset' to clear memory")
    print("=" * 60)

    graph, memory_manager = create_app()

    while True:
        try:
            user_input = input("\n👤 You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break
        if user_input.lower() == "reset":
            memory_manager.reset_all_memory()
            print("🔄 Memory reset complete.")
            continue

        result = chat(user_input, graph=graph, memory_manager=memory_manager)

        print(f"\n🏥 Assistant: {result['answer']}")

        if os.getenv("DEBUG_MODE", "false").lower() == "true":
            print(f"\n   [Intent: {result['intent']}]")
            if result.get("self_check"):
                print(f"   [Self-check: {result['self_check'].get('verdict', 'n/a')}]")


if __name__ == "__main__":
    cli_mode()
