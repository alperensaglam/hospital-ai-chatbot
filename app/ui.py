"""
Streamlit chat UI for the Hospital AI Assistant.

Features:
  - Chat message history display
  - Debug panel showing retrieved chunks, intent, guardrail results
  - Memory controls (reset session, reset all memory)
  - Memory stats in sidebar

Run with:
    uv run streamlit run app/ui.py
"""

from __future__ import annotations

import json
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# ── Page config ─────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="City General Hospital AI Assistant",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ──────────────────────────────────────────────────────────────

st.markdown("""
<style>
    /* Main header */
    .main-header {
        text-align: center;
        padding: 1rem 0;
        border-bottom: 2px solid #e0e0e0;
        margin-bottom: 1rem;
    }
    .main-header h1 {
        color: #1a73e8;
        font-size: 1.8rem;
        margin: 0;
    }
    .main-header p {
        color: #666;
        font-size: 0.9rem;
        margin: 0.3rem 0 0 0;
    }

    /* Debug panel */
    .debug-panel {
        background-color: #f8f9fa;
        border: 1px solid #e0e0e0;
        border-radius: 8px;
        padding: 1rem;
        font-size: 0.8rem;
        margin-top: 0.5rem;
    }

    /* Sidebar styling */
    .sidebar-section {
        background-color: #f0f4f8;
        border-radius: 8px;
        padding: 0.8rem;
        margin-bottom: 0.8rem;
    }
</style>
""", unsafe_allow_html=True)


# ── Initialize session state ────────────────────────────────────────────────

def init_session():
    """Initialize Streamlit session state."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "graph" not in st.session_state:
        from app.main import create_app
        graph, memory_manager = create_app()
        st.session_state.graph = graph
        st.session_state.memory_manager = memory_manager
    if "debug_mode" not in st.session_state:
        st.session_state.debug_mode = os.getenv("DEBUG_MODE", "true").lower() == "true"
    if "last_trace" not in st.session_state:
        st.session_state.last_trace = []


init_session()


# ── Sidebar ─────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## ⚙️ Controls")

    # Debug toggle
    st.session_state.debug_mode = st.toggle("Debug Mode", value=st.session_state.debug_mode)

    st.divider()

    # Memory controls
    st.markdown("### 🧠 Memory")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Reset Session", use_container_width=True):
            st.session_state.memory_manager.reset_session()
            st.session_state.messages = []
            st.toast("Session memory cleared!", icon="🔄")
            st.rerun()

    with col2:
        if st.button("Reset All", use_container_width=True, type="secondary"):
            st.session_state.memory_manager.reset_all_memory()
            st.session_state.messages = []
            st.toast("All memory cleared!", icon="🗑️")
            st.rerun()

    # Memory stats
    stats = st.session_state.memory_manager.get_memory_stats()
    st.markdown(f"""
    **Active memories:**
    - 💬 Conversation: {stats['conversation_messages']} messages
    - 📝 Long-term: {stats['long_term_memories']} preferences
    """)

    if stats["long_term_details"]:
        with st.expander("View Stored Preferences"):
            for mem in stats["long_term_details"]:
                st.markdown(f"- **{mem['type']}**: {mem['content']}")

    st.divider()

    # Audit log
    st.markdown("### 📋 Recent Audit Log")
    try:
        from guardrails.audit_logger import get_recent_events
        events = get_recent_events(n=5)
        if events:
            for event in reversed(events):
                icon = "🟢" if event["outcome"] == "allowed" else "🔴" if event["outcome"] == "blocked" else "🟡"
                st.markdown(f"{icon} **{event['event_type']}** — {event['outcome']}")
        else:
            st.markdown("_No events yet_")
    except Exception:
        st.markdown("_Audit log unavailable_")


# ── Main chat area ──────────────────────────────────────────────────────────

st.markdown("""
<div class="main-header">
    <h1>🏥 City General Hospital AI Assistant</h1>
    <p>Ask about hospital services, departments, doctors, policies, pricing, and more.</p>
</div>
""", unsafe_allow_html=True)

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

        # Show debug info for assistant messages
        if msg["role"] == "assistant" and st.session_state.debug_mode and msg.get("debug"):
            with st.expander("🔍 Debug Info", expanded=False):
                debug = msg["debug"]

                if debug.get("intent"):
                    st.markdown(f"**Intent:** `{debug['intent']}`")

                if debug.get("rewritten_queries"):
                    st.markdown("**Rewritten queries:**")
                    for q in debug["rewritten_queries"]:
                        st.markdown(f"- `{q}`")

                if debug.get("self_check"):
                    verdict = debug["self_check"].get("verdict", "n/a")
                    icon = "✅" if verdict == "supported" else "⚠️" if verdict == "partially_supported" else "❌"
                    st.markdown(f"**Self-check:** {icon} `{verdict}`")

                if debug.get("chunks"):
                    st.markdown(f"**Retrieved chunks:** {len(debug['chunks'])}")
                    for i, chunk in enumerate(debug["chunks"][:3]):
                        st.markdown(f"  {i+1}. {chunk.get('title', 'Unknown')} (score: {chunk.get('score', 'n/a'):.3f})")

                if debug.get("trace"):
                    with st.expander("Full trace"):
                        st.json(debug["trace"])


# ── Chat input ──────────────────────────────────────────────────────────────

if prompt := st.chat_input("Type your question here..."):
    # Display user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Process through agent
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            from app.main import chat

            result = chat(
                prompt,
                graph=st.session_state.graph,
                memory_manager=st.session_state.memory_manager,
            )

        answer = result["answer"]
        st.markdown(answer)

        # Build debug info
        debug_info = {
            "intent": result.get("intent"),
            "self_check": result.get("self_check", {}),
            "rewritten_queries": [],
            "chunks": [],
            "trace": result.get("trace", []),
            "metrics": result.get("metrics"),
        }

        # Extract rewritten queries from trace
        for t in result.get("trace", []):
            if t.get("node") == "rewrite_query" and t.get("queries"):
                debug_info["rewritten_queries"] = t["queries"]
            if t.get("node") == "retrieve":
                debug_info["retrieve_info"] = t

        # Extract chunk info
        for chunk in result.get("retrieved_chunks", []):
            debug_info["chunks"].append({
                "title": chunk.title,
                "score": chunk.score,
                "source_id": chunk.source_id,
            })

        # Show debug panel
        if st.session_state.debug_mode:
            with st.expander("🔍 Debug Info", expanded=False):
                if debug_info.get("intent"):
                    st.markdown(f"**Intent:** `{debug_info['intent']}`")

                if debug_info.get("rewritten_queries"):
                    st.markdown("**Rewritten queries:**")
                    for q in debug_info["rewritten_queries"]:
                        st.markdown(f"- `{q}`")

                if debug_info.get("self_check"):
                    verdict = debug_info["self_check"].get("verdict", "n/a")
                    icon = "✅" if verdict == "supported" else "⚠️" if verdict == "partially_supported" else "❌"
                    st.markdown(f"**Self-check:** {icon} `{verdict}`")

                if debug_info.get("chunks"):
                    st.markdown(f"**Retrieved chunks:** {len(debug_info['chunks'])}")
                    for i, chunk in enumerate(debug_info["chunks"][:5]):
                        score_val = chunk.get('score', 0)
                        st.markdown(f"  {i+1}. {chunk.get('title', 'Unknown')} (score: {score_val:.3f})")

                # Runtime metrics display
                if debug_info.get("metrics"):
                    m = debug_info["metrics"]
                    st.markdown("**Runtime Metrics:**")
                    cols = st.columns(5)
                    cols[0].metric("LLM Calls", m.get("total-llm-calls", 0))
                    cols[1].metric("E2E Time", f"{m.get('e2e-response-time', 0)}s")
                    cols[2].metric("Mean Time", f"{m.get('mean-response-time', 0)}s")
                    cols[3].metric("Total Tokens", m.get("total-tokens", 0))
                    cols[4].metric("Tok/Call", m.get("mean-token-per-call", 0))

                with st.expander("Full trace"):
                    st.json(debug_info["trace"])

        # Save to messages
        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "debug": debug_info,
        })

    # Refresh sidebar stats
    st.rerun()
