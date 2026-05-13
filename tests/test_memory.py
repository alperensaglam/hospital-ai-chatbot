"""
Test suite for memory system.

Tests from roadmap §14 Memory Evaluation:
  - Follow-up question uses short-term context
  - Preference recall across turns
  - Sensitive detail is NOT persisted
  - Memory reset clears stored preferences
"""

import pytest
from memory.memory_manager import MemoryManager
from memory.memory_store import ShortTermStore, LongTermStore
from memory.memory_policy import contains_sensitive_data, can_store_long_term


@pytest.fixture
def memory_manager(tmp_path):
    """Create a test memory manager with temp database."""
    db_path = tmp_path / "test_memory.db"
    return MemoryManager(db_path=str(db_path), default_user_id="test_user")


class TestShortTermMemory:
    """Test short-term conversation memory."""

    def test_add_and_retrieve_messages(self, memory_manager):
        """Messages should be stored and retrievable."""
        memory_manager.add_message("user", "Hello")
        memory_manager.add_message("assistant", "Hi there!")

        context = memory_manager.get_conversation_context()
        assert "Hello" in context
        assert "Hi there!" in context

    def test_conversation_ordering(self, memory_manager):
        """Messages should be in chronological order."""
        memory_manager.add_message("user", "First message")
        memory_manager.add_message("assistant", "First response")
        memory_manager.add_message("user", "Second message")

        context = memory_manager.get_conversation_context()
        first_pos = context.find("First message")
        second_pos = context.find("Second message")
        assert first_pos < second_pos

    def test_session_state(self, memory_manager):
        """Session state variables should persist within session."""
        memory_manager.set_session_state("selected_department", "cardiology")
        value = memory_manager.get_session_state("selected_department")
        assert value == "cardiology"

    def test_session_reset_clears_conversation(self, memory_manager):
        """Resetting session should clear conversation history."""
        memory_manager.add_message("user", "test message")
        memory_manager.reset_session()
        context = memory_manager.get_conversation_context()
        assert context == ""


class TestLongTermMemory:
    """Test long-term preference storage."""

    def test_store_safe_preference(self, memory_manager):
        """Safe preferences should be stored successfully."""
        stored, _ = memory_manager.try_store_preference(
            "language_preference", "User prefers English"
        )
        assert stored is True

        prefs = memory_manager.get_user_preferences()
        assert len(prefs) == 1
        assert prefs[0]["content"] == "User prefers English"

    def test_store_concise_preference(self, memory_manager):
        """Concise preference should be detected."""
        memory_manager.try_store_preference(
            "answer_length_preference", "User prefers concise, short answers"
        )
        assert memory_manager.prefers_concise() is True

    def test_block_sensitive_content(self, memory_manager):
        """Sensitive content should NOT be stored."""
        stored, reason = memory_manager.try_store_preference(
            "general_preference", "Patient has diagnosis of diabetes type 2"
        )
        assert stored is False
        assert "sensitive" in reason.lower() or "blocked" in reason.lower()

    def test_block_phone_number(self, memory_manager):
        """Phone numbers should NOT be stored."""
        stored, _ = memory_manager.try_store_preference(
            "general_preference", "Call me at (555) 123-4567"
        )
        assert stored is False

    def test_block_wrong_type(self, memory_manager):
        """Invalid memory types should be rejected."""
        stored, _ = memory_manager.try_store_preference(
            "medical_record", "Some medical data"
        )
        assert stored is False


class TestMemoryReset:
    """Test memory deletion and reset."""

    def test_reset_clears_long_term(self, memory_manager):
        """Reset all should clear long-term preferences."""
        memory_manager.try_store_preference(
            "language_preference", "User prefers English"
        )
        memory_manager.try_store_preference(
            "answer_length_preference", "Short answers preferred"
        )

        count = memory_manager.reset_all_memory()
        assert count == 2

        prefs = memory_manager.get_user_preferences()
        assert len(prefs) == 0

    def test_reset_clears_short_term(self, memory_manager):
        """Reset all should also clear short-term memory."""
        memory_manager.add_message("user", "test")
        memory_manager.set_session_state("key", "value")

        memory_manager.reset_all_memory()

        context = memory_manager.get_conversation_context()
        assert context == ""
        assert memory_manager.get_session_state("key") is None


class TestMemoryStats:
    """Test memory statistics."""

    def test_stats_reflect_state(self, memory_manager):
        """Stats should accurately reflect memory state."""
        memory_manager.add_message("user", "msg1")
        memory_manager.add_message("assistant", "msg2")
        memory_manager.try_store_preference("language_preference", "English preferred")

        stats = memory_manager.get_memory_stats()
        assert stats["conversation_messages"] == 2
        assert stats["long_term_memories"] == 1
