"""
Test suite for retrieval quality.

Tests 15-20 sample queries with expected relevant doc_type / department.
Validates that the retrieval system returns relevant chunks.
"""

import pytest

# These tests require ChromaDB to be populated (run `python -m rag.ingest` first)
# They also require an embedding API key

pytestmark = pytest.mark.skipif(
    True,  # Set to False after running ingestion
    reason="Requires ChromaDB to be populated and embedding API key",
)


RETRIEVAL_TEST_CASES = [
    {
        "query": "What are the visiting hours?",
        "expected_doc_type": "faq",
        "expected_keywords": ["visiting", "hours", "10:00"],
    },
    {
        "query": "How do I schedule an appointment?",
        "expected_doc_type": "policy",
        "expected_keywords": ["appointment", "schedule", "portal"],
    },
    {
        "query": "What does the cardiology department do?",
        "expected_doc_type": "department_info",
        "expected_department": "cardiology",
    },
    {
        "query": "Tell me about Dr. Sarah Mitchell",
        "expected_doc_type": "doctor_bio",
        "expected_keywords": ["Mitchell", "cardiol"],
    },
    {
        "query": "What should I prepare before surgery?",
        "expected_doc_type": "policy",
        "expected_keywords": ["surgery", "preparation", "fasting"],
    },
    {
        "query": "How much does an MRI cost?",
        "expected_doc_type": "pricing",
        "expected_keywords": ["MRI", "800", "1500"],
    },
    {
        "query": "Does the hospital accept BlueCross insurance?",
        "expected_doc_type": "policy",
        "expected_keywords": ["BlueCross", "insurance"],
    },
    {
        "query": "What are my rights as a patient?",
        "expected_doc_type": "policy",
        "expected_keywords": ["rights", "patient"],
    },
    {
        "query": "When should I go to the emergency department?",
        "expected_doc_type": "service_info",
        "expected_keywords": ["emergency", "life-threatening"],
    },
    {
        "query": "How does telehealth work?",
        "expected_doc_type": "service_info",
        "expected_keywords": ["telehealth", "video"],
    },
    {
        "query": "Where can I park at the hospital?",
        "expected_doc_type": "policy",
        "expected_keywords": ["parking", "Lot"],
    },
    {
        "query": "Which doctor treats knee pain?",
        "expected_doc_type": "doctor_bio",
        "expected_department": "orthopedics",
    },
    {
        "query": "What is the cancellation policy for appointments?",
        "expected_doc_type": "policy",
        "expected_keywords": ["cancel", "24 hours"],
    },
    {
        "query": "Is there a pharmacy in the hospital?",
        "expected_doc_type": "faq",
        "expected_keywords": ["pharmacy", "ground floor"],
    },
    {
        "query": "What payment plans are available?",
        "expected_doc_type": "pricing",
        "expected_keywords": ["payment", "plan", "interest-free"],
    },
]


class TestRetrieval:
    """Test retrieval quality with sample queries."""

    def test_retrieval_returns_results(self):
        """All test queries should return at least 1 chunk."""
        from rag.retriever import retrieve

        for case in RETRIEVAL_TEST_CASES:
            chunks = retrieve(case["query"], top_k=5)
            assert len(chunks) > 0, f"No results for: {case['query']}"

    def test_retrieval_relevance_by_doc_type(self):
        """Top result should match expected doc_type."""
        from rag.retriever import retrieve

        correct = 0
        total = 0

        for case in RETRIEVAL_TEST_CASES:
            if "expected_doc_type" not in case:
                continue
            total += 1
            chunks = retrieve(case["query"], top_k=3)
            doc_types = [c.metadata.get("doc_type") for c in chunks]
            if case["expected_doc_type"] in doc_types:
                correct += 1

        precision = correct / total if total > 0 else 0
        print(f"\nRetrieval precision (doc_type match in top-3): {correct}/{total} = {precision:.2%}")
        assert precision >= 0.6, f"Retrieval precision too low: {precision:.2%}"

    def test_retrieval_keyword_presence(self):
        """Retrieved chunks should contain expected keywords."""
        from rag.retriever import retrieve

        correct = 0
        total = 0

        for case in RETRIEVAL_TEST_CASES:
            if "expected_keywords" not in case:
                continue
            total += 1
            chunks = retrieve(case["query"], top_k=5)
            all_text = " ".join(c.text for c in chunks).lower()
            keywords_found = sum(1 for kw in case["expected_keywords"] if kw.lower() in all_text)
            if keywords_found >= len(case["expected_keywords"]) * 0.5:
                correct += 1

        recall = correct / total if total > 0 else 0
        print(f"\nKeyword recall: {correct}/{total} = {recall:.2%}")
        assert recall >= 0.6, f"Keyword recall too low: {recall:.2%}"

    def test_retrieval_with_metadata_filter(self):
        """Test that metadata filters work correctly."""
        from rag.retriever import retrieve

        chunks = retrieve("hospital information", top_k=5, filters={"doc_type": "faq"})
        for chunk in chunks:
            assert chunk.metadata.get("doc_type") == "faq", \
                f"Filter violation: got {chunk.metadata.get('doc_type')}"
