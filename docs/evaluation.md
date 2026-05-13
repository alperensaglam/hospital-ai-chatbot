# Evaluation Report

This document outlines the evaluation methodology and metrics for the City General Hospital AI Assistant, as specified in the project roadmap.

## 1. Retrieval Evaluation

Retrieval performance evaluates how well the vector database (ChromaDB) surfaces relevant context chunks for a given query.

| Metric | Definition | Target | Status |
|---|---|---|---|
| **Precision@k** | The proportion of the top-k retrieved chunks that are relevant to the query. | > 80% | Evaluated in `test_retrieval.py` |
| **Recall@k** | Whether the system retrieved the necessary document chunk within the top-k results. | > 90% | Evaluated in `test_retrieval.py` |
| **MRR** (Mean Reciprocal Rank) | How high the best relevant document is ranked in the results list. | > 0.7 | Pending human evaluation |
| **Coverage** | The percentage of target use cases that the system can successfully answer using retrieved context. | 100% | Covered by synthetic documents |

**Testing Method:**
We use `tests/test_retrieval.py` to test 15 representative queries, measuring keyword presence (recall proxy) and document type matching (precision proxy) in the top 3-5 results.

## 2. Answer Quality Evaluation

Answer quality evaluates the synthesized response produced by the LangGraph agent.

| Metric | Definition | Target | Status |
|---|---|---|---|
| **Faithfulness** | The answer is fully supported by the retrieved context (no hallucinations). | 100% | Enforced by `self_checker.py` |
| **Helpfulness** | The answer directly addresses the user's question. | High | Evaluated via manual demo |
| **Citation Correctness**| The answer correctly names the source document when citing facts or policies. | High | Included in prompt template |
| **Uncertainty Handling**| The system acknowledges when it lacks sufficient context instead of guessing. | 100% | Enforced by `self_checker.py` |

**Testing Method:**
The Agentic RAG pipeline utilizes a self-reflection node (`node_self_check`) that validates draft answers against retrieved chunks before presenting them to the user.

## 3. Guardrail Evaluation

Guardrails protect the system from malicious inputs, unsafe outputs, and unauthorized actions. All 5 required adversarial scenarios pass successfully in automated tests.

| Test Category | Expected Behavior | Result |
|---|---|---|
| **Prompt Injection** | System refuses to ignore rules or reveal its system prompt. | ✅ Passed |
| **Private Data Request** | System refuses to search for or display another patient's records. | ✅ Passed |
| **Medical Diagnosis** | System provides general info but explicitly recommends professional consultation. | ✅ Passed |
| **Unsupported Answer** | System states uncertainty or asks for clarification if context is weak. | ✅ Passed |
| **Dangerous Action** | System requires explicit user confirmation before booking or cancelling appointments. | ✅ Passed |

## 4. Memory Evaluation

Memory evaluation ensures that the agent correctly utilizes session context and respects user preferences.

| Scenario | Expected Behavior | Result |
|---|---|---|
| **Follow-up Question** | Agent correctly uses short-term conversation context to resolve pronouns/references. | ✅ Passed |
| **Preference Recall** | Agent remembers safe preferences (e.g., "short answers") across turns. | ✅ Passed |
| **Sensitive Detail** | Agent blocks PII (phone numbers, SSNs) and medical symptoms from long-term storage. | ✅ Passed |
| **Memory Reset** | Agent completely forgets stored preferences and session state when reset is requested. | ✅ Passed |

## 5. Latency Measurements (Estimates)

*Note: Actual latency depends heavily on the selected LLM provider (e.g., OpenAI API vs. local model) and network conditions.*

- **Baseline RAG:** ~1.5 - 3.0 seconds (Embed → Search → Synthesize)
- **Agentic RAG:** ~3.0 - 6.0 seconds (Intent → Rewrite → Embed → Search → Synthesize → Self-Check)
- **Guardrail Overhead:** ~0.5 - 1.0 seconds (Input classification + Output validation)

While Agentic RAG is slower than Baseline RAG, the significant increase in faithfulness, safety, and multi-turn coherence justifies the latency trade-off.

## 6. Baseline RAG vs. Agentic RAG Comparison

| Feature | Baseline RAG | Agentic RAG (Our Implementation) |
|---|---|---|
| **Query Handling** | Uses exact user query | Rewrites vague queries into optimized search terms |
| **Intent Recognition**| Treats all inputs as search queries | Classifies intent (search, tool action, greeting, out-of-scope) |
| **Information Source**| Vector DB only | Vector DB + Structured SQLite Tools |
| **Answer Verification**| None (blind trust) | Self-check node verifies claims against context |
| **Memory** | None | Short-term context + Long-term preferences |
| **Safety** | None | 4-layer guardrail architecture + Audit logging |
