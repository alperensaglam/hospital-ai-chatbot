"""
Output guardrail — validates the final answer before sending to the user.

From roadmap §8:
  - Check grounding (are claims supported by context?)
  - Remove unsupported medical claims
  - Block sensitive data leakage (PHI)
  - Add uncertainty or escalation when needed
"""

from __future__ import annotations

import re

from guardrails.audit_logger import log_event

# Patterns that should NEVER appear in output
PHI_PATTERNS = [
    re.compile(r"\b\d{3}[-.\s]?\d{2}[-.\s]?\d{4}\b"),  # SSN
    re.compile(r"\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"),  # Phone (non-hospital)
    re.compile(r"\b[A-Za-z0-9._%+-]+@(?!citygeneralhospital)[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"),  # Non-hospital email
    re.compile(r"\bpatient\s+(?:id|ID|#|number)\s*[:\s]?\s*\w+", re.IGNORECASE),  # Patient ID references
    re.compile(r"\bmedical\s+record\s+(?:number|#|id)\s*[:\s]?\s*\w+", re.IGNORECASE),  # MRN
]

# Phrases that indicate the system might be giving medical advice
MEDICAL_ADVICE_PATTERNS = [
    "you should take",
    "i recommend taking",
    "your diagnosis is",
    "you have been diagnosed",
    "you are suffering from",
    "take this medication",
    "increase your dosage",
    "stop taking your medication",
    "you don't need to see a doctor",
    "this is definitely",
    "i can confirm you have",
]

MEDICAL_DISCLAIMER = (
    "\n\n📋 *This is general information only. Please consult a healthcare "
    "professional for medical advice specific to your situation.*"
)


def check_output(
    answer: str,
    user_id: str = "demo_user",
    add_medical_disclaimer: bool = True,
) -> dict:
    """
    Validate the output answer and apply safety modifications.

    Returns:
        dict with keys: 
            modified_answer (str), 
            modifications (list[str]),
            blocked (bool)
    """
    modifications: list[str] = []
    modified = answer
    blocked = False

    # 1. Check for PHI leakage
    for pattern in PHI_PATTERNS:
        matches = pattern.findall(modified)
        if matches:
            for match in matches:
                modified = modified.replace(match, "[REDACTED]")
            modifications.append(f"Redacted PHI matching pattern: {pattern.pattern}")

    # 2. Check for medical advice
    answer_lower = modified.lower()
    contains_medical_advice = False
    for phrase in MEDICAL_ADVICE_PATTERNS:
        if phrase in answer_lower:
            contains_medical_advice = True
            modifications.append(f"Detected medical advice pattern: '{phrase}'")
            break

    # 3. Add medical disclaimer if needed
    if contains_medical_advice and add_medical_disclaimer:
        if MEDICAL_DISCLAIMER not in modified:
            modified += MEDICAL_DISCLAIMER
            modifications.append("Added medical disclaimer")

    # 4. Check answer length (unusually long answers might indicate prompt leakage)
    if len(modified) > 5000:
        modified = modified[:5000] + "\n\n[Response truncated for safety]"
        modifications.append("Truncated unusually long response")

    # Log
    if modifications:
        log_event(
            "output_guardrail",
            user_id=user_id,
            outcome="modified",
            reason="; ".join(modifications),
            details={"modification_count": len(modifications)},
        )
    else:
        log_event(
            "output_guardrail",
            user_id=user_id,
            outcome="allowed",
            reason="No modifications needed",
        )

    return {
        "modified_answer": modified,
        "modifications": modifications,
        "blocked": blocked,
    }
