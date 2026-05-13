"""
Intent classifier — determines the type of user request.

Intent categories (from roadmap §6):
  - policy_info:        Hospital policies, procedures, patient rights
  - service_info:       Department descriptions, services offered
  - doctor_info:        Doctor biographies, specialties, availability
  - pricing:            Costs, fees, payment options
  - appointment_action: Booking, cancelling, rescheduling appointments
  - general_greeting:   Hellos, thanks, small talk
  - out_of_scope:       Unrelated to hospital (politics, recipes, etc.)
"""

from __future__ import annotations

import json

from agents.llm import llm_call_json

VALID_INTENTS = [
    "policy_info",
    "service_info",
    "doctor_info",
    "pricing",
    "appointment_action",
    "general_greeting",
    "out_of_scope",
]

SYSTEM_PROMPT = """\
You are an intent classifier for a hospital AI assistant.

Given a user message, classify it into exactly ONE of these intents:
- policy_info: questions about hospital policies, procedures, rules, patient rights, insurance, visiting hours
- service_info: questions about departments, services, facilities, emergency, telehealth
- doctor_info: questions about specific doctors, their specialties, availability, biographies
- pricing: questions about costs, fees, payment plans, billing
- appointment_action: requests to book, cancel, reschedule, or check appointment availability
- general_greeting: greetings, thanks, goodbyes, small talk
- out_of_scope: topics unrelated to the hospital (politics, recipes, weather, etc.)

Respond ONLY with a JSON object in this format:
{"intent": "<intent>", "confidence": <0.0-1.0>, "reasoning": "<brief explanation>"}
"""


def classify_intent(user_message: str, conversation_context: str = "") -> dict:
    """
    Classify the user message into an intent category.

    Returns:
        dict with keys: intent, confidence, reasoning
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
    ]
    if conversation_context:
        messages.append(
            {
                "role": "user",
                "content": f"Conversation context:\n{conversation_context}\n\nCurrent user message: {user_message}",
            }
        )
    else:
        messages.append({"role": "user", "content": user_message})

    raw = llm_call_json(messages, temperature=0.0)

    try:
        result = json.loads(raw)
    except json.JSONDecodeError:
        result = {"intent": "out_of_scope", "confidence": 0.0, "reasoning": "Failed to parse LLM response"}

    # Validate intent
    if result.get("intent") not in VALID_INTENTS:
        result["intent"] = "out_of_scope"
        result["confidence"] = 0.0

    return result
