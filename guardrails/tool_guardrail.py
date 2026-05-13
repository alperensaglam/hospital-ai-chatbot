"""
Tool guardrail — controls structured DB/API actions.

From roadmap §8:
  - Booking/cancellation requires explicit confirmation
  - Wraps tool calls with a confirmation step
"""

from __future__ import annotations

from guardrails.audit_logger import log_event

# Actions that require explicit user confirmation
CONFIRMATION_REQUIRED_ACTIONS = {
    "book_appointment",
    "cancel_appointment",
    "reschedule_appointment",
    "update_patient_info",
}

# Actions that are read-only and safe
SAFE_ACTIONS = {
    "check_availability",
    "get_doctor_info",
    "get_department_info",
    "get_pricing",
    "search_schedule",
}


def check_tool_action(
    action_name: str,
    action_params: dict | None = None,
    user_confirmed: bool = False,
    user_id: str = "demo_user",
) -> dict:
    """
    Check whether a tool action is allowed to proceed.

    Args:
        action_name: Name of the tool action.
        action_params: Parameters for the action.
        user_confirmed: Whether the user has explicitly confirmed.
        user_id: For audit logging.

    Returns:
        dict with keys: allowed, needs_confirmation, message
    """
    if action_name in SAFE_ACTIONS:
        log_event(
            "tool_guardrail",
            user_id=user_id,
            outcome="allowed",
            reason=f"Safe read-only action: {action_name}",
            details={"action": action_name, "params": action_params},
        )
        return {
            "allowed": True,
            "needs_confirmation": False,
            "message": None,
        }

    if action_name in CONFIRMATION_REQUIRED_ACTIONS:
        if user_confirmed:
            log_event(
                "tool_guardrail",
                user_id=user_id,
                outcome="allowed",
                reason=f"User confirmed action: {action_name}",
                details={"action": action_name, "params": action_params},
            )
            return {
                "allowed": True,
                "needs_confirmation": False,
                "message": None,
            }
        else:
            log_event(
                "tool_guardrail",
                user_id=user_id,
                outcome="blocked",
                reason=f"Action requires confirmation: {action_name}",
                details={"action": action_name, "params": action_params},
            )
            return {
                "allowed": False,
                "needs_confirmation": True,
                "message": (
                    f"I need your explicit confirmation before proceeding with "
                    f"'{action_name.replace('_', ' ')}'. "
                    f"Would you like me to go ahead? Please confirm with 'yes' or 'no'."
                ),
            }

    # Unknown action — block by default
    log_event(
        "tool_guardrail",
        user_id=user_id,
        outcome="blocked",
        reason=f"Unknown action blocked: {action_name}",
        details={"action": action_name},
    )
    return {
        "allowed": False,
        "needs_confirmation": False,
        "message": "This action is not supported by the system.",
    }
