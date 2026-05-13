"""
Mock appointment API — simulates appointment booking/checking.

From roadmap §4 (out of scope for MVP):
  - No autonomous booking without explicit user confirmation
  - This is a mock that demonstrates the flow
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

# In-memory appointment store
_appointments: dict[str, dict[str, Any]] = {}


def check_availability(
    doctor_id: str,
    date: str,
    time_slot: str | None = None,
) -> dict:
    """
    Check if a doctor has availability on a given date.

    This is a MOCK — always returns synthetic availability.
    """
    # Simulate: doctors have 3 slots available per day
    available_slots = ["09:00", "11:00", "14:00"]
    if time_slot and time_slot not in available_slots:
        return {
            "available": False,
            "doctor_id": doctor_id,
            "date": date,
            "requested_slot": time_slot,
            "available_slots": available_slots,
            "message": f"The requested time slot {time_slot} is not available. Available slots: {', '.join(available_slots)}",
        }

    return {
        "available": True,
        "doctor_id": doctor_id,
        "date": date,
        "available_slots": available_slots,
        "message": f"Doctor has availability on {date}. Available slots: {', '.join(available_slots)}",
    }


def book_appointment(
    doctor_id: str,
    date: str,
    time_slot: str,
    patient_name: str = "Demo Patient",
    reason: str = "",
) -> dict:
    """
    Book an appointment (MOCK).

    In a real system, this would require authentication and confirmation.
    The tool_guardrail ensures this is only called after explicit user confirmation.
    """
    appointment_id = f"apt_{uuid.uuid4().hex[:8]}"

    appointment = {
        "appointment_id": appointment_id,
        "doctor_id": doctor_id,
        "date": date,
        "time_slot": time_slot,
        "patient_name": patient_name,
        "reason": reason,
        "status": "confirmed",
        "created_at": datetime.now(timezone.utc).isoformat(),
    }

    _appointments[appointment_id] = appointment

    return {
        "success": True,
        "appointment_id": appointment_id,
        "message": f"Appointment booked successfully for {date} at {time_slot}.",
        "details": appointment,
    }


def cancel_appointment(appointment_id: str) -> dict:
    """Cancel an appointment (MOCK)."""
    if appointment_id in _appointments:
        _appointments[appointment_id]["status"] = "cancelled"
        return {
            "success": True,
            "message": f"Appointment {appointment_id} has been cancelled.",
        }
    return {
        "success": False,
        "message": f"Appointment {appointment_id} not found.",
    }


def list_appointments(patient_name: str = "Demo Patient") -> list[dict]:
    """List all appointments for a patient (MOCK)."""
    return [
        apt for apt in _appointments.values()
        if apt.get("patient_name") == patient_name and apt.get("status") != "cancelled"
    ]
