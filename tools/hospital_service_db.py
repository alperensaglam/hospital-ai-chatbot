"""
Hospital service database — SQLite backend for structured data.

From roadmap §10:
  - Doctor schedules → SQL (needs exact, current data)
  - Appointment records → SQL
  - Pricing → SQL for structured queries

Seeds synthetic data on first use.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

DB_PATH = Path("data/hospital.db")


def get_connection() -> sqlite3.Connection:
    """Get a connection to the hospital database."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_database() -> None:
    """Create tables and seed synthetic data."""
    conn = get_connection()

    conn.executescript("""
        CREATE TABLE IF NOT EXISTS doctors (
            doctor_id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            department TEXT NOT NULL,
            specialty TEXT NOT NULL,
            accepting_new_patients INTEGER DEFAULT 1
        );

        CREATE TABLE IF NOT EXISTS schedules (
            schedule_id INTEGER PRIMARY KEY AUTOINCREMENT,
            doctor_id TEXT NOT NULL,
            day_of_week TEXT NOT NULL,
            start_time TEXT NOT NULL,
            end_time TEXT NOT NULL,
            slot_type TEXT DEFAULT 'regular',
            FOREIGN KEY (doctor_id) REFERENCES doctors(doctor_id)
        );

        CREATE TABLE IF NOT EXISTS pricing (
            service_id TEXT PRIMARY KEY,
            service_name TEXT NOT NULL,
            department TEXT,
            min_price REAL NOT NULL,
            max_price REAL NOT NULL,
            currency TEXT DEFAULT 'USD',
            notes TEXT
        );
    """)

    # Seed doctors
    doctors = [
        ("dr_mitchell", "Dr. Sarah Mitchell", "Cardiology", "Interventional Cardiology"),
        ("dr_park", "Dr. James Park", "Cardiology", "Electrophysiology"),
        ("dr_chen", "Dr. Emily Chen", "Orthopedics", "Joint Replacement"),
        ("dr_okonkwo", "Dr. Michael Okonkwo", "Orthopedics", "Sports Medicine"),
        ("dr_rahman", "Dr. Aisha Rahman", "Neurology", "Stroke & Epilepsy"),
        ("dr_kim", "Dr. Robert Kim", "Pediatrics", "General Pediatrics"),
        ("dr_fernandez", "Dr. Lisa Fernandez", "General Surgery", "Laparoscopic Surgery"),
        ("dr_nguyen", "Dr. David Nguyen", "Internal Medicine", "Chronic Disease Management"),
        ("dr_sharma", "Dr. Priya Sharma", "Radiology", "Diagnostic Imaging"),
        ("dr_weber", "Dr. Thomas Weber", "Emergency Medicine", "Trauma & Emergency"),
    ]
    for d in doctors:
        conn.execute(
            "INSERT OR IGNORE INTO doctors VALUES (?, ?, ?, ?, 1)", d
        )

    # Seed schedules
    schedules = [
        ("dr_mitchell", "Monday", "08:00", "12:00", "regular"),
        ("dr_mitchell", "Wednesday", "08:00", "12:00", "regular"),
        ("dr_mitchell", "Friday", "08:00", "12:00", "regular"),
        ("dr_park", "Tuesday", "09:00", "17:00", "regular"),
        ("dr_park", "Thursday", "09:00", "17:00", "regular"),
        ("dr_chen", "Monday", "08:00", "17:00", "regular"),
        ("dr_chen", "Tuesday", "08:00", "17:00", "regular"),
        ("dr_chen", "Wednesday", "08:00", "17:00", "regular"),
        ("dr_chen", "Thursday", "08:00", "17:00", "regular"),
        ("dr_chen", "Friday", "08:00", "17:00", "surgery"),
        ("dr_okonkwo", "Monday", "08:00", "17:00", "regular"),
        ("dr_okonkwo", "Tuesday", "08:00", "17:00", "regular"),
        ("dr_okonkwo", "Wednesday", "08:00", "17:00", "regular"),
        ("dr_okonkwo", "Thursday", "08:00", "11:00", "walk_in"),
        ("dr_rahman", "Monday", "08:30", "16:30", "regular"),
        ("dr_rahman", "Wednesday", "08:30", "16:30", "regular"),
        ("dr_rahman", "Friday", "08:30", "16:30", "regular"),
        ("dr_kim", "Monday", "08:00", "17:00", "regular"),
        ("dr_kim", "Tuesday", "08:00", "17:00", "regular"),
        ("dr_kim", "Wednesday", "08:00", "17:00", "regular"),
        ("dr_kim", "Thursday", "08:00", "17:00", "regular"),
        ("dr_kim", "Friday", "08:00", "17:00", "regular"),
        ("dr_fernandez", "Tuesday", "09:00", "17:00", "consultation"),
        ("dr_fernandez", "Thursday", "09:00", "17:00", "consultation"),
        ("dr_fernandez", "Monday", "07:00", "15:00", "surgery"),
        ("dr_fernandez", "Wednesday", "07:00", "15:00", "surgery"),
        ("dr_fernandez", "Friday", "07:00", "15:00", "surgery"),
        ("dr_nguyen", "Monday", "08:00", "11:00", "walk_in"),
        ("dr_nguyen", "Tuesday", "08:00", "17:00", "regular"),
        ("dr_nguyen", "Wednesday", "08:00", "17:00", "regular"),
        ("dr_nguyen", "Thursday", "08:00", "17:00", "regular"),
        ("dr_nguyen", "Friday", "08:00", "17:00", "regular"),
    ]
    # Clear and re-seed
    conn.execute("DELETE FROM schedules")
    for s in schedules:
        conn.execute(
            "INSERT INTO schedules (doctor_id, day_of_week, start_time, end_time, slot_type) VALUES (?, ?, ?, ?, ?)",
            s,
        )

    # Seed pricing
    pricing_data = [
        ("svc_pcp", "Primary Care Visit", "Internal Medicine", 150, 250, "USD", "Self-pay rate"),
        ("svc_specialist", "Specialist Consultation", None, 250, 400, "USD", "Cardiology, Neurology, Orthopedics"),
        ("svc_pediatric", "Pediatrics Well-Child Visit", "Pediatrics", 120, 200, "USD", None),
        ("svc_telehealth", "Telehealth Consultation", None, 100, 175, "USD", None),
        ("svc_ed_noncrit", "ED Visit (non-critical)", "Emergency", 500, 1500, "USD", None),
        ("svc_ed_crit", "ED Visit (critical/trauma)", "Emergency", 2000, 5000, "USD", None),
        ("svc_xray_single", "X-ray (single view)", "Radiology", 100, 200, "USD", None),
        ("svc_ct_no_contrast", "CT Scan (without contrast)", "Radiology", 500, 1000, "USD", None),
        ("svc_mri_no_contrast", "MRI (without contrast)", "Radiology", 800, 1500, "USD", None),
        ("svc_mri_contrast", "MRI (with contrast)", "Radiology", 1000, 2000, "USD", None),
        ("svc_mammography", "Mammography (screening)", "Radiology", 150, 300, "USD", None),
    ]
    for p in pricing_data:
        conn.execute(
            "INSERT OR IGNORE INTO pricing VALUES (?, ?, ?, ?, ?, ?, ?)", p
        )

    conn.commit()
    conn.close()


# ── Query functions (used as agent tools) ────────────────────────────────────


def search_doctors(
    department: str | None = None,
    specialty: str | None = None,
) -> list[dict]:
    """Search for doctors by department and/or specialty."""
    conn = get_connection()
    query = "SELECT * FROM doctors WHERE 1=1"
    params: list = []
    if department:
        query += " AND LOWER(department) LIKE ?"
        params.append(f"%{department.lower()}%")
    if specialty:
        query += " AND LOWER(specialty) LIKE ?"
        params.append(f"%{specialty.lower()}%")
    rows = conn.execute(query, params).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_doctor_schedule(
    doctor_id: str | None = None,
    doctor_name: str | None = None,
    day_of_week: str | None = None,
) -> list[dict]:
    """Get schedule for a specific doctor, optionally filtered by day."""
    conn = get_connection()
    query = """
        SELECT d.name, d.department, s.day_of_week, s.start_time, s.end_time, s.slot_type
        FROM schedules s
        JOIN doctors d ON s.doctor_id = d.doctor_id
        WHERE 1=1
    """
    params: list = []
    if doctor_id:
        query += " AND s.doctor_id = ?"
        params.append(doctor_id)
    if doctor_name:
        query += " AND LOWER(d.name) LIKE ?"
        params.append(f"%{doctor_name.lower()}%")
    if day_of_week:
        query += " AND LOWER(s.day_of_week) = ?"
        params.append(day_of_week.lower())
    query += " ORDER BY s.day_of_week, s.start_time"
    rows = conn.execute(query, params).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_pricing(
    service_name: str | None = None,
    department: str | None = None,
) -> list[dict]:
    """Look up pricing information."""
    conn = get_connection()
    query = "SELECT * FROM pricing WHERE 1=1"
    params: list = []
    if service_name:
        query += " AND LOWER(service_name) LIKE ?"
        params.append(f"%{service_name.lower()}%")
    if department:
        query += " AND LOWER(department) LIKE ?"
        params.append(f"%{department.lower()}%")
    rows = conn.execute(query, params).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def format_tool_results(results: list[dict], tool_name: str) -> str:
    """Format tool results into a readable string for the answer synthesizer."""
    if not results:
        return f"No results found for {tool_name}."

    lines = [f"Results from {tool_name}:"]
    for r in results:
        parts = [f"  {k}: {v}" for k, v in r.items() if v is not None]
        lines.append("\n".join(parts))
        lines.append("  ---")
    return "\n".join(lines)


# Initialize on import
init_database()
