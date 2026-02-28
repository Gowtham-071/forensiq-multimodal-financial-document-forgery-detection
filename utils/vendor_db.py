"""
utils/vendor_db.py
Vendor Enrollment Registry + Bill History Logger
SQLite-backed — auto-creates vendor_registry.db on first import
"""

import sqlite3
import json
import os
import re
from datetime import datetime
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DB_PATH  = BASE_DIR / "vendor_registry.db"


def _get_conn():
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Create tables if they don't exist and seed demo vendors."""
    conn = _get_conn()
    cur  = conn.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS vendors (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            company_name TEXT    NOT NULL,
            gst_numbers  TEXT    NOT NULL,   -- JSON list of valid GST strings
            amount_min   REAL    DEFAULT 0,
            amount_max   REAL    DEFAULT 9999999,
            bill_format  TEXT    DEFAULT 'Indian GST',  -- 'Indian GST' | 'Generic Invoice'
            notes        TEXT    DEFAULT '',
            enrolled_at  TEXT    DEFAULT (datetime('now','localtime'))
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS bill_history (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp     TEXT    DEFAULT (datetime('now','localtime')),
            company_name  TEXT    DEFAULT 'Unknown',
            gst_found     TEXT    DEFAULT '',
            bill_book_id  TEXT    DEFAULT '',   -- invoice number extracted by OCR
            verdict       TEXT    NOT NULL,     -- GENUINE / SUSPICIOUS / FRAUD
            fraud_score   REAL    DEFAULT 0.0,
            visual_score  REAL    DEFAULT 0.0,
            ocr_confidence REAL   DEFAULT 0.0,
            time_taken_sec REAL   DEFAULT 0.0
        )
    """)

    conn.commit()

    # Seed demo vendors if table is empty
    count = cur.execute("SELECT COUNT(*) FROM vendors").fetchone()[0]
    if count == 0:
        seed_demo_vendors(conn)

    conn.close()


def seed_demo_vendors(conn=None):
    """Pre-populate 4 mock companies for faculty demo."""
    close_after = conn is None
    if conn is None:
        conn = _get_conn()
    cur = conn.cursor()

    demos = [
        {
            "company_name": "Reliance Retail Ltd",
            "gst_numbers":  ["29AAACR5055K1ZK", "27AAACR5055K1ZK", "33AAACR5055K1ZK"],
            "amount_min":   500,
            "amount_max":   500000,
            "bill_format":  "Indian GST",
            "notes":        "Karnataka=29, Maharashtra=27, Tamil Nadu=33 — multi-state branches"
        },
        {
            "company_name": "Infosys BPM Ltd",
            "gst_numbers":  ["29AABCI1681G1ZF", "07AABCI1681G1ZF"],
            "amount_min":   1000,
            "amount_max":   1000000,
            "bill_format":  "Indian GST",
            "notes":        "Karnataka=29, Delhi=07"
        },
        {
            "company_name": "Tata Consultancy Services",
            "gst_numbers":  ["27AAACT2727Q1ZW", "33AAACT2727Q1ZW"],
            "amount_min":   5000,
            "amount_max":   5000000,
            "bill_format":  "Indian GST",
            "notes":        "Maharashtra=27, Tamil Nadu=33"
        },
        {
            "company_name": "SROIE Demo Company",
            "gst_numbers":  ["945-82-2137", "942-80-0517"],
            "amount_min":   100,
            "amount_max":   50000,
            "bill_format":  "Generic Invoice",
            "notes":        "CORD/SROIE-style Tax IDs for demo with included invoice images"
        },
    ]

    for d in demos:
        cur.execute("""
            INSERT INTO vendors (company_name, gst_numbers, amount_min, amount_max, bill_format, notes)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            d["company_name"],
            json.dumps(d["gst_numbers"]),
            d["amount_min"],
            d["amount_max"],
            d["bill_format"],
            d["notes"],
        ))

    conn.commit()
    if close_after:
        conn.close()
    print(f"✅ Seeded {len(demos)} demo vendors into vendor_registry.db")


# ── Public API ──────────────────────────────────────────────────────────────

def enroll_vendor(company_name: str, gst_list: list, amount_min: float,
                  amount_max: float, bill_format: str = "Indian GST", notes: str = "") -> int:
    """Register a company with their valid GST numbers. Returns new vendor ID."""
    conn = _get_conn()
    cur  = conn.cursor()
    cur.execute("""
        INSERT INTO vendors (company_name, gst_numbers, amount_min, amount_max, bill_format, notes)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (company_name, json.dumps(gst_list), amount_min, amount_max, bill_format, notes))
    conn.commit()
    new_id = cur.lastrowid
    conn.close()
    return new_id


def lookup_vendor_by_gst(gst_number: str) -> dict | None:
    """
    Check if a GST/Tax-ID belongs to any enrolled vendor.
    Returns the vendor row as a dict, or None if not found.
    """
    if not gst_number:
        return None
    gst_clean = gst_number.strip().upper()
    conn = _get_conn()
    cur  = conn.cursor()
    # Load all vendors and check JSON list in Python (SQLite JSON support varies)
    rows = cur.execute("SELECT * FROM vendors").fetchall()
    conn.close()
    for row in rows:
        registered_gsts = json.loads(row["gst_numbers"])
        if gst_clean in [g.strip().upper() for g in registered_gsts]:
            return dict(row)
    return None


def get_all_vendors() -> list:
    """Return all enrolled vendors as a list of dicts."""
    conn = _get_conn()
    rows = conn.execute("SELECT * FROM vendors ORDER BY enrolled_at DESC").fetchall()
    conn.close()
    result = []
    for row in rows:
        d = dict(row)
        d["gst_list"] = json.loads(d["gst_numbers"])  # parsed list for templates
        d["gst_count"] = len(d["gst_list"])
        result.append(d)
    return result


def log_bill(company_name: str, gst_found: str, bill_book_id: str,
             verdict: str, fraud_score: float, visual_score: float,
             ocr_confidence: float, time_taken_sec: float):
    """Insert a bill verification record into history."""
    conn = _get_conn()
    conn.execute("""
        INSERT INTO bill_history
            (company_name, gst_found, bill_book_id, verdict,
             fraud_score, visual_score, ocr_confidence, time_taken_sec)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (company_name, gst_found, bill_book_id, verdict,
          round(fraud_score, 4), round(visual_score, 4),
          round(ocr_confidence, 4), round(time_taken_sec, 3)))
    conn.commit()
    conn.close()


def get_bill_history(limit: int = 100) -> list:
    """Return recent bill history records."""
    conn = _get_conn()
    rows = conn.execute(
        "SELECT * FROM bill_history ORDER BY id DESC LIMIT ?", (limit,)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_dashboard_stats() -> dict:
    """Aggregate stats for the history dashboard."""
    conn  = _get_conn()
    cur   = conn.cursor()

    total      = cur.execute("SELECT COUNT(*) FROM bill_history").fetchone()[0]
    fraud_cnt  = cur.execute("SELECT COUNT(*) FROM bill_history WHERE verdict='FRAUD'").fetchone()[0]
    susp_cnt   = cur.execute("SELECT COUNT(*) FROM bill_history WHERE verdict='SUSPICIOUS'").fetchone()[0]
    genuine_cnt= cur.execute("SELECT COUNT(*) FROM bill_history WHERE verdict='GENUINE'").fetchone()[0]
    avg_time   = cur.execute("SELECT AVG(time_taken_sec) FROM bill_history").fetchone()[0] or 0
    vendors_ct = cur.execute("SELECT COUNT(*) FROM vendors").fetchone()[0]
    conn.close()

    fraud_rate    = round((fraud_cnt / total * 100), 1) if total > 0 else 0.0
    manual_sec    = 9 * 60  # 9 minutes average manual audit
    time_saved_hr = round((total * (manual_sec - avg_time)) / 3600, 2) if total > 0 else 0.0

    return {
        "total_bills":     total,
        "fraud_count":     fraud_cnt,
        "suspicious_count":susp_cnt,
        "genuine_count":   genuine_cnt,
        "fraud_rate_pct":  fraud_rate,
        "avg_time_sec":    round(avg_time, 2),
        "time_saved_hr":   time_saved_hr,
        "enrolled_vendors":vendors_ct,
    }


def get_bills_over_time() -> list:
    """Return daily bill counts for the bills-over-time chart."""
    conn = _get_conn()
    rows = conn.execute("""
        SELECT DATE(timestamp) as day,
               COUNT(*) as total,
               SUM(CASE WHEN verdict='FRAUD' THEN 1 ELSE 0 END) as fraud
        FROM bill_history
        GROUP BY DATE(timestamp)
        ORDER BY day ASC
    """).fetchall()
    conn.close()
    return [dict(r) for r in rows]


# Auto-init on import
init_db()
