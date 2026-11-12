"""SQLite Database for Onion Lot Inspections and Human Override Audit Logs."""

import sqlite3
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime

DEFAULT_DB_PATH = Path(__file__).resolve().parent.parent / "data" / "onion_grader.db"


class Database:
    """Manages persistence of inspection lots, individual onion classifications, and overrides."""

    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = Path(db_path) if db_path else DEFAULT_DB_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_tables()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def _init_tables(self) -> None:
        """Initializes tables for lots, onions, and human override audit trail."""
        with self._get_connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS lots (
                    lot_id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    image_path TEXT,
                    image_sha256 TEXT,
                    total_count INTEGER,
                    grade_a_count INTEGER,
                    grade_a_pct REAL,
                    urs_count INTEGER,
                    urs_pct REAL,
                    reject_count INTEGER,
                    reject_pct REAL,
                    needs_review_count INTEGER,
                    needs_review_pct REAL,
                    avg_diameter_mm REAL,
                    rules_version TEXT,
                    is_placeholder INTEGER,
                    report_path TEXT,
                    report_sha256 TEXT
                )
            """)

            conn.execute("""
                CREATE TABLE IF NOT EXISTS onions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    lot_id TEXT NOT NULL,
                    onion_id INTEGER NOT NULL,
                    diameter_mm REAL,
                    defect_label TEXT,
                    confidence REAL,
                    initial_grade TEXT,
                    initial_reason TEXT,
                    final_grade TEXT,
                    is_overridden INTEGER DEFAULT 0,
                    override_reason TEXT,
                    overridden_by TEXT,
                    overridden_at TEXT,
                    FOREIGN KEY (lot_id) REFERENCES lots (lot_id)
                )
            """)

            conn.execute("""
                CREATE TABLE IF NOT EXISTS overrides_audit (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    lot_id TEXT NOT NULL,
                    onion_id INTEGER NOT NULL,
                    old_grade TEXT NOT NULL,
                    new_grade TEXT NOT NULL,
                    reason TEXT NOT NULL,
                    operator TEXT,
                    timestamp TEXT NOT NULL,
                    FOREIGN KEY (lot_id) REFERENCES lots (lot_id)
                )
            """)
            conn.commit()

    def save_lot(self, lot_data: Dict[str, Any], onions_data: List[Dict[str, Any]]) -> None:
        """Persists a complete lot inspection and its constituent onions."""
        with self._get_connection() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO lots (
                    lot_id, created_at, image_path, image_sha256,
                    total_count, grade_a_count, grade_a_pct,
                    urs_count, urs_pct, reject_count, reject_pct,
                    needs_review_count, needs_review_pct,
                    avg_diameter_mm, rules_version, is_placeholder,
                    report_path, report_sha256
                ) VALUES (
                    :lot_id, :created_at, :image_path, :image_sha256,
                    :total_count, :grade_a_count, :grade_a_pct,
                    :urs_count, :urs_pct, :reject_count, :reject_pct,
                    :needs_review_count, :needs_review_pct,
                    :avg_diameter_mm, :rules_version, :is_placeholder,
                    :report_path, :report_sha256
                )
            """, lot_data)

            # Insert onions
            for o in onions_data:
                conn.execute("""
                    INSERT INTO onions (
                        lot_id, onion_id, diameter_mm, defect_label,
                        confidence, initial_grade, initial_reason,
                        final_grade, is_overridden, override_reason,
                        overridden_by, overridden_at
                    ) VALUES (
                        :lot_id, :onion_id, :diameter_mm, :defect_label,
                        :confidence, :initial_grade, :initial_reason,
                        :final_grade, :is_overridden, :override_reason,
                        :overridden_by, :overridden_at
                    )
                """, o)
            conn.commit()

    def record_override(
        self,
        lot_id: str,
        onion_id: int,
        new_grade: str,
        reason: str,
        operator: str = "Expert Auditor",
    ) -> bool:
        """Records a human override with a mandatory justification reason."""
        if not reason or not reason.strip():
            raise ValueError("A mandatory override justification reason must be provided.")

        now_str = datetime.utcnow().isoformat()

        with self._get_connection() as conn:
            cur = conn.execute(
                "SELECT final_grade FROM onions WHERE lot_id = ? AND onion_id = ?",
                (lot_id, onion_id),
            )
            row = cur.fetchone()
            if not row:
                return False
            old_grade = row["final_grade"]

            # Update onion record
            conn.execute("""
                UPDATE onions
                SET final_grade = ?, is_overridden = 1, override_reason = ?, overridden_by = ?, overridden_at = ?
                WHERE lot_id = ? AND onion_id = ?
            """, (new_grade, reason.strip(), operator, now_str, lot_id, onion_id))

            # Audit log
            conn.execute("""
                INSERT INTO overrides_audit (
                    lot_id, onion_id, old_grade, new_grade, reason, operator, timestamp
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (lot_id, onion_id, old_grade, new_grade, reason.strip(), operator, now_str))

            # Recalculate lot statistics
            cur2 = conn.execute("SELECT final_grade FROM onions WHERE lot_id = ?", (lot_id,))
            grades = [r["final_grade"] for r in cur2.fetchall()]
            total = len(grades)
            if total > 0:
                ga = sum(1 for g in grades if g == "Grade A")
                urs = sum(1 for g in grades if g == "URS")
                rej = sum(1 for g in grades if g == "Reject")
                rev = sum(1 for g in grades if g == "Needs Human Review")

                conn.execute("""
                    UPDATE lots
                    SET grade_a_count = ?, grade_a_pct = ?,
                        urs_count = ?, urs_pct = ?,
                        reject_count = ?, reject_pct = ?,
                        needs_review_count = ?, needs_review_pct = ?
                    WHERE lot_id = ?
                """, (
                    ga, round((ga / total) * 100.0, 1),
                    urs, round((urs / total) * 100.0, 1),
                    rej, round((rej / total) * 100.0, 1),
                    rev, round((rev / total) * 100.0, 1),
                    lot_id,
                ))

            conn.commit()
            return True

    def get_lot(self, lot_id: str) -> Optional[Dict[str, Any]]:
        with self._get_connection() as conn:
            cur = conn.execute("SELECT * FROM lots WHERE lot_id = ?", (lot_id,))
            row = cur.fetchone()
            return dict(row) if row else None

    def get_lot_onions(self, lot_id: str) -> List[Dict[str, Any]]:
        with self._get_connection() as conn:
            cur = conn.execute("SELECT * FROM onions WHERE lot_id = ? ORDER BY onion_id ASC", (lot_id,))
            return [dict(r) for r in cur.fetchall()]

    def get_overrides(self, lot_id: str) -> List[Dict[str, Any]]:
        with self._get_connection() as conn:
            cur = conn.execute("SELECT * FROM overrides_audit WHERE lot_id = ? ORDER BY timestamp ASC", (lot_id,))
            return [dict(r) for r in cur.fetchall()]

    def list_lots(self) -> List[Dict[str, Any]]:
        with self._get_connection() as conn:
            cur = conn.execute("SELECT * FROM lots ORDER BY created_at DESC LIMIT 50")
            return [dict(r) for r in cur.fetchall()]
