"""Unit tests for SQLite database persistence, human overrides, and PDF report generation."""

import tempfile
from pathlib import Path
import pytest
import numpy as np

from src.db import Database
from src.report import InspectionReportGenerator, calculate_sha256


@pytest.fixture
def temp_db(tmp_path):
    db_file = tmp_path / "test_grader.db"
    return Database(db_path=db_file)


def test_db_save_and_retrieve_lot(temp_db):
    lot_data = {
        "lot_id": "LOT-TEST-101",
        "created_at": "2026-11-11T10:00:00",
        "image_path": "/fake/path.jpg",
        "image_sha256": "abcdef1234567890",
        "total_count": 3,
        "grade_a_count": 2,
        "grade_a_pct": 66.7,
        "urs_count": 1,
        "urs_pct": 33.3,
        "reject_count": 0,
        "reject_pct": 0.0,
        "needs_review_count": 0,
        "needs_review_pct": 0.0,
        "avg_diameter_mm": 52.3,
        "rules_version": "0.1-placeholder",
        "is_placeholder": 1,
        "report_path": "",
        "report_sha256": "",
    }
    onions_data = [
        {
            "lot_id": "LOT-TEST-101",
            "onion_id": 1,
            "diameter_mm": 52.0,
            "defect_label": "healthy",
            "confidence": 0.95,
            "initial_grade": "Grade A",
            "initial_reason": "Optimal size",
            "final_grade": "Grade A",
            "is_overridden": 0,
            "override_reason": None,
            "overridden_by": None,
            "overridden_at": None,
        },
        {
            "lot_id": "LOT-TEST-101",
            "onion_id": 2,
            "diameter_mm": 58.0,
            "defect_label": "healthy",
            "confidence": 0.92,
            "initial_grade": "Grade A",
            "initial_reason": "Optimal size",
            "final_grade": "Grade A",
            "is_overridden": 0,
            "override_reason": None,
            "overridden_by": None,
            "overridden_at": None,
        },
        {
            "lot_id": "LOT-TEST-101",
            "onion_id": 3,
            "diameter_mm": 40.0,
            "defect_label": "healthy",
            "confidence": 0.88,
            "initial_grade": "URS",
            "initial_reason": "Undersize",
            "final_grade": "URS",
            "is_overridden": 0,
            "override_reason": None,
            "overridden_by": None,
            "overridden_at": None,
        },
    ]

    temp_db.save_lot(lot_data, onions_data)
    retrieved_lot = temp_db.get_lot("LOT-TEST-101")
    assert retrieved_lot is not None
    assert retrieved_lot["total_count"] == 3
    assert retrieved_lot["grade_a_pct"] == 66.7

    retrieved_onions = temp_db.get_lot_onions("LOT-TEST-101")
    assert len(retrieved_onions) == 3


def test_human_override_with_mandatory_reason(temp_db):
    test_db_save_and_retrieve_lot(temp_db)

    # Empty reason must raise ValueError
    with pytest.raises(ValueError):
        temp_db.record_override("LOT-TEST-101", onion_id=1, new_grade="Reject", reason="")

    # Valid override
    success = temp_db.record_override(
        lot_id="LOT-TEST-101",
        onion_id=1,
        new_grade="Reject",
        reason="Detected concealed basal rot not caught by camera",
        operator="Chief Inspector",
    )
    assert success is True

    # Check updated onion
    onions = temp_db.get_lot_onions("LOT-TEST-101")
    o1 = [o for o in onions if o["onion_id"] == 1][0]
    assert o1["final_grade"] == "Reject"
    assert o1["is_overridden"] == 1
    assert "concealed basal rot" in o1["override_reason"]

    # Check updated lot percentages (1 Grade A, 1 URS, 1 Reject => 33.3% each)
    updated_lot = temp_db.get_lot("LOT-TEST-101")
    assert updated_lot["grade_a_count"] == 1
    assert updated_lot["reject_count"] == 1
    assert updated_lot["grade_a_pct"] == 33.3

    # Check audit log
    overrides = temp_db.get_overrides("LOT-TEST-101")
    assert len(overrides) == 1
    assert overrides[0]["old_grade"] == "Grade A"
    assert overrides[0]["new_grade"] == "Reject"


def test_pdf_report_generation(tmp_path):
    generator = InspectionReportGenerator(output_dir=tmp_path)
    lot_summary = {
        "total_count": 2,
        "grade_a_count": 1,
        "grade_a_pct": 50.0,
        "urs_count": 1,
        "urs_pct": 50.0,
        "reject_count": 0,
        "reject_pct": 0.0,
        "needs_review_count": 0,
        "needs_review_pct": 0.0,
        "avg_diameter_mm": 51.5,
        "rules_version": "0.1-placeholder",
        "is_placeholder": True,
    }
    onions = [
        {"onion_id": 1, "diameter_mm": 52.0, "defect_label": "healthy", "confidence": 0.95, "final_grade": "Grade A", "initial_reason": "Optimal size"},
        {"onion_id": 2, "diameter_mm": 41.0, "defect_label": "sprouted", "confidence": 0.88, "final_grade": "URS", "initial_reason": "Sprout blemish"},
    ]
    dummy_annotated = np.ones((200, 300, 3), dtype=np.uint8) * 180

    pdf_path, pdf_hash = generator.generate_pdf(
        lot_id="LOT-PDF-1",
        lot_summary=lot_summary,
        onions=onions,
        annotated_image=dummy_annotated,
        image_sha256="test_sha_image_123",
    )

    assert pdf_path.exists()
    assert pdf_path.stat().st_size > 2000
    assert len(pdf_hash) == 64  # SHA-256 length
