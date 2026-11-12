"""Unit tests for the YAML-driven grading rule engine."""

import pytest
from pathlib import Path
from src.rules import GradingRuleEngine, GradingResult, LotSummary, load_grading_rules


@pytest.fixture
def engine():
    return load_grading_rules()


def test_placeholder_flag_and_disclaimer(engine):
    assert engine.is_placeholder is True
    assert "0.1-placeholder" in engine.rules_version
    assert "PLACEHOLDER" in engine.disclaimer.upper()


def test_grade_a_healthy_in_range(engine):
    # Standard Grade A range is 45.0 - 65.0 mm
    res = engine.grade_onion(onion_id=1, diameter_mm=55.0, defect_label="healthy", confidence=0.92)
    assert res.grade == "Grade A"
    assert "Grade A" in res.reason
    assert res.needs_review is False


def test_urs_undersize_healthy(engine):
    # Undersize 35.0 - 44.9 mm is URS
    res = engine.grade_onion(onion_id=2, diameter_mm=40.0, defect_label="healthy", confidence=0.88)
    assert res.grade == "URS"
    assert "undersized" in res.reason.lower()


def test_urs_oversize_healthy(engine):
    # Oversize 65.1 - 75.0 mm is URS
    res = engine.grade_onion(onion_id=3, diameter_mm=70.0, defect_label="healthy", confidence=0.85)
    assert res.grade == "URS"
    assert "oversized" in res.reason.lower()


def test_reject_extreme_undersize(engine):
    # < 35.0 mm is Reject cull
    res = engine.grade_onion(onion_id=4, diameter_mm=30.0, defect_label="healthy", confidence=0.90)
    assert res.grade == "Reject"
    assert "undersize" in res.reason.lower()


def test_reject_extreme_oversize(engine):
    # > 75.0 mm is Reject
    res = engine.grade_onion(onion_id=5, diameter_mm=80.0, defect_label="healthy", confidence=0.90)
    assert res.grade == "Reject"
    assert "oversize" in res.reason.lower()


def test_rotten_unconditional_reject(engine):
    # Even if size is perfect 55mm, rotten -> Reject
    res = engine.grade_onion(onion_id=6, diameter_mm=55.0, defect_label="rotten", confidence=0.95)
    assert res.grade == "Reject"
    assert "rotten" in res.reason.lower()


def test_sprouted_defect_maps_to_urs(engine):
    # Sprouted -> URS
    res = engine.grade_onion(onion_id=7, diameter_mm=55.0, defect_label="sprouted", confidence=0.80)
    assert res.grade == "URS"
    assert "sprouted" in res.reason.lower()


def test_damaged_defect_maps_to_urs(engine):
    # Damaged -> URS
    res = engine.grade_onion(onion_id=8, diameter_mm=52.0, defect_label="damaged", confidence=0.75)
    assert res.grade == "URS"
    assert "damage" in res.reason.lower()


def test_low_confidence_needs_review(engine):
    # Confidence 0.35 is below 0.50 threshold
    res = engine.grade_onion(onion_id=9, diameter_mm=55.0, defect_label="healthy", confidence=0.35)
    assert res.grade == "Needs Human Review"
    assert res.needs_review is True
    assert "confidence" in res.reason.lower()


def test_lot_summary_percentages(engine):
    results = [
        engine.grade_onion(1, 52.0, "healthy", 0.95),  # Grade A
        engine.grade_onion(2, 58.0, "healthy", 0.92),  # Grade A
        engine.grade_onion(3, 40.0, "healthy", 0.88),  # URS (undersize)
        engine.grade_onion(4, 55.0, "sprouted", 0.85), # URS (defect)
        engine.grade_onion(5, 55.0, "rotten", 0.90),   # Reject (rotten)
    ]
    summary = engine.summarize_lot(results)
    assert summary.total_count == 5
    assert summary.grade_a_count == 2
    assert summary.grade_a_pct == 40.0
    assert summary.urs_count == 2
    assert summary.urs_pct == 40.0
    assert summary.reject_count == 1
    assert summary.reject_pct == 20.0
    assert summary.is_placeholder is True
    assert summary.defect_counts["rotten"] == 1
    assert summary.defect_counts["sprouted"] == 1
    assert summary.defect_counts["healthy"] == 3


def test_empty_lot_summary(engine):
    summary = engine.summarize_lot([])
    assert summary.total_count == 0
    assert summary.grade_a_pct == 0.0
