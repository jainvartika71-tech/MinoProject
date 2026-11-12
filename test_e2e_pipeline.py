"""End-to-End integration test for the Onion Grading Automation Pipeline."""

import tempfile
from pathlib import Path
import pytest
import numpy as np

from src.app import create_synthetic_sample_image
from src.capture_check import check_capture_quality
from src.scale import detect_aruco_marker
from src.segment import OnionSegmenter
from src.defects import DefectClassifier
from src.rules import load_grading_rules
from src.report import InspectionReportGenerator, calculate_sha256
from src.db import Database


def test_end_to_end_lot_inspection_flow(tmp_path):
    # 1. Create a calibrated synthetic sample lot
    image_bgr = create_synthetic_sample_image(marker_id=0, dict_name="DICT_4X4_50")
    assert image_bgr is not None
    assert image_bgr.shape == (800, 1000, 3)

    # 2. Quality Pre-flight check
    quality_res = check_capture_quality(image_bgr, blur_threshold=50.0, require_marker=True)
    assert quality_res.passed is True
    assert quality_res.marker_found is True
    assert quality_res.is_blurry is False

    # 3. ArUco Marker Scale Calibration
    marker_res = detect_aruco_marker(image_bgr, target_id=0, physical_size_mm=50.0)
    assert marker_res.found is True
    assert abs(marker_res.pixels_per_mm - 4.0) < 0.2

    # 4. Onion Segmentation & Sizing
    segmenter = OnionSegmenter(model_path=None)
    detected_onions, annotated = segmenter.segment_and_measure(
        image_bgr,
        mm_per_pixel=marker_res.mm_per_pixel,
        marker_corners=marker_res.corner_points,
    )
    assert len(detected_onions) >= 5

    # 5. Defect Classification & Rule Engine Evaluation
    defect_classifier = DefectClassifier()
    engine = load_grading_rules()

    results = []
    for onion in detected_onions:
        x1, y1, x2, y2 = onion.bbox
        crop_mask = onion.mask[y1:y2, x1:x2] if onion.mask is not None else None
        defect, conf, _ = defect_classifier.classify_crop(
            onion.crop, onion.defect_label, mask=crop_mask
        )
        onion.defect_label = defect
        res = engine.grade_onion(
            onion_id=onion.onion_id,
            diameter_mm=onion.minor_axis_mm,
            defect_label=defect,
            confidence=conf,
        )
        results.append(res)

    summary = engine.summarize_lot(results)
    assert summary.total_count == len(detected_onions)
    assert summary.grade_a_count > 0
    assert (summary.grade_a_count + summary.urs_count + summary.reject_count + summary.needs_review_count) == summary.total_count
    assert abs((summary.grade_a_pct + summary.urs_pct + summary.reject_pct + summary.needs_review_pct) - 100.0) < 0.5

    # 6. Database Storage & Human Override
    db = Database(db_path=tmp_path / "test_e2e.db")
    lot_id = "LOT-E2E-TEST-001"
    lot_dict = {
        "lot_id": lot_id,
        "created_at": "2026-11-11T12:00:00",
        "image_path": str(tmp_path / "sample.png"),
        "image_sha256": calculate_sha256(image_bgr),
        "total_count": summary.total_count,
        "grade_a_count": summary.grade_a_count,
        "grade_a_pct": summary.grade_a_pct,
        "urs_count": summary.urs_count,
        "urs_pct": summary.urs_pct,
        "reject_count": summary.reject_count,
        "reject_pct": summary.reject_pct,
        "needs_review_count": summary.needs_review_count,
        "needs_review_pct": summary.needs_review_pct,
        "avg_diameter_mm": summary.avg_diameter_mm,
        "rules_version": summary.rules_version,
        "is_placeholder": 1 if summary.is_placeholder else 0,
        "report_path": "",
        "report_sha256": "",
    }
    onions_dict = [
        {
            "lot_id": lot_id,
            "onion_id": r.onion_id,
            "diameter_mm": r.diameter_mm,
            "defect_label": r.defect_label,
            "confidence": r.confidence,
            "initial_grade": r.grade,
            "initial_reason": r.reason,
            "final_grade": r.grade,
            "is_overridden": 0,
            "override_reason": None,
            "overridden_by": None,
            "overridden_at": None,
        }
        for r in results
    ]
    db.save_lot(lot_dict, onions_dict)

    # Perform override
    override_ok = db.record_override(
        lot_id=lot_id,
        onion_id=1,
        new_grade="Reject",
        reason="E2E manual auditor test override",
        operator="Quality Inspector",
    )
    assert override_ok is True
    overrides = db.get_overrides(lot_id)
    assert len(overrides) == 1

    # 7. PDF Report Generation
    report_gen = InspectionReportGenerator(output_dir=tmp_path)
    pdf_path, pdf_hash = report_gen.generate_pdf(
        lot_id=lot_id,
        lot_summary=summary.__dict__,
        onions=onions_dict,
        annotated_image=annotated,
        image_sha256=lot_dict["image_sha256"],
        overrides=overrides,
    )
    assert pdf_path.exists()
    assert pdf_path.stat().st_size > 5000
    assert len(pdf_hash) == 64
