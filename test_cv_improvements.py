"""Unit and integration tests for CV segmentation, touching onion separation, and calibration gating."""

import pytest
import numpy as np
import cv2

from src.segment import OnionSegmenter, DetectedOnion
from src.scale import detect_aruco_marker, generate_aruco_marker, is_valid_calibration
from src.rules import load_grading_rules


def create_synthetic_red_onion(canvas, cx, cy, rx, ry, ang=0, with_highlight=True):
    """Draws a realistic red onion with red-purple skin hue and optional specular highlight."""
    # Red onion BGR: high red, low green, moderate blue (e.g. B: 70, G: 35, R: 160 -> Hue in HSV ~165-175)
    cv2.ellipse(canvas, (cx, cy), (rx, ry), ang, 0, 360, (70, 35, 160), -1, cv2.LINE_AA)
    # Inner gradient
    cv2.ellipse(canvas, (cx, cy), (int(rx * 0.85), int(ry * 0.85)), ang, 0, 360, (85, 45, 180), -1, cv2.LINE_AA)
    # Dark outer border
    cv2.ellipse(canvas, (cx, cy), (rx, ry), ang, 0, 360, (50, 20, 120), 2, cv2.LINE_AA)

    if with_highlight:
        # Specular reflection (white shine spot) near top-left of bulb
        hx, hy = cx - int(rx * 0.35), cy - int(ry * 0.35)
        cv2.circle(canvas, (hx, hy), max(4, int(rx * 0.15)), (240, 240, 250), -1, cv2.LINE_AA)


def test_separate_touching_adjacent_onions():
    """Requirement A: Two adjacent touching onions must NOT be merged into one."""
    h, w = 600, 800
    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    canvas[:, :] = (40, 40, 40)  # Neutral dark mat

    # Onion 1: center (300, 300), radius 80 px (~40mm at 4 px/mm)
    create_synthetic_red_onion(canvas, 300, 300, 80, 80, with_highlight=False)

    # Onion 2: center (450, 300), radius 80 px (distance between centers = 150px < 160px -> touching/overlapping by 10px!)
    create_synthetic_red_onion(canvas, 450, 300, 80, 80, with_highlight=False)

    segmenter = OnionSegmenter(model_path=None)
    mm_per_pixel = 0.25  # 4 px per mm
    onions, _ = segmenter.segment_and_measure(canvas, mm_per_pixel=mm_per_pixel, is_calibrated=True)

    # Must detect exactly 2 onions, NOT 1 merged blob!
    assert len(onions) == 2, f"Expected 2 separated onions, got {len(onions)}"

    # Check that measured diameters are plausible (each should be ~40 mm, NOT ~75 mm / 137 mm)
    diameters = sorted([o.minor_axis_mm for o in onions])
    assert abs(diameters[0] - 40.0) < 4.0
    assert abs(diameters[1] - 40.0) < 4.0


def test_eliminate_internal_duplicate_contours():
    """Requirement A & C: Specular reflections or internal flakes must NOT be detected as separate small onions."""
    h, w = 500, 500
    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    canvas[:, :] = (35, 35, 35)

    # Single bulb at (250, 250) with prominent specular highlight and inner streak
    create_synthetic_red_onion(canvas, 250, 250, 90, 95, with_highlight=True)
    # Add a secondary dark root/peel streak inside
    cv2.line(canvas, (230, 240), (270, 260), (30, 15, 60), 3)

    segmenter = OnionSegmenter(model_path=None)
    onions, _ = segmenter.segment_and_measure(canvas, mm_per_pixel=0.25, is_calibrated=True)

    # Must detect exactly 1 onion, NOT 2 or 3!
    assert len(onions) == 1, f"Expected 1 bulb without internal fragments, got {len(onions)}"
    assert onions[0].minor_axis_mm > 40.0


def test_reject_false_positive_shadows_and_jagged_shapes():
    """Requirement A & C: Shadows, non-bulbous strips, and tiny noise must be rejected."""
    h, w = 600, 600
    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    canvas[:, :] = (35, 35, 35)

    # 1. Valid red onion
    create_synthetic_red_onion(canvas, 300, 300, 85, 85, with_highlight=False)

    # 2. Jagged shadow polygon (low solidity < 0.70)
    shadow_pts = np.array([[50, 50], [150, 60], [120, 180], [80, 100], [50, 180]], dtype=np.int32)
    cv2.fillPoly(canvas, [shadow_pts], (20, 20, 20))

    # 3. Elongated thin strip (aspect ratio < 0.35)
    cv2.rectangle(canvas, (400, 50), (420, 250), (60, 30, 140), -1)

    # 4. Tiny noise blob (< 150 px)
    cv2.circle(canvas, (500, 500), 5, (70, 35, 160), -1)

    segmenter = OnionSegmenter(model_path=None)
    onions, _ = segmenter.segment_and_measure(canvas, mm_per_pixel=0.25, is_calibrated=True)

    # Only the genuine bulb should pass
    assert len(onions) == 1, f"Expected only 1 valid onion, got {len(onions)}"
    assert onions[0].solidity >= 0.78
    assert onions[0].aspect_ratio >= 0.45


def test_missing_aruco_marker_uncalibrated_behavior():
    """Requirement B: When marker is missing, do NOT invent physical dimensions."""
    h, w = 500, 500
    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    canvas[:, :] = (35, 35, 35)
    create_synthetic_red_onion(canvas, 250, 250, 80, 80)

    # Detect marker on image without marker
    marker_res = detect_aruco_marker(canvas, target_id=0)
    assert marker_res.found is False
    assert marker_res.is_calibrated is False
    assert marker_res.pixels_per_mm is None
    assert is_valid_calibration(marker_res) is False

    # Segment and measure with uncalibrated parameters
    segmenter = OnionSegmenter(model_path=None)
    onions, _ = segmenter.segment_and_measure(
        canvas,
        mm_per_pixel=marker_res.mm_per_pixel,
        is_calibrated=marker_res.is_calibrated,
    )

    assert len(onions) == 1
    onion = onions[0]
    assert onion.is_calibrated is False
    # minor_axis_mm must be None, NOT an invented number like 20.1 mm or 137 mm!
    assert onion.minor_axis_mm is None
    assert onion.minor_axis_px > 0

    # Rule engine must grade uncalibrated healthy bulb as "Needs Calibration"
    engine = load_grading_rules()
    res = engine.grade_onion(
        onion_id=onion.onion_id,
        diameter_mm=onion.minor_axis_mm,
        defect_label=onion.defect_label,
        confidence=onion.confidence,
        is_calibrated=onion.is_calibrated,
    )
    assert res.grade == "Needs Calibration"
    assert "unavailable" in res.reason.lower()
    assert res.is_calibrated is False


def test_real_scenario_three_red_onions():
    """Full scenario test matching the user's test: exactly 3 red onions (2 adjacent touching + 1 separate)."""
    h, w = 800, 1000
    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    canvas[:, :] = (45, 45, 45)  # Neutral background table/mat

    # Two adjacent touching onions
    create_synthetic_red_onion(canvas, 320, 400, 105, 110, ang=10, with_highlight=True)
    create_synthetic_red_onion(canvas, 510, 420, 100, 105, ang=-15, with_highlight=True)

    # One separate onion
    create_synthetic_red_onion(canvas, 750, 380, 110, 112, ang=5, with_highlight=True)

    segmenter = OnionSegmenter(model_path=None)

    # 1. Uncalibrated pass (marker missing)
    onions_uncal, _ = segmenter.segment_and_measure(canvas, is_calibrated=False)
    assert len(onions_uncal) == 3, f"Expected exactly 3 onions detected, got {len(onions_uncal)}"
    for o in onions_uncal:
        assert o.minor_axis_mm is None
        assert o.is_calibrated is False

    # 2. Calibrated pass (with 4 px/mm)
    onions_cal, _ = segmenter.segment_and_measure(canvas, mm_per_pixel=0.25, is_calibrated=True)
    assert len(onions_cal) == 3
    for o in onions_cal:
        assert o.is_calibrated is True
        assert o.minor_axis_mm is not None
        # All 3 are ~50-55mm bulbs, none should be 137 mm or 20.1 mm!
        assert 45.0 <= o.minor_axis_mm <= 60.0
