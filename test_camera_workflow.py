"""Integration and unit tests for camera capture, defect classification, and segmentation robustness."""

import pytest
import numpy as np
import cv2

from src.capture_check import check_capture_quality
from src.defects import DefectClassifier
from src.segment import OnionSegmenter
from src.calibration import CameraCalibration
from tests.test_cv_improvements import create_synthetic_red_onion


def test_quality_check_passes_when_camera_calibrated_without_marker():
    """Verifies that pre-flight quality check passes without requiring an ArUco marker when camera is calibrated."""
    canvas = np.ones((500, 500, 3), dtype=np.uint8) * 180
    cv2.putText(canvas, "Sample Onion Lot", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (20, 20, 20), 2)
    cv2.circle(canvas, (250, 250), 60, (40, 20, 150), -1)

    # With camera_calibrated=True and require_marker=False
    res = check_capture_quality(
        canvas,
        blur_threshold=50.0,
        min_brightness=45.0,
        max_brightness=235.0,
        require_marker=False,
        camera_calibrated=True,
    )

    assert res.passed is True
    assert res.is_blurry is False
    assert res.is_too_dark is False
    assert len(res.rejection_reasons) == 0


def test_red_onion_dark_skin_not_misclassified_as_rot():
    """Requirement 5: Deep red/purple onion skin and surface folds must NOT be falsely classified as rotten."""
    classifier = DefectClassifier()

    # Dark red/purple onion skin crop with natural skin folds
    crop = np.zeros((100, 100, 3), dtype=np.uint8)
    crop[:, :] = (55, 25, 130)  # Saturated crimson red
    # Add dark skin fold (Value ~ 35, but high Saturation ~ 160)
    cv2.line(crop, (20, 50), (80, 50), (30, 12, 65), 4)

    label, conf, details = classifier.classify_crop(crop)
    assert label == "healthy", f"Expected healthy on red skin, got {label} (details: {details})"
    assert details["rot_ratio"] < 0.05


def test_genuine_black_mold_rot_correctly_classified():
    """Requirement 5: True black mold rot lesion (>5% area) must be detected as rotten."""
    classifier = DefectClassifier()

    # Base onion with a large black mold lesion
    crop = np.zeros((100, 100, 3), dtype=np.uint8)
    crop[:, :] = (50, 140, 210)
    # Fungal necrotic black mold patch (40x40 px on 100x100 crop = ~16% area)
    crop[30:70, 30:70] = (12, 12, 12)

    label, conf, details = classifier.classify_crop(crop)
    assert label == "rotten", f"Expected rotten on black mold patch, got {label}"
    assert details["rot_ratio"] >= 0.05


def test_onion_with_roots_and_stem_equatorial_diameter():
    """Requirement 4: Loose root hairs and dried stem neck must be excluded from equatorial diameter measurement."""
    canvas = np.zeros((500, 500, 3), dtype=np.uint8)
    canvas[:, :] = (40, 40, 40)

    # 80px radius bulb -> 40 mm diameter at 0.25 mm/px
    create_synthetic_red_onion(canvas, 250, 250, 80, 80, with_highlight=True)

    # Add thin root hairs extending downwards (2px wide)
    cv2.line(canvas, (245, 330), (240, 370), (70, 35, 160), 2)
    cv2.line(canvas, (250, 330), (252, 375), (70, 35, 160), 2)
    cv2.line(canvas, (255, 330), (260, 368), (70, 35, 160), 2)

    # Add dry neck/stem on top (3px wide)
    cv2.line(canvas, (250, 170), (250, 130), (70, 35, 160), 3)

    segmenter = OnionSegmenter(model_path=None)
    onions, _ = segmenter.segment_and_measure(canvas, mm_per_pixel=0.25, is_calibrated=True)

    assert len(onions) == 1
    # Minor axis diameter should be ~40 mm (+- 2.5 mm)
    assert abs(onions[0].minor_axis_mm - 40.0) < 2.5


def test_reject_human_hand_and_fingers():
    """Requirement 4: Human hands, arms, and fingertips in the capture must NOT be detected as onions."""
    canvas = np.zeros((600, 600, 3), dtype=np.uint8)
    canvas[:, :] = (40, 40, 40)

    # Valid onion bulb
    create_synthetic_red_onion(canvas, 300, 300, 80, 80)

    # Hand/arm polygon at edge of mat
    arm_pts = np.array([[50, 400], [70, 420], [80, 580], [40, 580]], dtype=np.int32)
    cv2.fillPoly(canvas, [arm_pts], (120, 150, 200))
    # Small fingertips
    cv2.line(canvas, (70, 420), (90, 380), (120, 150, 200), 10)
    cv2.line(canvas, (60, 410), (80, 370), (120, 150, 200), 10)

    segmenter = OnionSegmenter(model_path=None)
    onions, _ = segmenter.segment_and_measure(canvas, mm_per_pixel=0.25, is_calibrated=True)

    # Only the genuine onion bulb must be detected
    assert len(onions) == 1, f"Expected exactly 1 onion, got {len(onions)}"
    assert onions[0].minor_axis_mm >= 35.0
