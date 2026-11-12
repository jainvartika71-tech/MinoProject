"""Unit tests for capture quality pre-flight checks."""

import pytest
import numpy as np
import cv2
from src.scale import generate_aruco_marker
from src.capture_check import check_capture_quality


def test_reject_dark_image():
    # Completely black or very dim image
    dark_img = np.ones((400, 400, 3), dtype=np.uint8) * 10
    res = check_capture_quality(dark_img, min_brightness=45.0, require_marker=False)
    assert res.passed is False
    assert res.is_too_dark is True
    assert any("dark" in r.lower() for r in res.rejection_reasons)


def test_reject_blurry_image():
    # Uniform gray image or heavily Gaussian-blurred pattern
    canvas = np.random.randint(0, 255, (400, 400, 3), dtype=np.uint8)
    blurred = cv2.GaussianBlur(canvas, (51, 51), 0)
    res = check_capture_quality(blurred, blur_threshold=80.0, require_marker=False)
    assert res.passed is False
    assert res.is_blurry is True
    assert any("blurry" in r.lower() for r in res.rejection_reasons)


def test_reject_missing_marker():
    # Sharp, well-lit image without ArUco marker
    sharp_img = np.random.randint(60, 200, (400, 400, 3), dtype=np.uint8)
    res = check_capture_quality(sharp_img, require_marker=True)
    assert res.passed is False
    assert res.marker_found is False
    assert any("marker" in r.lower() for r in res.rejection_reasons)


def test_pass_valid_image():
    # Well-lit canvas with sharp text/patterns and a valid ArUco marker
    canvas = np.ones((600, 600, 3), dtype=np.uint8) * 200
    cv2.putText(canvas, "Onion Lot Sample #401", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
    cv2.circle(canvas, (300, 300), 50, (30, 30, 180), -1)

    marker = generate_aruco_marker(marker_id=0, dict_name="DICT_4X4_50", size_pixels=150)
    marker_bgr = cv2.cvtColor(marker, cv2.COLOR_GRAY2BGR)
    canvas[50:200, 400:550] = marker_bgr

    res = check_capture_quality(
        canvas,
        blur_threshold=50.0,
        min_brightness=45.0,
        max_brightness=235.0,
        require_marker=True,
    )
    assert res.passed is True
    assert res.marker_found is True
    assert res.is_blurry is False
    assert res.is_too_dark is False
