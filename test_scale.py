"""Unit tests for ArUco marker detection and scale factor computation."""

import pytest
import numpy as np
import cv2
from src.scale import (
    generate_aruco_marker,
    detect_aruco_marker,
    MarkerDetectionResult,
)


def test_generate_aruco_marker():
    marker_img = generate_aruco_marker(marker_id=0, dict_name="DICT_4X4_50", size_pixels=200)
    assert marker_img is not None
    assert marker_img.shape == (200, 200)
    # Check that it's binary black and white
    unique_vals = np.unique(marker_img)
    assert len(unique_vals) <= 2


def test_detect_known_marker_scale():
    # Create white canvas of 600x600
    canvas = np.ones((600, 600, 3), dtype=np.uint8) * 255

    # Generate a 200x200 pixel marker (ID 0)
    marker = generate_aruco_marker(marker_id=0, dict_name="DICT_4X4_50", size_pixels=200)
    marker_bgr = cv2.cvtColor(marker, cv2.COLOR_GRAY2BGR)

    # Place marker in canvas at (200, 200)
    canvas[200:400, 200:400] = marker_bgr

    # Known physical size = 50.0 mm
    # Expected px_per_mm = 200 / 50.0 = 4.00 px/mm
    res = detect_aruco_marker(canvas, target_id=0, physical_size_mm=50.0, dict_name="DICT_4X4_50")

    assert res.found is True
    assert res.marker_id == 0
    # Allow small 1% edge rasterization margin
    assert abs(res.pixels_per_mm - 4.0) < 0.1
    # Check mm measurement accuracy: 200 px * mm_per_pixel ~ 50.0 mm (+- 0.5 mm)
    measured_mm = 200.0 * res.mm_per_pixel
    assert abs(measured_mm - 50.0) < 0.5


def test_missing_marker():
    blank = np.ones((400, 400, 3), dtype=np.uint8) * 200
    res = detect_aruco_marker(blank, physical_size_mm=50.0)
    assert res.found is False
    assert "no aruco marker detected" in res.error_message.lower()


def test_empty_image():
    res = detect_aruco_marker(np.array([]))
    assert res.found is False
    assert "empty" in res.error_message.lower()
