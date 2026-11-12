"""Unit tests for persistent camera calibration and markerless optical scaling."""

import json
from pathlib import Path
import pytest
import numpy as np
import cv2

from src.calibration import (
    CameraCalibration,
    load_camera_calibration,
    save_camera_calibration,
    clear_camera_calibration,
    calibrate_from_pixels,
    calibrate_from_marker,
    is_valid_camera_calibration,
    CALIBRATION_PRESETS,
)
from src.scale import generate_aruco_marker
from src.segment import OnionSegmenter
from tests.test_cv_improvements import create_synthetic_red_onion


def test_calibration_presets_exist():
    assert "Standard ID / Credit Card (Width: 85.6 mm)" in CALIBRATION_PRESETS
    assert "Standard Ruler Segment (100.0 mm)" in CALIBRATION_PRESETS
    assert CALIBRATION_PRESETS["Standard Ruler Segment (100.0 mm)"] == 100.0


def test_calibrate_from_pixels_and_persistence(tmp_path: Path):
    cfg_file = tmp_path / "camera_calibration.json"

    # Known 100mm ruler span spanning 400 pixels (4.0 px/mm)
    calib = calibrate_from_pixels(
        pixel_length=400.0,
        physical_mm=100.0,
        method="ruler",
        reference_name="100mm Steel Ruler",
        camera_name="MacBook Pro Camera",
        fixed_height_cm=35.0,
        config_path=cfg_file,
    )

    assert calib.is_calibrated is True
    assert abs(calib.pixels_per_mm - 4.0) < 1e-4
    assert abs(calib.mm_per_pixel - 0.25) < 1e-4
    assert is_valid_camera_calibration(calib) is True

    # Verify JSON persistence
    assert cfg_file.exists()
    loaded = load_camera_calibration(cfg_file)
    assert loaded.is_calibrated is True
    assert abs(loaded.pixels_per_mm - 4.0) < 1e-4
    assert loaded.reference_name == "100mm Steel Ruler"

    # Clear calibration
    clear_camera_calibration(cfg_file)
    cleared = load_camera_calibration(cfg_file)
    assert cleared.is_calibrated is False
    assert is_valid_camera_calibration(cleared) is False


def test_calibrate_from_marker_snapshot(tmp_path: Path):
    cfg_file = tmp_path / "camera_calibration.json"

    # Create image with 200x200 px ArUco marker (50mm -> 4 px/mm)
    canvas = np.ones((500, 500, 3), dtype=np.uint8) * 240
    marker = generate_aruco_marker(marker_id=0, dict_name="DICT_4X4_50", size_pixels=200)
    canvas[100:300, 100:300] = cv2.cvtColor(marker, cv2.COLOR_GRAY2BGR)

    calib, marker_res = calibrate_from_marker(
        image=canvas,
        physical_size_mm=50.0,
        target_id=0,
        dict_name="DICT_4X4_50",
        config_path=cfg_file,
    )

    assert calib.is_calibrated is True
    assert marker_res.found is True
    assert abs(calib.pixels_per_mm - 4.0) < 0.1
    assert abs(calib.mm_per_pixel - 0.25) < 0.01

    # Reload from disk
    loaded = load_camera_calibration(cfg_file)
    assert loaded.is_calibrated is True
    assert "ArUco" in loaded.reference_name


def test_markerless_inspection_with_saved_calibration(tmp_path: Path):
    """Verifies that an inspection image without ANY marker measures onions in mm using saved calibration."""
    # 1. Calibrate once (e.g. 4.0 px/mm -> 0.25 mm/px)
    cfg_file = tmp_path / "camera_calibration.json"
    calib = calibrate_from_pixels(
        pixel_length=400.0,
        physical_mm=100.0,
        method="ruler",
        reference_name="100mm Ruler",
        config_path=cfg_file,
    )

    # 2. Inspection photo contains ONLY an onion, NO marker!
    canvas = np.zeros((600, 600, 3), dtype=np.uint8)
    canvas[:, :] = (40, 40, 40)
    # 80 px radius -> 160 px diameter -> at 0.25 mm/px = 40.0 mm
    create_synthetic_red_onion(canvas, 300, 300, 80, 80, with_highlight=True)

    segmenter = OnionSegmenter(model_path=None)
    onions, _ = segmenter.segment_and_measure(
        canvas,
        mm_per_pixel=calib.mm_per_pixel,
        marker_corners=None,
        is_calibrated=calib.is_calibrated,
    )

    assert len(onions) == 1
    o = onions[0]
    assert o.is_calibrated is True
    assert o.minor_axis_mm is not None
    assert abs(o.minor_axis_mm - 40.0) < 2.0


def test_get_standard_camera_calibration(tmp_path: Path):
    from src.calibration import get_standard_camera_calibration
    cfg_file = tmp_path / "camera_calibration.json"

    calib = get_standard_camera_calibration(
        fixed_height_cm=35.0,
        pixels_per_mm=3.80,
        reference_name="Standard 100mm Reference Ruler",
        config_path=cfg_file,
    )

    assert calib.is_calibrated is True
    assert calib.pixels_per_mm == 3.80
    assert abs(calib.mm_per_pixel - (1.0 / 3.80)) < 1e-4
    assert calib.reference_name == "Standard 100mm Reference Ruler"
    assert cfg_file.exists()


def test_auto_detect_reference_scale_aruco(tmp_path: Path):
    from src.calibration import auto_detect_reference_scale
    cfg_file = tmp_path / "camera_calibration.json"

    canvas = np.ones((500, 500, 3), dtype=np.uint8) * 240
    marker = generate_aruco_marker(marker_id=0, dict_name="DICT_4X4_50", size_pixels=200)
    canvas[100:300, 100:300] = cv2.cvtColor(marker, cv2.COLOR_GRAY2BGR)

    calib, msg = auto_detect_reference_scale(canvas, physical_dimension_mm=50.0, config_path=cfg_file)
    assert calib is not None
    assert calib.is_calibrated is True
    assert "ArUco" in msg
    assert abs(calib.pixels_per_mm - 4.0) < 0.2


def test_auto_detect_reference_scale_card(tmp_path: Path):
    from src.calibration import auto_detect_reference_scale
    cfg_file = tmp_path / "camera_calibration.json"

    # Dark background with a high-contrast white rectangular card (aspect ratio ~ 1.58)
    canvas = np.zeros((600, 600, 3), dtype=np.uint8)
    # 316 x 200 rectangle (ar = 1.58, card dimensions)
    cv2.rectangle(canvas, (100, 100), (416, 300), (250, 250, 250), -1)

    calib, msg = auto_detect_reference_scale(canvas, physical_dimension_mm=85.6, config_path=cfg_file)
    assert calib is not None
    assert calib.is_calibrated is True
    assert "Card" in msg
    # 316 px for 85.6 mm -> ~3.69 px/mm
    assert abs(calib.pixels_per_mm - (316.0 / 85.6)) < 0.2


def test_auto_detect_reference_scale_negative():
    from src.calibration import auto_detect_reference_scale

    canvas = np.zeros((400, 400, 3), dtype=np.uint8)
    calib, msg = auto_detect_reference_scale(canvas)
    assert calib is None
    assert "No standard reference object" in msg

