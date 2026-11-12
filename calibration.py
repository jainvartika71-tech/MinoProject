"""Persistent fixed-camera calibration module for Onion Grading Automation.

Replaces mandatory per-inspection ArUco markers with a one-time calibration profile:
- Calibrate using a standard ruler (e.g., 100mm) or known reference object (ID card, coin).
- Optionally perform a one-time ArUco calibration and remove the marker forever.
- Persists calibration parameters to `config/camera_calibration.json`.
- Subsequent inspections automatically reuse the saved scale without needing markers.
- Supports recalibration and clearing when camera height/position changes.
"""

import json
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
import numpy as np
import cv2

from src.scale import detect_aruco_marker, MarkerDetectionResult


DEFAULT_CALIBRATION_FILE = Path(__file__).resolve().parent.parent / "config" / "camera_calibration.json"

# Common physical reference dimensions in millimeters
CALIBRATION_PRESETS: Dict[str, float] = {
    "Standard ID / Credit Card (Width: 85.6 mm)": 85.6,
    "Standard ID / Credit Card (Height: 53.98 mm)": 53.98,
    "Standard Ruler Segment (100.0 mm)": 100.0,
    "Standard Ruler Segment (150.0 mm)": 150.0,
    "Standard Ruler Segment (50.0 mm)": 50.0,
    "Indian ₹5 Coin (Diameter: 23.0 mm)": 23.0,
    "Indian ₹10 Coin (Diameter: 27.0 mm)": 27.0,
    "US Quarter (Diameter: 24.26 mm)": 24.26,
    "ArUco Marker (One-Time: 50.0 mm)": 50.0,
}


@dataclass
class CameraCalibration:
    """Persistent calibration profile for fixed overhead camera."""
    is_calibrated: bool = False
    pixels_per_mm: Optional[float] = None
    mm_per_pixel: Optional[float] = None
    method: str = "uncalibrated"  # "ruler", "preset_object", "aruco_one_time", "manual"
    reference_name: str = ""
    reference_dimension_mm: Optional[float] = None
    measured_pixels: Optional[float] = None
    camera_name: str = "MacBook Built-in Camera"
    fixed_height_cm: Optional[float] = 35.0
    calibrated_at: Optional[str] = None
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CameraCalibration":
        valid_keys = cls.__dataclass_fields__.keys()
        filtered = {k: v for k, v in data.items() if k in valid_keys}
        return cls(**filtered)


def get_default_calibration_path() -> Path:
    return DEFAULT_CALIBRATION_FILE


def load_camera_calibration(config_path: Optional[Path] = None) -> CameraCalibration:
    """Loads camera calibration from persistent JSON storage."""
    path = config_path or get_default_calibration_path()
    if not path.exists():
        return CameraCalibration()

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return CameraCalibration.from_dict(data)
    except Exception:
        return CameraCalibration()


def save_camera_calibration(calib: CameraCalibration, config_path: Optional[Path] = None) -> bool:
    """Persists camera calibration to JSON storage."""
    path = config_path or get_default_calibration_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(calib.to_dict(), f, indent=2)
        return True
    except Exception:
        return False


def clear_camera_calibration(config_path: Optional[Path] = None) -> bool:
    """Resets the persistent calibration to an uncalibrated state."""
    uncalibrated = CameraCalibration(
        is_calibrated=False,
        pixels_per_mm=None,
        mm_per_pixel=None,
        method="uncalibrated",
        notes="Calibration cleared by user.",
    )
    return save_camera_calibration(uncalibrated, config_path)


def is_valid_camera_calibration(calib: Optional[CameraCalibration]) -> bool:
    """Checks whether a calibration profile provides valid physical scaling factors."""
    return (
        calib is not None
        and calib.is_calibrated is True
        and calib.pixels_per_mm is not None
        and calib.pixels_per_mm > 0.0
        and calib.mm_per_pixel is not None
        and calib.mm_per_pixel > 0.0
    )


def calibrate_from_pixels(
    pixel_length: float,
    physical_mm: float,
    method: str = "ruler",
    reference_name: str = "Standard Reference",
    camera_name: str = "MacBook Built-in Camera",
    fixed_height_cm: Optional[float] = 35.0,
    notes: str = "Fixed overhead calibration",
    config_path: Optional[Path] = None,
) -> CameraCalibration:
    """Calibrates camera scale using a measured pixel span corresponding to a known physical length in mm."""
    if pixel_length <= 0 or physical_mm <= 0:
        raise ValueError("pixel_length and physical_mm must be positive non-zero values.")

    px_per_mm = float(pixel_length / physical_mm)
    mm_per_px = float(physical_mm / pixel_length)

    calib = CameraCalibration(
        is_calibrated=True,
        pixels_per_mm=round(px_per_mm, 4),
        mm_per_pixel=round(mm_per_px, 6),
        method=method,
        reference_name=reference_name,
        reference_dimension_mm=round(float(physical_mm), 2),
        measured_pixels=round(float(pixel_length), 2),
        camera_name=camera_name,
        fixed_height_cm=fixed_height_cm,
        calibrated_at=datetime.utcnow().isoformat() + "Z",
        notes=notes,
    )
    save_camera_calibration(calib, config_path)
    return calib


def calibrate_from_marker(
    image: np.ndarray,
    physical_size_mm: float = 50.0,
    target_id: Optional[int] = 0,
    dict_name: str = "DICT_4X4_50",
    camera_name: str = "MacBook Built-in Camera",
    fixed_height_cm: Optional[float] = 35.0,
    notes: str = "One-time ArUco calibration",
    config_path: Optional[Path] = None,
) -> Tuple[CameraCalibration, MarkerDetectionResult]:
    """Calibrates camera scale from an image containing an ArUco marker, saving it permanently."""
    marker_res = detect_aruco_marker(
        image=image,
        target_id=target_id,
        physical_size_mm=physical_size_mm,
        dict_name=dict_name,
    )

    if not marker_res.found or not marker_res.is_calibrated or marker_res.pixels_per_mm is None:
        uncalib = CameraCalibration(
            is_calibrated=False,
            notes=marker_res.error_message or "ArUco marker not detected in image.",
        )
        return uncalib, marker_res

    calib = CameraCalibration(
        is_calibrated=True,
        pixels_per_mm=round(marker_res.pixels_per_mm, 4),
        mm_per_pixel=round(marker_res.mm_per_pixel, 6),
        method="aruco_one_time",
        reference_name=f"ArUco {dict_name} ID {marker_res.marker_id} ({physical_size_mm}mm)",
        reference_dimension_mm=round(physical_size_mm, 2),
        measured_pixels=round(marker_res.pixels_per_mm * physical_size_mm, 2),
        camera_name=camera_name,
        fixed_height_cm=fixed_height_cm,
        calibrated_at=datetime.utcnow().isoformat() + "Z",
        notes=notes,
    )
    save_camera_calibration(calib, config_path)
    return calib, marker_res


def get_standard_camera_calibration(
    fixed_height_cm: float = 35.0,
    pixels_per_mm: float = 3.80,
    reference_name: str = "Standard 100mm Reference Ruler",
    config_path: Optional[Path] = None,
) -> CameraCalibration:
    """Returns and persists a standard verified calibration profile for overhead desk setups."""
    mm_per_px = round(1.0 / pixels_per_mm, 6)
    calib = CameraCalibration(
        is_calibrated=True,
        pixels_per_mm=pixels_per_mm,
        mm_per_pixel=mm_per_px,
        method="standard_fixed_mount",
        reference_name=reference_name,
        reference_dimension_mm=100.0,
        measured_pixels=round(100.0 * pixels_per_mm, 1),
        camera_name="MacBook Built-in Camera",
        fixed_height_cm=fixed_height_cm,
        calibrated_at=datetime.utcnow().isoformat() + "Z",
        notes=f"Standard overhead camera calibration ({pixels_per_mm:.2f} px/mm at {fixed_height_cm:.0f}cm mount height)",
    )
    save_camera_calibration(calib, config_path)
    return calib


def auto_detect_reference_scale(
    image: np.ndarray,
    physical_dimension_mm: Optional[float] = None,
    config_path: Optional[Path] = None,
) -> Tuple[Optional[CameraCalibration], str]:
    """Attempts to automatically detect a reference scale in the image.

    Checks:
    1. ArUco Marker (DICT_4X4_50 or any detected dictionary).
    2. High-contrast standard calibration card / rectangle (ID card ratio ~1.58).
    """
    if image is None or image.size == 0:
        return None, "Empty or invalid image."

    # 1. Try ArUco marker first
    calib, marker_res = calibrate_from_marker(
        image=image,
        physical_size_mm=physical_dimension_mm if physical_dimension_mm else 50.0,
        config_path=config_path,
    )
    if calib.is_calibrated:
        return calib, f"Detected ArUco Marker ({calib.pixels_per_mm:.2f} px/mm)"

    # 2. Try Card detection
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 3)
    cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best_card = None
    best_card_area = 0

    for c in cnts:
        area = cv2.contourArea(c)
        if area < 3000:
            continue
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.03 * peri, True)
        if len(approx) == 4:
            x, y, w, h = cv2.boundingRect(c)
            long_side = max(w, h)
            short_side = min(w, h)
            ar = long_side / short_side if short_side > 0 else 0
            if 1.40 <= ar <= 1.75 and area > best_card_area:
                best_card_area = area
                best_card = (long_side, short_side)

    if best_card is not None:
        long_side_px, _ = best_card
        ref_mm = physical_dimension_mm if physical_dimension_mm else 85.6
        calib = calibrate_from_pixels(
            pixel_length=float(long_side_px),
            physical_mm=ref_mm,
            method="auto_card_detection",
            reference_name="Standard ID / Credit Card (85.6 mm)",
            config_path=config_path,
        )
        return calib, f"Detected Standard Reference Card ({calib.pixels_per_mm:.2f} px/mm)"

    return None, "No standard reference object or marker detected in current image."
