"""Capture quality pre-check module.

Validates that uploaded/captured photos meet minimum procurement imaging standards:
- Sufficient lighting (not underexposed/too dark)
- No extreme glare (not overexposed)
- High sharpness (Laplacian variance above blur threshold)
- ArUco reference marker detected for metric calibration
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Any
import numpy as np
import cv2

from src.scale import detect_aruco_marker, MarkerDetectionResult


@dataclass
class CaptureQualityResult:
    """Assessment outcome of the captured image quality."""
    passed: bool
    is_blurry: bool
    is_too_dark: bool
    is_too_bright: bool
    marker_found: bool
    blur_score: float
    brightness_score: float
    rejection_reasons: List[str] = field(default_factory=list)
    marker_result: Optional[MarkerDetectionResult] = None

    @property
    def feedback_message(self) -> str:
        if self.passed:
            return "Image quality passed: sharp focus, balanced illumination, and ArUco scale detected."
        return "Image rejected for procurement analysis:\n• " + "\n• ".join(self.rejection_reasons)


def check_capture_quality(
    image: np.ndarray,
    blur_threshold: float = 80.0,
    min_brightness: float = 45.0,
    max_brightness: float = 235.0,
    marker_dict: str = "DICT_4X4_50",
    marker_size_mm: float = 50.0,
    target_marker_id: Optional[int] = None,
    require_marker: bool = False,
    camera_calibrated: bool = False,
) -> CaptureQualityResult:
    """Evaluates sharpness, exposure, and scale calibration status.

    Args:
        image: BGR or RGB image as numpy array.
        blur_threshold: Minimum acceptable Laplacian variance.
        min_brightness: Minimum mean intensity (0-255).
        max_brightness: Maximum mean intensity (0-255).
        marker_dict: ArUco dictionary to search.
        marker_size_mm: Physical dimension of the marker.
        target_marker_id: Expected marker ID (if any).
        require_marker: Whether marker presence is strictly mandatory.
        camera_calibrated: Whether persistent camera calibration is active.

    Returns:
        CaptureQualityResult.
    """
    if image is None or image.size == 0:
        return CaptureQualityResult(
            passed=False,
            is_blurry=True,
            is_too_dark=True,
            is_too_bright=False,
            marker_found=False,
            blur_score=0.0,
            brightness_score=0.0,
            rejection_reasons=["Empty or unreadable image uploaded."],
        )

    # Convert to grayscale
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    # 1. Blur Detection (Laplacian Variance)
    laplacian_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    is_blurry = laplacian_var < blur_threshold

    # 2. Exposure / Brightness Check
    mean_brightness = float(np.mean(gray))
    is_too_dark = mean_brightness < min_brightness
    is_too_bright = mean_brightness > max_brightness

    # 3. Scale Calibration Check (Fixed camera calibration OR ArUco Marker)
    marker_res = None
    marker_calibrated = False
    if require_marker or not camera_calibrated:
        marker_res = detect_aruco_marker(
            image=image,
            target_id=target_marker_id,
            physical_size_mm=marker_size_mm,
            dict_name=marker_dict,
        )
        marker_calibrated = marker_res.found and marker_res.is_calibrated

    reasons: List[str] = []
    if is_blurry:
        reasons.append(
            f"Image is too blurry (sharpness score {laplacian_var:.1f} < threshold {blur_threshold:.1f}). "
            "Please steady the camera and ensure sharp focus."
        )
    if is_too_dark:
        reasons.append(
            f"Image is underexposed / too dark (brightness {mean_brightness:.1f} < {min_brightness:.1f}). "
            "Please capture under uniform overhead lighting."
        )
    if is_too_bright:
        reasons.append(
            f"Image has severe glare / overexposure (brightness {mean_brightness:.1f} > {max_brightness:.1f}). "
            "Please eliminate direct flash reflection."
        )
    if require_marker and not (marker_calibrated or camera_calibrated):
        reasons.append(
            f"Scale calibration missing: {(marker_res.error_message if marker_res else None) or 'ArUco marker or saved camera calibration required.'}"
        )

    passed = (len(reasons) == 0)

    return CaptureQualityResult(
        passed=passed,
        is_blurry=is_blurry,
        is_too_dark=is_too_dark,
        is_too_bright=is_too_bright,
        marker_found=marker_calibrated,
        blur_score=round(laplacian_var, 2),
        brightness_score=round(mean_brightness, 2),
        rejection_reasons=reasons,
        marker_result=marker_res,
    )
