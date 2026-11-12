"""ArUco Reference Marker detection and metric scale calibration.

Computes physical pixels-per-mm ratio from a known reference marker
(standard 50mm x 50mm ArUco DICT_4X4_50, ID 0) captured in the overhead photo.
If marker is absent or calibration is invalid, explicitly reports calibration unavailable.
"""

from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any
import numpy as np
import cv2


ARUCO_DICT_MAP = {
    "DICT_4X4_50": cv2.aruco.DICT_4X4_50 if hasattr(cv2, "aruco") and hasattr(cv2.aruco, "DICT_4X4_50") else 0,
    "DICT_4X4_100": cv2.aruco.DICT_4X4_100 if hasattr(cv2, "aruco") and hasattr(cv2.aruco, "DICT_4X4_100") else 1,
    "DICT_5X5_50": cv2.aruco.DICT_5X5_50 if hasattr(cv2, "aruco") and hasattr(cv2.aruco, "DICT_5X5_50") else 4,
    "DICT_6X6_50": cv2.aruco.DICT_6X6_50 if hasattr(cv2, "aruco") and hasattr(cv2.aruco, "DICT_6X6_50") else 8,
}


@dataclass
class MarkerDetectionResult:
    """Outcome of ArUco marker detection."""
    found: bool
    is_calibrated: bool = False
    marker_id: Optional[int] = None
    pixels_per_mm: Optional[float] = None
    mm_per_pixel: Optional[float] = None
    physical_size_mm: float = 50.0
    corner_points: Optional[np.ndarray] = None
    annotated_image: Optional[np.ndarray] = None
    error_message: Optional[str] = None


def get_aruco_dictionary(dict_name: str = "DICT_4X4_50"):
    """Safely retrieves the ArUco dictionary across OpenCV versions."""
    if not hasattr(cv2, "aruco"):
        raise ImportError("OpenCV ArUco module is not available in the installed opencv package.")
    dict_enum = ARUCO_DICT_MAP.get(dict_name, cv2.aruco.DICT_4X4_50)
    return cv2.aruco.getPredefinedDictionary(dict_enum)


def generate_aruco_marker(
    marker_id: int = 0,
    dict_name: str = "DICT_4X4_50",
    size_pixels: int = 400,
    border_bits: int = 1,
) -> np.ndarray:
    """Generates a printable ArUco marker image."""
    dictionary = get_aruco_dictionary(dict_name)
    if hasattr(cv2.aruco, "generateImageMarker"):
        img = cv2.aruco.generateImageMarker(dictionary, marker_id, size_pixels, borderBits=border_bits)
    else:
        # Compatibility with OpenCV < 4.7
        img = np.zeros((size_pixels, size_pixels), dtype=np.uint8)
        cv2.aruco.drawMarker(dictionary, marker_id, size_pixels, img, border_bits)
    return img


def is_valid_calibration(result: MarkerDetectionResult) -> bool:
    """Returns True if the marker detection outcome provides a verified physical scale."""
    return (
        result is not None
        and result.found is True
        and result.is_calibrated is True
        and result.pixels_per_mm is not None
        and result.pixels_per_mm > 0.0
        and result.mm_per_pixel is not None
        and result.mm_per_pixel > 0.0
    )


def detect_aruco_marker(
    image: np.ndarray,
    target_id: Optional[int] = 0,
    physical_size_mm: float = 50.0,
    dict_name: str = "DICT_4X4_50",
) -> MarkerDetectionResult:
    """Detects the ArUco marker in the image and computes pixels_per_mm.

    Args:
        image: BGR or RGB or Grayscale numpy image.
        target_id: Expected marker ID (default 0 for DICT_4X4_50; None accepts any ID).
        physical_size_mm: Known physical side length of the square marker in mm.
        dict_name: ArUco dictionary name (default 'DICT_4X4_50').

    Returns:
        MarkerDetectionResult with scale factor and diagnostic overlay.
    """
    if image is None or image.size == 0:
        return MarkerDetectionResult(
            found=False,
            is_calibrated=False,
            error_message="Image is empty or null.",
            physical_size_mm=physical_size_mm,
        )

    # Convert to grayscale
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        annotated = image.copy()
    else:
        gray = image.copy()
        annotated = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    dictionary = get_aruco_dictionary(dict_name)

    # Detect markers using appropriate OpenCV API
    if hasattr(cv2.aruco, "ArucoDetector"):
        detector_params = cv2.aruco.DetectorParameters()
        detector = cv2.aruco.ArucoDetector(dictionary, detector_params)
        corners, ids, rejected = detector.detectMarkers(gray)
    else:
        detector_params = cv2.aruco.DetectorParameters_create()
        corners, ids, rejected = cv2.aruco.detectMarkers(gray, dictionary, parameters=detector_params)

    if ids is None or len(ids) == 0:
        return MarkerDetectionResult(
            found=False,
            is_calibrated=False,
            pixels_per_mm=None,
            mm_per_pixel=None,
            error_message=(
                f"No ArUco marker detected with dictionary {dict_name}. "
                "Ensure a high-contrast printed marker is placed in the frame."
            ),
            physical_size_mm=physical_size_mm,
            annotated_image=annotated,
        )

    ids_flat = ids.flatten()
    selected_idx = None
    if target_id is not None:
        matches = np.where(ids_flat == target_id)[0]
        if len(matches) > 0:
            selected_idx = matches[0]
        else:
            return MarkerDetectionResult(
                found=False,
                is_calibrated=False,
                pixels_per_mm=None,
                mm_per_pixel=None,
                error_message=f"Target marker ID {target_id} not found among detected IDs {list(ids_flat)}.",
                physical_size_mm=physical_size_mm,
                annotated_image=annotated,
            )
    else:
        selected_idx = 0

    marker_id_found = int(ids_flat[selected_idx])
    pts = corners[selected_idx][0]  # Shape: (4, 2)

    # Compute Euclidean lengths of all 4 sides of the marker
    side_lengths = [
        float(np.linalg.norm(pts[0] - pts[1])),
        float(np.linalg.norm(pts[1] - pts[2])),
        float(np.linalg.norm(pts[2] - pts[3])),
        float(np.linalg.norm(pts[3] - pts[0])),
    ]
    avg_side_px = float(np.mean(side_lengths))
    min_side = min(side_lengths)
    max_side = max(side_lengths)

    # Geometry verification: squareness check (perspective distortion guard)
    if max_side > 0 and (min_side / max_side) < 0.60:
        return MarkerDetectionResult(
            found=True,
            is_calibrated=False,
            marker_id=marker_id_found,
            pixels_per_mm=None,
            mm_per_pixel=None,
            error_message="Detected marker is severely skewed or distorted. Keep camera overhead.",
            physical_size_mm=physical_size_mm,
            annotated_image=annotated,
            corner_points=pts,
        )

    if avg_side_px < 15.0 or physical_size_mm <= 0.0:
        return MarkerDetectionResult(
            found=True,
            is_calibrated=False,
            marker_id=marker_id_found,
            pixels_per_mm=None,
            mm_per_pixel=None,
            error_message="Detected marker is too small (< 15 px) for reliable millimetre scaling.",
            physical_size_mm=physical_size_mm,
            annotated_image=annotated,
            corner_points=pts,
        )

    pixels_per_mm = avg_side_px / physical_size_mm
    mm_per_pixel = physical_size_mm / avg_side_px

    # Draw marker outline and annotation
    pts_int = pts.astype(int)
    cv2.polylines(annotated, [pts_int], isClosed=True, color=(0, 255, 0), thickness=3)
    c_x, c_y = int(np.mean(pts[:, 0])), int(np.mean(pts[:, 1]))
    cv2.putText(
        annotated,
        f"Scale ID:{marker_id_found} ({pixels_per_mm:.2f} px/mm)",
        (c_x - 50, max(25, c_y - 20)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )

    return MarkerDetectionResult(
        found=True,
        is_calibrated=True,
        marker_id=marker_id_found,
        pixels_per_mm=round(pixels_per_mm, 4),
        mm_per_pixel=round(mm_per_pixel, 4),
        physical_size_mm=physical_size_mm,
        corner_points=pts,
        annotated_image=annotated,
    )
