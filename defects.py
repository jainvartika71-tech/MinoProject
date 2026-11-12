"""Onion surface defect classification module.

Detects and classifies onion flaws:
- 'healthy': Uniform surface, no decay, no shoots
- 'rotten': Dark black mold (Aspergillus), soft rot, or necrotic tissue
- 'sprouted': Green apical shoot emergence
- 'damaged': Cuts, mechanical cracking, heavy peeling, sunburn bruises
- 'other': Unclassified blemishes
"""

from typing import Tuple, Dict, Any, Optional
import numpy as np
import cv2


class DefectClassifier:
    """Classifies onion surface condition from cropped bulb image."""

    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path
        # When a deep learning crop classifier weights file is supplied, load it here.

    def classify_crop(
        self,
        crop: Optional[np.ndarray],
        base_label: str = "healthy",
        mask: Optional[np.ndarray] = None,
    ) -> Tuple[str, float, Dict[str, float]]:
        """Evaluates onion crop image for visible defects strictly within bulb mask.

        Args:
            crop: BGR image crop of the individual onion.
            base_label: Pre-existing label (e.g. from YOLO multi-class model).
            mask: Optional binary mask of the onion bulb corresponding to the crop.

        Returns:
            Tuple of (defect_label, confidence, diagnostic_scores)
        """
        if base_label in ("rotten", "sprouted", "damaged", "other") and base_label != "healthy":
            return base_label, 0.88, {"source": "yolo_detector"}

        if crop is None or crop.size == 0 or crop.shape[0] < 15 or crop.shape[1] < 15:
            return "healthy", 0.75, {"note": "crop too small for visual defect pass"}

        h, w = crop.shape[:2]

        # Derive foreground mask (bulb surface) to exclude background mat
        if mask is not None and mask.shape == (h, w):
            fg_mask = (mask > 0).astype(np.uint8) * 255
        else:
            # Elliptical foreground approximation within bounding box
            fg_mask = np.zeros((h, w), dtype=np.uint8)
            cv2.ellipse(fg_mask, (w // 2, h // 2), (w // 2 - 2, h // 2 - 2), 0, 0, 360, 255, -1)

        fg_pixels = int(np.count_nonzero(fg_mask))
        if fg_pixels < 50:
            return "healthy", 0.75, {"note": "foreground area too small"}

        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)

        # 1. Green Sprout Detection within bulb & apical region
        # Green hue roughly 35 to 85 in OpenCV (0-180 scale)
        green_raw = cv2.inRange(hsv, (35, 45, 45), (85, 255, 255))
        # Sprout can extend slightly outside or within bulb
        green_ratio = float(np.count_nonzero(green_raw)) / float(fg_pixels)

        if green_ratio > 0.04:  # Over 4% green shoot pixels
            return "sprouted", min(0.95, 0.70 + green_ratio * 4.0), {"green_ratio": round(green_ratio, 4)}

        # 2. Black Mold / Deep Rot Necrosis Detection (strictly on bulb foreground)
        # Distinguish true necrotic rot (achromatic black mold, soft rot) from saturated dark red onion skin
        rot_candidates = cv2.inRange(hsv, (0, 0, 0), (180, 80, 42))
        deep_black = cv2.inRange(hsv, (0, 0, 0), (180, 130, 22))
        black_rot_raw = cv2.bitwise_or(rot_candidates, deep_black)
        rot_on_bulb = cv2.bitwise_and(black_rot_raw, black_rot_raw, mask=fg_mask)

        # Suppress hairline shadows and skin fold crevices using morphological opening
        kernel_rot = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        rot_clean = cv2.morphologyEx(rot_on_bulb, cv2.MORPH_OPEN, kernel_rot)
        rot_pixels = int(np.count_nonzero(rot_clean))
        rot_ratio = float(rot_pixels) / float(fg_pixels)

        if rot_ratio > 0.05:  # Over 5% necrotic rot on the actual bulb surface
            return "rotten", min(0.96, 0.70 + rot_ratio * 3.5), {"rot_ratio": round(rot_ratio, 4)}

        # 3. Surface Damage / Mechanical Cuts Detection
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 60, 160)
        edges_on_bulb = cv2.bitwise_and(edges, edges, mask=fg_mask)
        edge_ratio = float(np.count_nonzero(edges_on_bulb)) / float(fg_pixels)

        if edge_ratio > 0.18 and rot_ratio > 0.02:
            return "damaged", 0.82, {"edge_ratio": round(edge_ratio, 4), "rot_ratio": round(rot_ratio, 4)}

        return "healthy", 0.90, {
            "green_ratio": round(green_ratio, 4),
            "rot_ratio": round(rot_ratio, 4),
            "edge_ratio": round(edge_ratio, 4),
        }
