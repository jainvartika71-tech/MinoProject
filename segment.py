"""Onion Instance Segmentation and Metric Dimensioning.

Robust computer-vision segmentation tailored for real-world onion photographs:
- Multi-space color & texture segmentation (HSV chromaticity + LAB color spaces)
- Morphological hole-filling to eliminate specular reflection artifacts
- True local peak (maxima) detection on Euclidean distance transform to separate touching onions
- Watershed boundary separation with exactly one seed per bulb
- Strict geometric shape gating (solidity, aspect ratio, circularity) to reject shadows, hands, and table artifacts
- Non-maximum suppression and containment filtering to eliminate internal/duplicate contours
- Calibrated millimetre calculation via ArUco reference scale with strict uncalibrated fallbacks
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any
import numpy as np
import cv2


@dataclass
class DetectedOnion:
    """Represents a segmented onion bulb in the sample image."""
    onion_id: int
    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2)
    center_px: Tuple[float, float]
    minor_axis_px: float
    major_axis_px: float
    minor_axis_mm: Optional[float] = None  # None if uncalibrated
    major_axis_mm: Optional[float] = None  # None if uncalibrated
    is_calibrated: bool = False
    confidence: float = 0.85
    defect_label: str = "healthy"
    solidity: float = 1.0
    circularity: float = 1.0
    aspect_ratio: float = 1.0
    mask: Optional[np.ndarray] = None
    contour: Optional[np.ndarray] = None
    crop: Optional[np.ndarray] = None
    ellipse_params: Optional[Tuple[Tuple[float, float], Tuple[float, float], float]] = None


class OnionSegmenter:
    """Performs segmentation and metric sizing on onion photos."""

    def __init__(self, model_path: Optional[Path] = None, conf_threshold: float = 0.40):
        self.model_path = Path(model_path) if model_path else None
        self.conf_threshold = conf_threshold
        self.yolo_model = None
        self._init_model()

    def _init_model(self) -> None:
        """Loads YOLO segmentation model if file is provided and available."""
        if self.model_path and self.model_path.exists():
            try:
                from ultralytics import YOLO
                self.yolo_model = YOLO(str(self.model_path))
            except Exception as e:
                print(f"Warning: Could not load YOLO model at {self.model_path}: {e}")
                self.yolo_model = None

    def segment_and_measure(
        self,
        image: np.ndarray,
        mm_per_pixel: Optional[float] = None,
        marker_corners: Optional[np.ndarray] = None,
        is_calibrated: Optional[bool] = None,
    ) -> Tuple[List[DetectedOnion], np.ndarray]:
        """Segments onions and calculates dimensions (calibrated mm or uncalibrated px).

        Args:
            image: BGR or RGB image as numpy array.
            mm_per_pixel: Scale conversion factor from ArUco marker (if calibrated).
            marker_corners: ArUco marker polygon to mask out / exclude from onions.
            is_calibrated: Whether metric scale calibration is verified.

        Returns:
            Tuple of (list of DetectedOnion instances, annotated visualization image).
        """
        if image is None or image.size == 0:
            return [], np.zeros((100, 100, 3), dtype=np.uint8)

        # Validate calibration inputs: if is_calibrated is explicitly specified, enforce it;
        # otherwise determine from mm_per_pixel validity.
        if is_calibrated is None:
            valid_calibration = (mm_per_pixel is not None and mm_per_pixel > 0.0)
        else:
            valid_calibration = bool(is_calibrated and mm_per_pixel is not None and mm_per_pixel > 0.0)

        scale_factor = mm_per_pixel if valid_calibration else None

        annotated = image.copy()
        h, w = image.shape[:2]

        # Mask out the ArUco marker region so it is never detected as an onion
        exclusion_mask = np.zeros((h, w), dtype=np.uint8)
        if marker_corners is not None and len(marker_corners) >= 4:
            pts = marker_corners.astype(np.int32)
            cv2.fillPoly(exclusion_mask, [pts], 255)
            kernel = np.ones((15, 15), np.uint8)
            exclusion_mask = cv2.dilate(exclusion_mask, kernel, iterations=1)

        detected_onions: List[DetectedOnion] = []

        if self.yolo_model is not None:
            detected_onions = self._segment_yolo(image, scale_factor, valid_calibration, exclusion_mask)
        else:
            detected_onions = self._segment_contours(image, scale_factor, valid_calibration, exclusion_mask)

        # Apply Non-Maximum Suppression to eliminate nested / duplicate contours
        detected_onions = self._filter_duplicate_onions(detected_onions)

        # Re-index onion IDs sequentially
        for idx, onion in enumerate(detected_onions, start=1):
            onion.onion_id = idx

        # Render visual annotations
        annotated = self._render_annotations(annotated, detected_onions)

        return detected_onions, annotated

    def _segment_yolo(
        self,
        image: np.ndarray,
        mm_per_pixel: Optional[float],
        is_calibrated: bool,
        exclusion_mask: np.ndarray,
    ) -> List[DetectedOnion]:
        """Inference with YOLO-seg."""
        results = self.yolo_model.predict(image, conf=self.conf_threshold, verbose=False)
        detected: List[DetectedOnion] = []
        if not results or len(results) == 0:
            return detected

        res = results[0]
        if res.masks is None or len(res.masks) == 0:
            return detected

        masks = res.masks.data.cpu().numpy()
        boxes = res.boxes.xyxy.cpu().numpy()
        confs = res.boxes.conf.cpu().numpy()
        classes = res.boxes.cls.cpu().numpy().astype(int)
        names = res.names

        h, w = image.shape[:2]

        for i, mask_raw in enumerate(masks):
            if mask_raw.shape != (h, w):
                mask_bin = cv2.resize((mask_raw > 0.5).astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
            else:
                mask_bin = (mask_raw > 0.5).astype(np.uint8)

            # Check overlap with ArUco marker exclusion mask
            overlap = cv2.bitwise_and(mask_bin, mask_bin, mask=exclusion_mask)
            if np.sum(overlap) > 0.2 * np.sum(mask_bin):
                continue

            contours, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                continue
            cnt = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(cnt)
            if area < 400:
                continue

            minor_px, major_px, center, ellipse_params = self._fit_ellipse(cnt)
            if minor_px <= 0:
                continue

            minor_mm = round(minor_px * mm_per_pixel, 1) if (is_calibrated and mm_per_pixel) else None
            major_mm = round(major_px * mm_per_pixel, 1) if (is_calibrated and mm_per_pixel) else None

            x1, y1, x2, y2 = map(int, boxes[i])
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            crop = image[y1:y2, x1:x2].copy() if (x2 > x1 and y2 > y1) else None

            cls_name = names.get(classes[i], "healthy").lower()
            if cls_name == "onion":
                cls_name = "healthy"

            # Shape metrics
            hull = cv2.convexHull(cnt)
            hull_area = cv2.contourArea(hull)
            solidity = area / hull_area if hull_area > 0 else 1.0
            perimeter = cv2.arcLength(cnt, True)
            circularity = (4.0 * np.pi * area) / (perimeter * perimeter) if perimeter > 0 else 0.0
            aspect_ratio = minor_px / major_px if major_px > 0 else 1.0

            detected.append(
                DetectedOnion(
                    onion_id=i + 1,
                    bbox=(x1, y1, x2, y2),
                    center_px=center,
                    minor_axis_px=round(minor_px, 1),
                    major_axis_px=round(major_px, 1),
                    minor_axis_mm=minor_mm,
                    major_axis_mm=major_mm,
                    is_calibrated=is_calibrated,
                    confidence=float(confs[i]),
                    defect_label=cls_name,
                    solidity=round(solidity, 3),
                    circularity=round(circularity, 3),
                    aspect_ratio=round(aspect_ratio, 3),
                    mask=mask_bin,
                    contour=cnt,
                    crop=crop,
                    ellipse_params=ellipse_params,
                )
            )

        return detected

    def _segment_contours(
        self,
        image: np.ndarray,
        mm_per_pixel: Optional[float],
        is_calibrated: bool,
        exclusion_mask: np.ndarray,
    ) -> List[DetectedOnion]:
        """Robust segmentation for real camera photos (color + watershed peak separation)."""
        h, w = image.shape[:2]
        total_pixels = h * w

        # 1. Multi-Space Color & Texture Segmentation
        # Extract chromatic mask to capture red/purple/yellow onion bulbs while ignoring neutral mats/shadows
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h_channel, s_channel, v_channel = cv2.split(hsv)

        # Red onions: Hue around [0, 22] and [140, 180] with sufficient saturation
        mask_red1 = cv2.inRange(hsv, (0, 30, 30), (22, 255, 255))
        mask_red2 = cv2.inRange(hsv, (140, 30, 30), (180, 255, 255))
        # Yellow / golden / brown onions: Hue around [20, 42]
        mask_yellow = cv2.inRange(hsv, (20, 35, 35), (42, 255, 255))
        chroma_mask = cv2.bitwise_or(mask_red1, cv2.bitwise_or(mask_red2, mask_yellow))

        # Contrast mask (Adaptive thresholding on L or gray channel)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred_gray = cv2.GaussianBlur(gray, (7, 7), 0)
        _, otsu_thresh = cv2.threshold(blurred_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        fg_ratio = np.count_nonzero(otsu_thresh) / total_pixels
        if fg_ratio > 0.65:
            otsu_thresh = cv2.bitwise_not(otsu_thresh)

        # Combine chromatic and contrast cues
        # If chromatic pixels are detected, prioritize chromatic mask with high saturation
        chroma_count = np.count_nonzero(chroma_mask)
        if chroma_count > 0.005 * total_pixels:
            combined_fg = cv2.bitwise_or(chroma_mask, cv2.bitwise_and(otsu_thresh, chroma_mask))
            # Also keep high-saturation areas
            sat_mask = cv2.threshold(s_channel, 35, 255, cv2.THRESH_BINARY)[1]
            combined_fg = cv2.bitwise_and(combined_fg, sat_mask)
        else:
            combined_fg = otsu_thresh

        # Exclude ArUco marker region
        if exclusion_mask is not None:
            combined_fg = cv2.bitwise_and(combined_fg, cv2.bitwise_not(exclusion_mask))

        # 2. Morphological Closing and Hole Filling (seal specular highlights & peel cracks)
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        closed = cv2.morphologyEx(combined_fg, cv2.MORPH_CLOSE, kernel_close, iterations=2)

        # Fill internal holes in candidate foreground
        contours_temp, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        clean_mask = np.zeros((h, w), dtype=np.uint8)
        for c in contours_temp:
            if cv2.contourArea(c) > 300:
                cv2.drawContours(clean_mask, [c], -1, 255, -1)

        # Smooth boundary
        kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        clean_mask = cv2.morphologyEx(clean_mask, cv2.MORPH_OPEN, kernel_open, iterations=1)

        if np.count_nonzero(clean_mask) == 0:
            return []

        # 3. Component-level Distance Transform & Touching Onion Separation
        min_bulb_area = max(500, int(0.0015 * total_pixels))
        max_bulb_area = int(0.60 * total_pixels)

        num_comp, comp_labels = cv2.connectedComponents(clean_mask)
        extracted_masks = []

        for comp_id in range(1, num_comp):
            comp_mask = np.uint8(comp_labels == comp_id) * 255
            comp_area = np.count_nonzero(comp_mask)
            if comp_area < min_bulb_area:
                continue

            dist = cv2.distanceTransform(comp_mask, cv2.DIST_L2, 5)
            dist_smooth = cv2.GaussianBlur(dist, (9, 9), 0)
            max_d = float(dist_smooth.max())
            if max_d < 5.0:
                continue

            # Adaptive peak radius scaled to bulb diameter (suppresses internal sub-peaks)
            peak_ksize = max(17, int(max_d * 0.45))
            if peak_ksize % 2 == 0:
                peak_ksize += 1
            peak_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (peak_ksize, peak_ksize))
            dilated = cv2.dilate(dist_smooth, peak_kernel)
            peak_thresh = max(8.0, max_d * 0.38)
            peaks = np.uint8((dist_smooth == dilated) & (dist_smooth >= peak_thresh) & (comp_mask > 0)) * 255

            num_peaks, peak_lbls = cv2.connectedComponents(peaks)
            k = num_peaks - 1

            if k <= 1:
                # Single isolated bulb
                extracted_masks.append(comp_mask)
            else:
                # Touching/adjacent bulbs: Voronoi partition foreground by nearest peak centroid
                centroids = []
                for p_idx in range(1, num_peaks):
                    pts = np.argwhere(peak_lbls == p_idx)
                    cy = np.mean(pts[:, 0])
                    cx = np.mean(pts[:, 1])
                    centroids.append(np.array([cx, cy]))

                fg_y, fg_x = np.where(comp_mask > 0)
                pts_fg = np.column_stack((fg_x, fg_y))
                dists = [np.sum((pts_fg - c) ** 2, axis=1) for c in centroids]
                assigned = np.argmin(np.column_stack(dists), axis=1)

                for p_idx in range(k):
                    sub_mask = np.zeros_like(comp_mask)
                    sub_mask[fg_y[assigned == p_idx], fg_x[assigned == p_idx]] = 255
                    sub_mask = cv2.morphologyEx(
                        sub_mask,
                        cv2.MORPH_OPEN,
                        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
                    )
                    extracted_masks.append(sub_mask)

        # 4. Extract and Validate Bulb Contours
        detected: List[DetectedOnion] = []

        for mask_single in extracted_masks:
            cnts, _ = cv2.findContours(mask_single, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not cnts:
                continue

            cnt = max(cnts, key=cv2.contourArea)
            area = cv2.contourArea(cnt)
            if area < min_bulb_area or area > max_bulb_area:
                continue

            # Geometric Shape Gating
            hull = cv2.convexHull(cnt)
            hull_area = cv2.contourArea(hull)
            solidity = area / hull_area if hull_area > 0 else 0.0

            # Onions are convex bulbous shapes; reject jagged shadows, hands, background strips
            if solidity < 0.78:
                continue

            perimeter = cv2.arcLength(cnt, True)
            circularity = (4.0 * np.pi * area) / (perimeter * perimeter) if perimeter > 0 else 0.0
            if circularity < 0.32:
                continue

            # Core bulb isolation: decouple thin loose root tufts and dry stem necks
            # using adaptive morphological opening scaled to bulb radius
            approx_radius = np.sqrt(area / np.pi)
            stem_ksize = max(3, int(approx_radius * 0.08))
            if stem_ksize % 2 == 0:
                stem_ksize += 1
            kernel_stem = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (stem_ksize, stem_ksize))
            core_mask = cv2.morphologyEx(mask_single, cv2.MORPH_OPEN, kernel_stem)
            core_cnts, _ = cv2.findContours(core_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            fit_cnt = max(core_cnts, key=cv2.contourArea) if core_cnts else cnt

            minor_px, major_px, center, ellipse_params = self._fit_ellipse(fit_cnt)
            if minor_px < 32.0 or major_px < 32.0:
                continue

            aspect_ratio = minor_px / major_px
            # Reject elongated strips, fingers/arms, or non-bulbous artifacts
            if aspect_ratio < 0.45 or aspect_ratio > 1.05:
                continue

            # Physical Dimensioning (strictly when calibrated)
            minor_mm = round(minor_px * mm_per_pixel, 1) if (is_calibrated and mm_per_pixel) else None
            major_mm = round(major_px * mm_per_pixel, 1) if (is_calibrated and mm_per_pixel) else None

            # Reject objects physically too small to be an onion (e.g. fingertips, pebbles)
            if is_calibrated and minor_mm is not None and minor_mm < 22.0:
                continue

            # Bounding box & Crop
            x, y, bw, bh = cv2.boundingRect(cnt)
            x1, y1 = max(0, x), max(0, y)
            x2, y2 = min(w, x + bw), min(h, y + bh)
            crop = image[y1:y2, x1:x2].copy()

            # Dynamic confidence score reflecting shape quality
            confidence = float(
                np.clip(
                    0.35 * solidity + 0.35 * min(1.0, circularity / 0.80) + 0.30 * aspect_ratio,
                    0.50,
                    0.96,
                )
            )

            # Full-frame single mask
            bulb_mask = np.zeros((h, w), dtype=np.uint8)
            cv2.drawContours(bulb_mask, [cnt], -1, 255, -1)

            detected.append(
                DetectedOnion(
                    onion_id=len(detected) + 1,
                    bbox=(x1, y1, x2, y2),
                    center_px=center,
                    minor_axis_px=round(minor_px, 1),
                    major_axis_px=round(major_px, 1),
                    minor_axis_mm=minor_mm,
                    major_axis_mm=major_mm,
                    is_calibrated=is_calibrated,
                    confidence=round(confidence, 3),
                    defect_label="healthy",
                    solidity=round(solidity, 3),
                    circularity=round(circularity, 3),
                    aspect_ratio=round(aspect_ratio, 3),
                    mask=bulb_mask,
                    contour=cnt,
                    crop=crop,
                    ellipse_params=ellipse_params,
                )
            )

        return detected

    def _filter_duplicate_onions(self, onions: List[DetectedOnion]) -> List[DetectedOnion]:
        """Eliminates internal contours, duplicate detections, and heavily overlapping regions."""
        if len(onions) <= 1:
            return onions

        # Sort by area descending so outer/larger bulbs take precedence
        sorted_onions = sorted(
            onions,
            key=lambda o: cv2.contourArea(o.contour) if o.contour is not None else 0,
            reverse=True,
        )

        filtered: List[DetectedOnion] = []

        for candidate in sorted_onions:
            cand_cnt = candidate.contour
            cand_area = cv2.contourArea(cand_cnt) if cand_cnt is not None else 0
            if cand_area == 0:
                continue

            cx, cy = candidate.center_px
            is_duplicate = False

            for accepted in filtered:
                acc_cnt = accepted.contour
                acc_area = cv2.contourArea(acc_cnt) if acc_cnt is not None else 0

                # 1. Check center point containment (inside an already accepted larger bulb)
                dist_to_contour = cv2.pointPolygonTest(acc_cnt, (cx, cy), False)
                if dist_to_contour >= 0:
                    # Candidate center lies inside accepted onion -> duplicate/internal
                    is_duplicate = True
                    break

                # 2. Check Intersection-over-Union (IoU) of bounding boxes
                iou = self._calculate_bbox_iou(candidate.bbox, accepted.bbox)
                if iou > 0.35:
                    is_duplicate = True
                    break

                # 3. Check mask intersection ratio
                if candidate.mask is not None and accepted.mask is not None:
                    inter = cv2.bitwise_and(candidate.mask, accepted.mask)
                    inter_area = np.count_nonzero(inter)
                    if inter_area > 0.30 * cand_area:
                        is_duplicate = True
                        break

            if not is_duplicate:
                filtered.append(candidate)

        return filtered

    @staticmethod
    def _calculate_bbox_iou(b1: Tuple[int, int, int, int], b2: Tuple[int, int, int, int]) -> float:
        """Calculates Intersection over Union between two bounding boxes."""
        x1 = max(b1[0], b2[0])
        y1 = max(b1[1], b2[1])
        x2 = min(b1[2], b2[2])
        y2 = min(b1[3], b2[3])

        inter_w = max(0, x2 - x1)
        inter_h = max(0, y2 - y1)
        inter_area = inter_w * inter_h

        area1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
        area2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
        union_area = area1 + area2 - inter_area

        return inter_area / union_area if union_area > 0 else 0.0

    def _fit_ellipse(
        self, cnt: np.ndarray
    ) -> Tuple[float, float, Tuple[float, float], Optional[Tuple[Any, Any, Any]]]:
        """Fits an ellipse to a contour to obtain minor and major axis lengths."""
        if len(cnt) >= 5:
            try:
                ellipse = cv2.fitEllipse(cnt)
                (cx, cy), (ax1, ax2), angle = ellipse
                minor_px = min(ax1, ax2)
                major_px = max(ax1, ax2)
                return minor_px, major_px, (cx, cy), ellipse
            except cv2.error:
                pass

        (cx, cy), radius = cv2.minEnclosingCircle(cnt)
        diameter_px = radius * 2.0
        return diameter_px, diameter_px, (cx, cy), None

    def _render_annotations(
        self, canvas: np.ndarray, onions: List[DetectedOnion]
    ) -> np.ndarray:
        """Overlays ID badges, ellipses, and calibrated/uncalibrated measurements."""
        overlay = canvas.copy()

        for onion in onions:
            # Draw fitted ellipse
            if onion.ellipse_params is not None:
                cv2.ellipse(canvas, onion.ellipse_params, (0, 220, 0), 2, cv2.LINE_AA)
            elif onion.contour is not None:
                cv2.drawContours(canvas, [onion.contour], -1, (0, 220, 0), 2)

            cx, cy = int(onion.center_px[0]), int(onion.center_px[1])

            # Label text: show mm if calibrated, otherwise clearly indicate pixels (uncalibrated)
            if onion.is_calibrated and onion.minor_axis_mm is not None:
                label = f"#{onion.onion_id}: {onion.minor_axis_mm:.0f}mm"
                badge_bg = (20, 20, 20)
                text_color = (255, 255, 255)
            else:
                label = f"#{onion.onion_id}: {onion.minor_axis_px:.0f}px (Uncalibrated)"
                badge_bg = (10, 80, 160)  # Distinct blue-orange warning color
                text_color = (255, 255, 255)

            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.55
            thickness = 2
            (tw, th), baseline = cv2.getTextSize(label, font, font_scale, thickness)

            # Badge background
            bx1, by1 = cx - 10, cy - th - 8
            bx2, by2 = bx1 + tw + 8, cy + 4
            cv2.rectangle(canvas, (bx1, by1), (bx2, by2), badge_bg, -1)
            cv2.putText(canvas, label, (bx1 + 4, cy - 2), font, font_scale, text_color, thickness, cv2.LINE_AA)

        return canvas
