"""Unit tests for onion segmentation, ellipse fitting, and defect detection."""

import pytest
import numpy as np
import cv2
from src.segment import OnionSegmenter, DetectedOnion
from src.defects import DefectClassifier


def test_fit_ellipse_and_measure_known_bulb():
    # 600x600 dark canvas (representing dark mat) with 2 bright onion bulbs
    canvas = np.zeros((600, 600, 3), dtype=np.uint8)

    # Bulb 1: circle at (200, 200), radius 80 px (diameter = 160 px)
    # Scale: 4 px/mm -> expected diameter = 160 / 4 = 40.0 mm (URS)
    cv2.circle(canvas, (200, 200), 80, (200, 200, 200), -1)

    # Bulb 2: ellipse at (420, 350), axes (110, 100), angle 30 (minor axis ~200 px)
    # Scale: 4 px/mm -> expected minor axis = 200 / 4 = 50.0 mm (Grade A)
    cv2.ellipse(canvas, (420, 350), (100, 110), 30, 0, 360, (200, 200, 200), -1)

    segmenter = OnionSegmenter(model_path=None)
    mm_per_pixel = 1.0 / 4.0  # 4 px per mm

    onions, annotated = segmenter.segment_and_measure(canvas, mm_per_pixel=mm_per_pixel)

    assert len(onions) == 2
    diameters = sorted([o.minor_axis_mm for o in onions])

    # Check that measured diameters match within ~1.5 mm
    assert abs(diameters[0] - 40.0) < 2.0
    assert abs(diameters[1] - 50.0) < 2.5
    assert annotated.shape == canvas.shape


def test_exclude_marker_region():
    canvas = np.zeros((500, 500, 3), dtype=np.uint8)
    # Place onion bulb at (350, 350)
    cv2.circle(canvas, (350, 350), 70, (220, 220, 220), -1)

    # Define marker corners at top-left
    marker_corners = np.array([[50, 50], [150, 50], [150, 150], [50, 150]], dtype=np.float32)

    # Draw fake marker rectangle in canvas
    cv2.rectangle(canvas, (50, 50), (150, 150), (255, 255, 255), -1)

    segmenter = OnionSegmenter(model_path=None)
    onions, _ = segmenter.segment_and_measure(
        canvas, mm_per_pixel=0.25, marker_corners=marker_corners
    )

    # Only the onion should be detected, not the marker
    assert len(onions) == 1
    assert onions[0].center_px[0] > 200


def test_defect_classifier_healthy():
    classifier = DefectClassifier()
    # Uniform light brown/orange crop
    crop = np.zeros((100, 100, 3), dtype=np.uint8)
    crop[:, :] = (50, 140, 210)  # Golden brown in BGR

    label, conf, details = classifier.classify_crop(crop)
    assert label == "healthy"
    assert conf >= 0.70


def test_defect_classifier_sprouted():
    classifier = DefectClassifier()
    # Healthy base with green shoot on top
    crop = np.zeros((100, 100, 3), dtype=np.uint8)
    crop[:, :] = (50, 140, 210)
    # Add green sprout patch (BGR: high green, low red/blue)
    crop[0:30, 40:60] = (20, 220, 40)

    label, conf, details = classifier.classify_crop(crop)
    assert label == "sprouted"
    assert "green_ratio" in details


def test_defect_classifier_rotten():
    classifier = DefectClassifier()
    # Crop with significant black fungal necrotic patch
    crop = np.zeros((100, 100, 3), dtype=np.uint8)
    crop[:, :] = (50, 140, 210)
    # Add black rot lesion (>6% area)
    crop[30:70, 30:70] = (10, 10, 10)

    label, conf, details = classifier.classify_crop(crop)
    assert label == "rotten"
    assert "rot_ratio" in details
