"""Onion Grading Automation - Streamlit Web Application.

Phone-friendly, real-time quality grading application:
1. Capture protocol checklist & marker generator
2. Camera capture or photo upload
3. Pre-flight input quality gating (sharpness, lighting, ArUco presence)
4. AI Segmentation, ellipse-fitted sizing & defect classification
5. YAML rule evaluation (Grade A, URS, Reject, Review)
6. Interactive human override with mandatory audit logging
7. Tamper-evident PDF certificate generation with SHA-256 verification
"""

import io
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional

# Ensure project root is in sys.path so 'src.*' modules import reliably
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import cv2
import numpy as np
from PIL import Image
import streamlit as st

from src.rules import load_grading_rules, GradingRuleEngine, GradingResult, LotSummary
from src.scale import detect_aruco_marker, generate_aruco_marker
from src.capture_check import check_capture_quality, CaptureQualityResult
from src.segment import OnionSegmenter, DetectedOnion
from src.defects import DefectClassifier
from src.report import InspectionReportGenerator, calculate_sha256
from src.db import Database
from src.calibration import (
    load_camera_calibration,
    save_camera_calibration,
    clear_camera_calibration,
    calibrate_from_pixels,
    calibrate_from_marker,
    is_valid_camera_calibration,
    auto_detect_reference_scale,
    get_standard_camera_calibration,
    CALIBRATION_PRESETS,
    CameraCalibration,
)


# Page Configuration
st.set_page_config(
    page_title="Onion Quality & Grading Automation",
    page_icon="🧅",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for phone-friendly and clean visual presentation
st.markdown("""
<style>
    .metric-card {
        background-color: #F8FAFC;
        border-radius: 8px;
        padding: 12px 16px;
        border: 1px solid #E2E8F0;
        margin-bottom: 8px;
        text-align: center;
    }
    .badge-grade-a {
        color: #22543D;
        background-color: #C6F6D5;
        padding: 3px 8px;
        border-radius: 4px;
        font-weight: bold;
    }
    .badge-urs {
        color: #744210;
        background-color: #FEFCBF;
        padding: 3px 8px;
        border-radius: 4px;
        font-weight: bold;
    }
    .badge-reject {
        color: #742A2A;
        background-color: #FED7D7;
        padding: 3px 8px;
        border-radius: 4px;
        font-weight: bold;
    }
    .badge-review {
        color: #2C5282;
        background-color: #BEE3F8;
        padding: 3px 8px;
        border-radius: 4px;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def get_rule_engine() -> GradingRuleEngine:
    return load_grading_rules()


@st.cache_resource
def get_segmenter() -> OnionSegmenter:
    model_path = Path("models/onion_yolo_seg.pt")
    return OnionSegmenter(model_path=model_path if model_path.exists() else None)


@st.cache_resource
def get_defect_classifier() -> DefectClassifier:
    return DefectClassifier()


@st.cache_resource
def get_database() -> Database:
    return Database()


@st.cache_resource
def get_report_generator() -> InspectionReportGenerator:
    return InspectionReportGenerator()


def main():
    engine = get_rule_engine()
    segmenter = get_segmenter()
    defect_classifier = get_defect_classifier()
    db = get_database()
    report_gen = get_report_generator()

    st.title("🧅 Onion Grading Automation")
    st.caption("AI-Powered Optical Sizing, Defect Grading, and Procurement Certification")

    # Display prominent placeholder warning banner if active
    if engine.is_placeholder:
        st.warning(
            f"⚠️ **{engine.disclaimer}**\n\n"
            f"Rules Version: `{engine.rules_version}` | Grade A standard: {engine.config.get('size_mm', {}).get('grade_a', {}).get('min', 45)}-"
            f"{engine.config.get('size_mm', {}).get('grade_a', {}).get('max', 65)} mm | "
            f"URS: Under Relaxed Specifications (35-45 mm / minor defects)."
        )

    # Sidebar: Setup, Calibration, and Protocol
    with st.sidebar:
        st.header("📋 Inspection Setup")
        st.markdown(
            "**Capture Protocol Checklist:**\n"
            "- [x] Single layer on plain contrasting mat\n"
            "- [x] Bulbs separated where possible\n"
            "- [x] Fixed camera overhead mount (~30-40cm)\n"
            "- [x] Uniform illumination, no direct flash glare"
        )
        st.divider()

        # Persistent Camera Calibration Management
        st.subheader("📐 Camera Scale Calibration")
        calib = load_camera_calibration()
        calibrated = is_valid_camera_calibration(calib)

        if calibrated:
            st.success(f"🟢 **Status: Calibrated** (`{calib.pixels_per_mm:.2f} px/mm`)")
            st.caption(
                f"• Reference: **{calib.reference_name}**\n"
                f"• Ratio: `{calib.pixels_per_mm:.2f} px/mm` (`{calib.mm_per_pixel:.4f} mm/px`)\n"
                f"• Mount Height: `{calib.fixed_height_cm} cm`\n"
                f"• Method: `{calib.method}`\n"
                f"• Profile Saved: `{calib.calibrated_at[:10] if calib.calibrated_at else 'Active'}`"
            )
            col_recal, col_reset = st.columns(2)
            with col_recal:
                if st.button("🔧 Recalibrate", key="btn_recalib_toggle", use_container_width=True):
                    st.session_state["show_calib_assistant"] = not st.session_state.get("show_calib_assistant", False)
            with col_reset:
                if st.button("🔄 Reset Scale", key="btn_clear_calib", use_container_width=True):
                    clear_camera_calibration()
                    st.session_state["show_calib_assistant"] = True
                    st.rerun()
        else:
            st.warning("🟡 **Status: Not Calibrated**\nPhysical millimetre sizing disabled.")

        show_assistant = st.session_state.get("show_calib_assistant", not calibrated)
        with st.expander("⚙️ Scale Calibration Assistant", expanded=show_assistant):
            st.markdown(
                "Calibrate the camera once using a standard ruler, preset object, or auto-detection. "
                "The calibration parameters are saved permanently in `config/camera_calibration.json` "
                "and reused across all subsequent inspection batches."
            )

            calib_choice = st.radio(
                "Calibration Method:",
                [
                    "Standard Ruler (Recommended)",
                    "Reference Object Preset",
                    "Auto-Detect Scale from Photo",
                    "One-Time ArUco Marker",
                ],
                key="calib_choice",
            )

            if calib_choice == "Standard Ruler (Recommended)":
                st.markdown("**Calibrate with a Standard Ruler**")
                r_col1, r_col2 = st.columns(2)
                with r_col1:
                    known_mm = st.number_input(
                        "Ruler Length (mm):",
                        min_value=5.0,
                        max_value=1000.0,
                        value=100.0,
                        step=5.0,
                        key="input_ruler_mm",
                    )
                with r_col2:
                    default_px = float(round(known_mm * (calib.pixels_per_mm if calibrated else 3.80), 1))
                    measured_px = st.number_input(
                        "Pixel Span (px):",
                        min_value=1.0,
                        max_value=8000.0,
                        value=default_px,
                        step=1.0,
                        key="input_ruler_px",
                    )

                calc_ratio = measured_px / known_mm if known_mm > 0 else 0
                st.info(f"📐 Calculated Scale: **`{calc_ratio:.2f} px/mm`** (`{1.0/calc_ratio:.4f} mm/px`)")

                ruler_name = st.text_input(
                    "Reference Label:",
                    value=f"Standard {known_mm:.0f}mm Reference Ruler",
                    key="input_ruler_name",
                )
                mount_height = st.number_input(
                    "Camera Mount Height (cm):",
                    value=float(calib.fixed_height_cm if (calibrated and calib.fixed_height_cm) else 35.0),
                    step=1.0,
                    key="ruler_h",
                )
                if st.button("💾 Save Calibration Profile", key="btn_save_ruler_calib", type="primary", use_container_width=True):
                    new_cal = calibrate_from_pixels(
                        pixel_length=measured_px,
                        physical_mm=known_mm,
                        method="ruler",
                        reference_name=ruler_name,
                        fixed_height_cm=mount_height,
                    )
                    st.session_state["show_calib_assistant"] = False
                    st.success(f"✅ Calibration saved! Scale: {new_cal.pixels_per_mm:.2f} px/mm")
                    st.rerun()

            elif calib_choice == "Reference Object Preset":
                st.markdown("**Calibrate with Known Reference Object**")
                selected_preset = st.selectbox(
                    "Choose Reference Object:",
                    list(CALIBRATION_PRESETS.keys()),
                    key="select_preset",
                )
                preset_mm = CALIBRATION_PRESETS[selected_preset]
                st.caption(f"Standard Dimension: **{preset_mm:.1f} mm**")

                default_preset_px = float(round(preset_mm * (calib.pixels_per_mm if calibrated else 3.80), 1))
                measured_px = st.number_input(
                    f"Measured span in pixels for {preset_mm:.1f} mm:",
                    min_value=10.0,
                    max_value=8000.0,
                    value=default_preset_px,
                    step=1.0,
                    key="input_preset_px",
                )
                calc_ratio = measured_px / preset_mm if preset_mm > 0 else 0
                st.info(f"📐 Calculated Scale: **`{calc_ratio:.2f} px/mm`** (`{1.0/calc_ratio:.4f} mm/px`)")

                mount_height = st.number_input(
                    "Camera Mount Height (cm):",
                    value=float(calib.fixed_height_cm if (calibrated and calib.fixed_height_cm) else 35.0),
                    step=1.0,
                    key="preset_h",
                )
                if st.button("💾 Save Calibration Profile", key="btn_save_preset_calib", type="primary", use_container_width=True):
                    new_cal = calibrate_from_pixels(
                        pixel_length=measured_px,
                        physical_mm=preset_mm,
                        method="preset_object",
                        reference_name=selected_preset,
                        fixed_height_cm=mount_height,
                    )
                    st.session_state["show_calib_assistant"] = False
                    st.success(f"✅ Calibration saved! Scale: {new_cal.pixels_per_mm:.2f} px/mm")
                    st.rerun()

            elif calib_choice == "Auto-Detect Scale from Photo":
                st.markdown("**Automatic Scale Detection from Inspection Photo**")
                st.caption("Scans the current photo for a standard reference card (ID card) or ArUco marker.")
                if "uploaded_image_bytes" in st.session_state:
                    if st.button("⚡ Auto-Detect Scale Now", key="btn_auto_detect_scale", type="primary", use_container_width=True):
                        arr = np.frombuffer(st.session_state["uploaded_image_bytes"], np.uint8)
                        curr_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                        auto_cal, msg = auto_detect_reference_scale(curr_bgr)
                        if auto_cal is not None and auto_cal.is_calibrated:
                            st.session_state["show_calib_assistant"] = False
                            st.success(f"✅ {msg}")
                            st.rerun()
                        else:
                            st.warning(f"⚠️ {msg} Please use the Standard Ruler method above.")
                else:
                    st.info("ℹ️ Upload or capture a photo first to auto-detect scale.")

            elif calib_choice == "One-Time ArUco Marker":
                st.markdown("**Calibrate using ArUco Marker Snapshot**")
                st.caption("Place a printed ArUco marker under the camera once. After calibration, remove it forever.")
                marker_size = st.number_input("Marker Dimension (mm):", value=50.0, step=5.0, key="input_marker_mm")
                if st.button("📸 Calibrate from Photo Frame", key="btn_calib_marker", type="primary", use_container_width=True):
                    if "uploaded_image_bytes" in st.session_state:
                        arr = np.frombuffer(st.session_state["uploaded_image_bytes"], np.uint8)
                        curr_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                        c_res, m_res = calibrate_from_marker(curr_bgr, physical_size_mm=marker_size)
                        if c_res.is_calibrated:
                            st.session_state["show_calib_assistant"] = False
                            st.success(f"✅ Calibrated successfully! Scale: {c_res.pixels_per_mm:.2f} px/mm.")
                            st.rerun()
                        else:
                            st.error(f"❌ Marker not detected: {m_res.error_message}")
                    else:
                        st.error("Capture or upload a photo containing the marker first.")

        with st.expander("🖨️ Printable ArUco Marker (Optional Tool)"):
            marker_id = engine.marker_config.get("id", 0)
            dict_name = engine.marker_config.get("dictionary", "DICT_4X4_50")
            size_mm = engine.marker_config.get("size_mm", 50.0)
            st.caption(f"Dictionary: `{dict_name}` | ID: `{marker_id}` | Size: `{size_mm} mm`")
            marker_img = generate_aruco_marker(marker_id=marker_id, dict_name=dict_name, size_pixels=300)
            buf = io.BytesIO()
            Image.fromarray(marker_img).save(buf, format="PNG")
            st.image(marker_img, caption=f"ArUco Marker (ID: {marker_id})", width=130)
            st.download_button(
                label="⬇️ Download Marker PNG",
                data=buf.getvalue(),
                file_name=f"aruco_marker_id{marker_id}_{dict_name}.png",
                mime="image/png",
            )
        st.divider()

        st.subheader("📜 Recent Lots History")
        recent_lots = db.list_lots()
        if recent_lots:
            for l in recent_lots[:5]:
                st.text(f"• {l['lot_id']} ({l['total_count']} onions, A: {l['grade_a_pct']}%)")
        else:
            st.caption("No inspected lots recorded yet.")

    # Main Tabs
    tab_inspect, tab_demo, tab_rules = st.tabs(["📸 New Inspection", "🧪 Synthetic Demo Generator", "⚙️ Active Rules Config"])

    with tab_rules:
        st.subheader("Active Grading Rules Configuration")
        st.markdown("All rules are dynamically loaded from `config/grading_rules.yaml`:")
        st.json(engine.config)

    with tab_demo:
        st.subheader("Synthetic Benchmark Lot Generator")
        st.markdown(
            "Generate a calibrated synthetic lot with known bulb sizes, simulated defects (rotten, sprouted), "
            "and an embedded 50mm ArUco marker for automated testing."
        )
        if st.button("Generate & Load Synthetic Lot Sample", type="secondary"):
            synthetic_img = create_synthetic_sample_image(marker_id=marker_id, dict_name=dict_name)
            st.session_state["uploaded_image_bytes"] = cv2.imencode(".png", synthetic_img)[1].tobytes()
            st.session_state["image_name"] = "synthetic_lot_sample.png"
            st.session_state["active_input_source"] = "demo"
            st.success("Synthetic sample generated! Switch to the 'New Inspection' tab to view results.")

    with tab_inspect:
        col_input1, col_input2 = st.columns([1, 1])

        with col_input1:
            capture_mode = st.radio("Capture Input Method:", ["Mac Camera Capture", "File Upload"], horizontal=True)
            active_source = st.session_state.get("active_input_source", None)

            if capture_mode == "Mac Camera Capture":
                camera_file = st.camera_input("📸 Capture Photo using MacBook Camera:", key="webcam_capture")
                if camera_file is not None:
                    raw_bytes = camera_file.getvalue()
                    st.session_state["uploaded_image_bytes"] = raw_bytes
                    st.session_state["image_name"] = "mac_camera_capture.jpg"
                    st.session_state["active_input_source"] = "camera"
                else:
                    if active_source != "camera" and "uploaded_image_bytes" in st.session_state:
                        # Clear old uploaded file so it doesn't accidentally analyze stale data
                        st.session_state.pop("uploaded_image_bytes", None)
                        st.session_state.pop("image_name", None)
                        st.session_state["active_input_source"] = "camera"

                    st.info(
                        "📷 **Mac Camera Ready**: Position onions in a single layer on a contrasting mat "
                        "under your MacBook camera and click **Take Photo** above."
                    )
                    with st.expander("ℹ️ Camera Permissions for macOS & Chrome"):
                        st.markdown(
                            "1. In Chrome, click the camera icon next to `localhost` in the URL bar and select **Allow**.\n"
                            "2. In macOS: open **System Settings > Privacy & Security > Camera** and ensure **Google Chrome** is toggled ON.\n"
                            "3. If prompted by the browser, click **Allow** to enable live video stream."
                        )

            else:  # File Upload
                uploaded_file = st.file_uploader("Upload Onion Inspection Photo:", type=["jpg", "jpeg", "png", "webp"], key="file_upload_input")
                if uploaded_file is not None:
                    st.session_state["uploaded_image_bytes"] = uploaded_file.getvalue()
                    st.session_state["image_name"] = getattr(uploaded_file, "name", "uploaded_photo.jpg")
                    st.session_state["active_input_source"] = "file"
                else:
                    if active_source == "camera" and "uploaded_image_bytes" in st.session_state:
                        st.session_state.pop("uploaded_image_bytes", None)
                        st.session_state.pop("image_name", None)
                        st.session_state["active_input_source"] = "file"

            if "uploaded_image_bytes" in st.session_state:
                if st.button("🔄 Reset / Clear Image", key="btn_reset_active_img"):
                    st.session_state.pop("uploaded_image_bytes", None)
                    st.session_state.pop("image_name", None)
                    st.session_state.pop("active_input_source", None)
                    st.rerun()

        with col_input2:
            if "uploaded_image_bytes" in st.session_state:
                raw_bytes = st.session_state["uploaded_image_bytes"]
                img_array = np.frombuffer(raw_bytes, np.uint8)
                image_bgr = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

                st.markdown("**Image Pre-Flight Quality Check:**")
                thresholds = engine.thresholds_config
                blur_thresh = float(thresholds.get("blur_var_threshold", 80.0))
                min_bright = float(thresholds.get("brightness_min", 45.0))
                max_bright = float(thresholds.get("brightness_max", 235.0))

                # Check if camera calibration is already available
                current_calib = load_camera_calibration()
                calib_active = is_valid_camera_calibration(current_calib)

                quality_res = check_capture_quality(
                    image=image_bgr,
                    blur_threshold=blur_thresh,
                    min_brightness=min_bright,
                    max_brightness=max_bright,
                    marker_dict=dict_name,
                    marker_size_mm=size_mm,
                    target_marker_id=marker_id,
                    require_marker=False,
                    camera_calibrated=calib_active,
                )

                if quality_res.passed:
                    st.success("✅ **Quality Check Passed**: Sharp focus and balanced illumination verified.")
                    if calib_active:
                        st.caption(f"📏 Scale: **{current_calib.pixels_per_mm:.2f} px/mm** ({current_calib.reference_name}) | Sharpness: **{quality_res.blur_score:.1f}**")
                    else:
                        st.caption(f"⚠️ Scale: **Uncalibrated** (advisory mode) | Sharpness: **{quality_res.blur_score:.1f}**")
                else:
                    st.error("❌ **Quality Pre-Check Warning:**")
                    for reason in quality_res.rejection_reasons:
                        st.markdown(f"- {reason}")
                    st.caption("Analysis will proceed in advisory mode, but capture protocol should be followed.")

        # If image is available, run grading pipeline
        if "uploaded_image_bytes" in st.session_state:
            raw_bytes = st.session_state["uploaded_image_bytes"]
            img_array = np.frombuffer(raw_bytes, np.uint8)
            image_bgr = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            image_sha = calculate_sha256(raw_bytes)

            st.divider()
            lot_input_col1, lot_input_col2 = st.columns([2, 1])
            with lot_input_col1:
                lot_id = st.text_input("Lot Identifier / Batch Number:", value=f"LOT-{datetime.utcnow().strftime('%Y%m%d-%H%M')}")
            with lot_input_col2:
                operator_name = st.text_input("Inspector Name / ID:", value="Auditor-01")

            # Scale Determination: Persistent Camera Calibration takes precedence
            active_calib = load_camera_calibration()
            marker_corners = None

            if is_valid_camera_calibration(active_calib):
                is_calibrated = True
                mm_per_pixel = active_calib.mm_per_pixel
                pixels_per_mm = active_calib.pixels_per_mm
                st.success(f"📏 **Physical Scale Active**: {active_calib.reference_name} ({pixels_per_mm:.2f} px/mm). Metric sizing active.")
            else:
                # Fallback to marker if in photo
                marker_res = detect_aruco_marker(image_bgr, target_id=marker_id, physical_size_mm=size_mm, dict_name=dict_name)
                if marker_res.found and marker_res.is_calibrated:
                    is_calibrated = True
                    mm_per_pixel = marker_res.mm_per_pixel
                    pixels_per_mm = marker_res.pixels_per_mm
                    marker_corners = marker_res.corner_points
                    st.info(f"📐 **ArUco Scale Detected**: {pixels_per_mm:.2f} px/mm (ID {marker_res.marker_id}).")
                else:
                    is_calibrated = False
                    mm_per_pixel = None
                    pixels_per_mm = None
                    st.warning(
                        "⚠️ **Scale Uncalibrated**: Physical millimetre sizing disabled. "
                        "Bulb sizes are displayed in pixels and size grading is marked as **'Needs Calibration'**. "
                        "Use the **Camera Scale Calibration** tool in the sidebar to calibrate using a ruler or known reference object."
                    )

            detected_onions, annotated_img = segmenter.segment_and_measure(
                image_bgr,
                mm_per_pixel=mm_per_pixel,
                marker_corners=marker_corners,
                is_calibrated=is_calibrated,
            )

            # Defect classification and rule engine
            grading_results: List[GradingResult] = []
            for onion in detected_onions:
                x1, y1, x2, y2 = onion.bbox
                crop_mask = onion.mask[y1:y2, x1:x2] if onion.mask is not None else None
                defect_label, defect_conf, _ = defect_classifier.classify_crop(
                    onion.crop, onion.defect_label, mask=crop_mask
                )
                onion.defect_label = defect_label
                combined_conf = (onion.confidence + defect_conf) / 2.0

                result = engine.grade_onion(
                    onion_id=onion.onion_id,
                    diameter_mm=onion.minor_axis_mm,
                    defect_label=defect_label,
                    confidence=combined_conf,
                    is_calibrated=is_calibrated,
                )
                grading_results.append(result)

            lot_summary = engine.summarize_lot(grading_results)

            # Store in session state for overrides with cache invalidation check
            session_key = f"lot_{lot_id}_results"
            cached = st.session_state.get(session_key)
            needs_refresh = (
                cached is None
                or cached.get("image_sha256") != image_sha
                or cached.get("is_calibrated") != is_calibrated
                or cached.get("pixels_per_mm") != pixels_per_mm
            )
            if needs_refresh:
                st.session_state[session_key] = {
                    "grading_results": grading_results,
                    "lot_summary": lot_summary,
                    "annotated_bgr": annotated_img,
                    "image_sha256": image_sha,
                    "is_calibrated": is_calibrated,
                    "pixels_per_mm": pixels_per_mm,
                }

            current_data = st.session_state[session_key]
            current_results = current_data["grading_results"]
            current_summary = engine.summarize_lot(current_results)

            # Summary Metrics Display
            st.subheader(f"📊 Lot Quality Summary: {lot_id}")
            m1, m2, m3, m4, m5, m6 = st.columns(6)
            m1.metric("Total Count", current_summary.total_count)
            m2.metric("Grade A", f"{current_summary.grade_a_pct:.1f}%", f"{current_summary.grade_a_count} bulbs")
            m3.metric("URS %", f"{current_summary.urs_pct:.1f}%", f"{current_summary.urs_count} bulbs")
            m4.metric("Reject %", f"{current_summary.reject_pct:.1f}%", f"{current_summary.reject_count} bulbs")
            if current_summary.uncalibrated_count > 0:
                m5.metric("Uncalibrated", f"{current_summary.uncalibrated_pct:.1f}%", f"{current_summary.uncalibrated_count} bulbs")
            else:
                m5.metric("Review %", f"{current_summary.needs_review_pct:.1f}%", f"{current_summary.needs_review_count} bulbs")

            if current_summary.is_calibrated and current_summary.avg_diameter_mm is not None:
                m6.metric(
                    "Mean Size",
                    f"{current_summary.avg_diameter_mm:.1f} mm",
                    f"Range: {current_summary.min_diameter_mm:.0f}-{current_summary.max_diameter_mm:.0f} mm",
                )
            else:
                m6.metric("Mean Size", "Calibration Required", "Camera uncalibrated")

            # Visual Evidence & Interactive Results
            col_img, col_details = st.columns([3, 2])

            with col_img:
                st.markdown("**Computer Vision Segmentation & Calibration Overlay:**")
                annotated_rgb = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
                st.image(annotated_rgb, use_container_width=True)

            with col_details:
                st.markdown("**Detected Onion Breakdown:**")
                onion_rows = []
                for r in current_results:
                    size_display = (
                        f"{r.diameter_mm:.1f} mm"
                        if (r.is_calibrated and r.diameter_mm is not None)
                        else "Calibration Required"
                    )
                    onion_rows.append({
                        "ID": r.onion_id,
                        "Size": size_display,
                        "Defect": r.defect_label,
                        "Confidence": f"{r.confidence:.2f}",
                        "Grade": r.grade,
                        "Reason": r.reason,
                    })
                st.dataframe(onion_rows, height=350, use_container_width=True)

            # Human Override Section
            st.divider()
            st.subheader("✍️ Human Expert Override (Auditor Gate)")
            st.markdown(
                "If optical inspection incorrectly identified a bulb (e.g. concealed bottom rot or false blemish), "
                "an auditor can override the grade with a **mandatory justification reason**."
            )

            ov_col1, ov_col2, ov_col3, ov_col4 = st.columns([1, 1, 2, 1])
            with ov_col1:
                available_ids = [r.onion_id for r in current_results]
                selected_onion_id = st.selectbox("Select Onion ID:", available_ids if available_ids else [1])
            with ov_col2:
                new_grade = st.selectbox("New Assigned Grade:", ["Grade A", "URS", "Reject", "Needs Human Review"])
            with ov_col3:
                override_reason = st.text_input("Mandatory Justification Reason:", placeholder="e.g. Verified internal neck rot upon cut")
            with ov_col4:
                st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
                apply_override = st.button("Apply Override", type="primary")

            if apply_override:
                if not override_reason.strip():
                    st.error("❌ Justification reason is required for any human override.")
                else:
                    # Update result in session state
                    for r in current_results:
                        if r.onion_id == selected_onion_id:
                            old_grade = r.grade
                            r.grade = new_grade
                            r.reason = f"Auditor override: {override_reason.strip()} (was {old_grade})"
                            break

                    # Save to DB
                    db.record_override(
                        lot_id=lot_id,
                        onion_id=selected_onion_id,
                        new_grade=new_grade,
                        reason=override_reason,
                        operator=operator_name,
                    )
                    st.success(f"Updated Onion #{selected_onion_id} to '{new_grade}'. Lot metrics refreshed!")
                    st.rerun()

            # Report Generation Section
            st.divider()
            st.subheader("📑 Tamper-Evident PDF Inspection Certificate")
            st.markdown(
                "Generate a cryptographically secured PDF report containing lot summary, "
                "the annotated visual evidence, per-onion measurements, and human override logs."
            )

            if st.button("Generate Official Inspection PDF", type="primary"):
                with st.spinner("Generating PDF report and computing cryptographic checksums..."):
                    overrides_list = db.get_overrides(lot_id)
                    onions_dict_list = [
                        {
                            "onion_id": r.onion_id,
                            "diameter_mm": r.diameter_mm,
                            "defect_label": r.defect_label,
                            "confidence": r.confidence,
                            "initial_grade": r.grade,
                            "final_grade": r.grade,
                            "initial_reason": r.reason,
                        }
                        for r in current_results
                    ]

                    pdf_path, pdf_hash = report_gen.generate_pdf(
                        lot_id=lot_id,
                        lot_summary=current_summary.__dict__,
                        onions=onions_dict_list,
                        annotated_image=annotated_img,
                        image_sha256=image_sha,
                        overrides=overrides_list,
                        operator_name=operator_name,
                    )

                    with open(pdf_path, "rb") as f:
                        pdf_bytes = f.read()

                    st.success(f"Certificate Generated! Report SHA-256: `{pdf_hash}`")
                    st.download_button(
                        label="⬇️ Download Inspection Certificate (PDF)",
                        data=pdf_bytes,
                        file_name=pdf_path.name,
                        mime="application/pdf",
                    )


def create_synthetic_sample_image(marker_id: int = 0, dict_name: str = "DICT_4X4_50") -> np.ndarray:
    """Generates a realistic synthetic test image containing onions and an ArUco marker."""
    # 800x1000 dark navy capture mat
    h, w = 800, 1000
    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    canvas[:, :] = (35, 25, 20)  # Dark background mat

    # Add ArUco marker at top right (200x200 px -> for 50mm marker = 4 px/mm)
    marker = generate_aruco_marker(marker_id=marker_id, dict_name=dict_name, size_pixels=200)
    marker_bgr = cv2.cvtColor(marker, cv2.COLOR_GRAY2BGR)
    canvas[40:240, 750:950] = marker_bgr

    # 4 px per mm => Grade A (45-65 mm) = 180 - 260 px diameter (radius 90-130)
    # URS undersize (35-45 mm) = 140 - 180 px diameter (radius 70-90)
    # Positions and bulbs
    bulbs = [
        # (center_x, center_y, radius_x, radius_y, angle, defect_type)
        (250, 250, 105, 110, 15, "healthy"),   # ~55mm Grade A
        (480, 220, 115, 118, -10, "healthy"),  # ~58mm Grade A
        (200, 520, 80, 82, 5, "healthy"),      # ~40mm URS (undersize)
        (450, 500, 100, 105, 20, "sprouted"),  # ~52mm URS (sprouted)
        (720, 520, 110, 112, 0, "rotten"),     # ~55mm Reject (rotten)
        (350, 680, 108, 112, -25, "healthy"),  # ~55mm Grade A
        (650, 700, 78, 80, 10, "healthy"),     # ~39mm URS (undersize)
    ]

    for cx, cy, rx, ry, ang, defect in bulbs:
        # Draw base onion (warm brownish-golden gradient)
        cv2.ellipse(canvas, (cx, cy), (rx, ry), ang, 0, 360, (40, 130, 205), -1, cv2.LINE_AA)
        cv2.ellipse(canvas, (cx, cy), (int(rx * 0.9), int(ry * 0.9)), ang, 0, 360, (50, 150, 225), -1, cv2.LINE_AA)
        cv2.ellipse(canvas, (cx, cy), (rx, ry), ang, 0, 360, (20, 90, 160), 2, cv2.LINE_AA)

        if defect == "sprouted":
            # Add prominent green sprout extending out
            cv2.ellipse(canvas, (cx, cy - ry + 10), (16, 40), ang, 0, 360, (20, 220, 40), -1, cv2.LINE_AA)
        elif defect == "rotten":
            # Add dark blackish necrotic lesion
            cv2.ellipse(canvas, (cx + 20, cy + 15), (32, 28), 15, 0, 360, (15, 15, 15), -1, cv2.LINE_AA)

    return canvas


if __name__ == "__main__":
    main()
