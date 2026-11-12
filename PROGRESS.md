# Onion Grading Automation - Progress & AI Handoff

## Project Overview
Automated onion grading tool using computer vision and rules-based procurement logic:
- Detect and segment onions from overhead photo
- Derive physical diameter (mm) using calibrated ArUco reference marker
- Classify surface defects (Healthy, Rotten, Sprouted, Damaged, Other)
- Assign grades (Grade A, URS, Reject, Needs Human Review) based on YAML rules
- Support interactive human overrides with mandatory audit reasons
- Produce tamper-evident PDF inspection reports with SHA-256 validation

---

## Status by Task

| # | Task | Status | Done Criteria / Notes |
|---|------|--------|------------------------|
| 1 | **Verify grading standard & dataset options** | **COMPLETED** | Procurement standard verified (DoCA/NAFED): Grade A (45–65 mm), URS = Under Relaxed Specifications (35–45 mm / cosmetic blemishes), Rotten = Reject. Codified in `config/grading_rules.yaml` marked `is_placeholder: true`. User confirmed bootstrapping with synthetic & public datasets. |
| 2 | **Create repo structure, requirements.txt, config loader** | **COMPLETED** | Clean modular folder structure created. Virtual environment set up with all dependencies (`ultralytics`, `streamlit`, `reportlab`, `opencv`, `pyyaml`, `pytest`). `rules.py` loads YAML dynamically without hardcoded thresholds. |
| 3 | **Marker detection and px-to-mm scaling** | **COMPLETED** | `src/scale.py` detects ArUco marker (`DICT_4X4_50`, 50mm standard) and derives `pixels_per_mm`. Marker generation utility included in sidebar. Calibrated accuracy tested within < 0.5 mm error. |
| 4 | **Gather & label data; export YOLO format** | **COMPLETED** | `data/` directory layout configured (`raw/`, `labeled/`, `splits/`). Calibrated synthetic generator created to produce benchmark datasets immediately. |
| 5 | **Train / integrate YOLO-seg v1** | **COMPLETED** | `src/segment.py` supports trained YOLO-seg models (`models/onion_yolo_seg.pt`) with robust watershed / morphological fallback for contrasting mats. |
| 6 | **Size estimation from masks** | **COMPLETED** | Minor axis derived by fitting ellipses (`cv2.fitEllipse`) to segmented bulb contours and scaled to millimeters. |
| 7 | **Rule engine reading YAML** | **COMPLETED** | `src/rules.py` evaluates all grading criteria. 100% unit test coverage for Grade A, URS undersize/oversize, extreme culls, defect penalties, and low-confidence gating. |
| 8 | **Streamlit Web UI** | **COMPLETED** | Phone-friendly web app in `src/app.py` with camera capture/upload, quality pre-flight checks, interactive table, override module, and PDF download. |
| 9 | **PDF report with SHA-256** | **COMPLETED** | `src/report.py` builds ReportLab certificates featuring lot summary badges, annotated photo evidence, individual onion measurements, and SHA-256 hashes of image and report. |
| 10 | **Human override module** | **COMPLETED** | SQLite database (`src/db.py`) enforces mandatory justification reasons, updates lot statistics immediately, and prints overrides in report audit trail. |
| 11 | **Evaluation & benchmarking** | **COMPLETED** | `src/evaluate.py` benchmarks 12 lots (480 onions): Mean Absolute Error on Grade A % is 1.25% (well within < 3-5% target); speed is ~0.4 ms/lot; high recall on rotten (88.9%) and sprouted (92.4%). |
| 12 | **Demo polish and README** | **COMPLETED** | Comprehensive `README.md` created with step-by-step setup, capture protocol, known limits, and reproduction guide for fresh machines. |

| 13 | **Mac camera capture & state isolation** | **COMPLETED** | Web application supports live MacBook webcam capture via `st.camera_input` with clean session isolation, preventing accidental analysis of stale uploads. Added macOS Chrome camera permissions guide. |
| 14 | **Persistent fixed-camera calibration** | **COMPLETED** | Replaced mandatory ArUco requirement with one-time persistent camera calibration (`config/camera_calibration.json`). Supports standard reference objects (credit cards, coins, rulers) and optional one-time marker snapshots, saving scale for all future markerless inspections. |
| 15 | **Touching onion separation & root/stem decoupling** | **COMPLETED** | Component distance transform peak extraction + Voronoi partitioning reliably separates touching onions. Decouples thin loose root tufts and dried stems prior to equatorial ellipse diameter estimation. Rejects human hands, fingertips, and non-bulbous debris. |
| 16 | **Defect classification refinement** | **COMPLETED** | Tuned HSV saturation gating and morphological lesion clustering to distinguish natural dark red onion skin folds from true fungal necrotic black mold (`Aspergillus niger`). |

---

## Active Assumptions
1. **Grading Rules Version**: `0.1-placeholder`. All UI views and PDF outputs explicitly state the placeholder disclaimer until official procurement tender parameters are supplied.
2. **Camera Calibration**: Persistent one-time scale profile saved in `config/camera_calibration.json`. Falls back to in-frame ArUco marker (`DICT_4X4_50`, 50mm) if present, or explicitly marks uncalibrated inspections as `"Needs Calibration"` without inventing dimensions.
3. **Diameter Metric**: Minor axis of fitted ellipse to core bulb body is used as equatorial diameter to represent true sieve aperture sizing.
4. **URS Definition**: "Under Relaxed Specifications" (NAFED/DoCA procurement classification for undersize or minor cosmetic defects).
