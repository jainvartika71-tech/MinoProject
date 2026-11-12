# 🧅 Onion Grading Automation (Lean MVP)

An automated, phone-friendly computer vision tool that inspects photos of onion samples and returns:
- **Bulb Count**: Total onions detected
- **Metric Sizing**: Calibrated minor-axis diameter (mm) per bulb via ArUco reference scale
- **Defect Classification**: Healthy, Rotten, Sprouted, Damaged, Other
- **Procurement Grading**: Dynamic YAML rule engine assigning **Grade A %**, **URS %**, and **Reject %**
- **Tamper-Evident PDF Certificate**: Downloadable inspection report with annotated photo, per-onion breakdown, human override log, and cryptographic SHA-256 checksums

---

## 1. System Architecture & Pipeline

```
[Phone Photo / Upload]
          │
          ▼
┌──────────────────────────┐
│  src/capture_check.py    │ ── Reject if blurry, underexposed, or missing ArUco marker
└──────────────────────────┘
          │ (Pass)
          ▼
┌──────────────────────────┐
│  src/scale.py            │ ── Computes pixels-per-mm ratio from 50mm ArUco marker
└──────────────────────────┘
          │
          ▼
┌──────────────────────────┐
│  src/segment.py          │ ── Segments bulbs, fits ellipses, measures minor axis (mm)
└──────────────────────────┘
          │
          ▼
┌──────────────────────────┐
│  src/defects.py          │ ── Classifies surface defects on bulb foreground
└──────────────────────────┘
          │
          ▼
┌──────────────────────────┐
│  src/rules.py            │ ── Evaluates config/grading_rules.yaml (Grade A, URS, Reject)
└──────────────────────────┘
          │
          ▼
┌──────────────────────────┐
│  src/app.py & report.py  │ ── Streamlit UI + Human Override + ReportLab PDF with SHA-256
└──────────────────────────┘
```

---

## 2. Grading Rules & Procurement Standards

> [!IMPORTANT]
> **Zero Hardcoded Thresholds**:
> All sizing ranges, defect penalty mappings, and confidence gates live strictly in `config/grading_rules.yaml`. Changing procurement criteria **never** requires retraining or code edits.

- **Standard Reference Standard**: Derived from operational guidelines by the Indian Ministry of Consumer Affairs (DoCA), NAFED, and NCCF:
  - **Grade A**: Bulb diameter **45 mm to 65 mm**, free from decay, rot, and active sprouts.
  - **URS (Under Relaxed Specifications)**: Accommodates smaller or non-standard bulbs (**35 mm to 45 mm** or **65 mm to 75 mm**) and minor cosmetic surface defects (superficial cuts, peeling, minor sunburn, mild sprout).
  - **Reject**: Rotten/decayed bulbs or extreme unmarketable culls (< 35 mm or > 75 mm).
- **Placeholder Transparency**: Marked with `is_placeholder: true` and `rules_version: 0.1-placeholder`. All UI views and generated PDF reports prominently display placeholder notices until officially ratified by procurement tender.

---

## 3. Strict Capture Protocol

For reliable computer vision measurement:
1. **Plain, Contrasting Mat**: Spread onions across a dark or uniform mat (e.g., navy blue, black, or gray mat).
2. **Single Layer, Non-Touching**: Bulbs should not be stacked or touching where possible.
3. **Reference Scale Marker**: Place a printed **50 mm x 50 mm ArUco marker** (`DICT_4X4_50`, ID `0`) in the frame. The web app includes a built-in tool to download and print this marker.
4. **Overhead Geometry**: Position the camera directly overhead (~0.8 to 1.2 meters above the mat).
5. **Uniform Lighting**: Use diffuse ambient or overhead lighting. Avoid single-point flash glare.

---

## 4. Quick Start (Fresh Machine)

### Prerequisites
- Python 3.9+ (Python 3.10 or 3.11 recommended)
- `pip`

### Step 1: Clone and Set Up Virtual Environment
```bash
git clone <repo-url> onion-grader
cd onion-grader

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 3: Run the Automated Test Suite
Run the 29 unit and end-to-end integration tests:
```bash
PYTHONPATH=. pytest tests/ -v
```

### Step 4: Run the Benchmark Evaluator
Run the AI vs human ground-truth benchmark across 12 simulated lots (480 onions):
```bash
PYTHONPATH=. python3 src/evaluate.py
```

### Step 5: Launch the Streamlit Web Application
```bash
streamlit run src/app.py
```
Open `http://localhost:8501` in your browser (or scan QR / open IP on your phone on the local Wi-Fi).

---

## 5. Web App Features

1. **Pre-Flight Input Gating**:
   - Rejects blurry photos (`Laplacian variance < 80.0`).
   - Rejects underexposed or overexposed photos.
   - Detects the ArUco marker and displays calibrated scale (`px/mm`).
2. **Instant Synthetic Lot Generator**:
   - In the **Synthetic Demo Generator** tab, generate a calibrated synthetic onion sample with embedded ArUco scale, varied bulb diameters (Grade A, URS), and simulated defects (rotten, sprouted).
3. **Interactive Measurement Overlay**:
   - Visualizes segmented contours, fitted ellipses, IDs, and diameters in millimeters.
4. **Auditor Human Override**:
   - Select any Onion ID, modify its grade, and provide a **mandatory justification reason** (e.g., "Concealed basal rot detected").
   - Immediately recalculates lot Grade A % and URS %, logged to SQLite database and printed on the audit section of the PDF.
5. **Tamper-Evident PDF Certificate**:
   - One-click generation of an official ReportLab certificate containing:
     - Header, timestamp, inspector name, lot ID
     - Visual annotated photo evidence
     - Metric summary card
     - Per-onion table
     - Human override audit trail
     - **Source photo SHA-256** and **PDF certificate SHA-256** cryptographic hashes.

---

## 6. Known Limitations

As specified in the procurement guidelines:
1. **Surface Defects Only**: Optical cameras can only evaluate visible surface flaws. Internal rot, black heart, or concealed center decay cannot be diagnosed optically without destructive slicing.
2. **Sampling Bias**: The accuracy of a lot grade depends on random sampling. Operators should collect 2 to 3 photos per lot, reshuffling between captures.
3. **Environmental Sensitivity**: Extreme lighting glare, shadows, or an uncalibrated scale marker can affect millimeter precision. Always adhere to the Capture Protocol.
4. **Varietal Differences**: Transitioning between red, yellow, and white onion varieties may require re-checking color thresholds or fine-tuning YOLO-seg on local harvests.

---

## 7. Folder Structure

```
onion-grader/
├── config/
│   └── grading_rules.yaml      # Sizing, defect mappings, and thresholds
├── data/                       # Local SQLite DB & dataset storage
│   ├── raw/
│   ├── labeled/
│   └── splits/
├── models/                     # Saved YOLO-seg weights (.pt)
├── reports/                    # Generated inspection certificates (PDF)
├── src/
│   ├── __init__.py
│   ├── app.py                  # Streamlit web interface
│   ├── capture_check.py        # Sharpness, exposure & ArUco presence
│   ├── db.py                   # SQLite persistence & override audit trail
│   ├── defects.py              # Foreground defect classifier
│   ├── evaluate.py             # Benchmark evaluation suite
│   ├── report.py               # ReportLab PDF generator with SHA-256
│   ├── rules.py                # Deterministic YAML rule engine
│   ├── scale.py                # ArUco marker detection & scale
│   └── segment.py              # YOLO / watershed ellipse dimensioning
├── tests/
│   ├── test_capture_check.py
│   ├── test_e2e_pipeline.py
│   ├── test_report.py
│   ├── test_rules.py
│   ├── test_scale.py
│   └── test_segment.py
├── PROGRESS.md                 # Living progress tracker and AI handoff notes
├── requirements.txt            # Dependency specifications
└── README.md                   # System documentation
```
