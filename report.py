"""Tamper-Evident PDF Inspection Report Generator.

Generates official quality certificates using ReportLab, complete with:
- Lot statistics (Grade A %, URS %, Reject %, Needs Review %)
- Embedded annotated inspection image
- Human override audit trail
- Granular per-onion measurement breakdown table
- Cryptographic SHA-256 verification hashes for image and report file
"""

import hashlib
import io
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime

import cv2
import numpy as np
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    Image as RLImage,
    KeepTogether,
    HRFlowable,
)

DEFAULT_REPORTS_DIR = Path(__file__).resolve().parent.parent / "reports"


def calculate_sha256(data_or_path: Any) -> str:
    """Computes SHA-256 hex digest of bytes, numpy array, or file path."""
    hasher = hashlib.sha256()
    if isinstance(data_or_path, (str, Path)):
        p = Path(data_or_path)
        if p.exists():
            with open(p, "rb") as f:
                while chunk := f.read(65536):
                    hasher.update(chunk)
            return hasher.hexdigest()
        data = str(data_or_path).encode("utf-8")
        hasher.update(data)
    elif isinstance(data_or_path, np.ndarray):
        hasher.update(data_or_path.tobytes())
    elif isinstance(data_or_path, bytes):
        hasher.update(data_or_path)
    else:
        hasher.update(str(data_or_path).encode("utf-8"))
    return hasher.hexdigest()


class InspectionReportGenerator:
    """Builds structured PDF inspection reports."""

    def __init__(self, output_dir: Optional[Path] = None):
        self.output_dir = Path(output_dir) if output_dir else DEFAULT_REPORTS_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.styles = getSampleStyleSheet()
        self._init_custom_styles()

    def _init_custom_styles(self) -> None:
        self.title_style = ParagraphStyle(
            "ReportTitle",
            parent=self.styles["Heading1"],
            fontSize=18,
            leading=22,
            textColor=colors.HexColor("#1A365D"),
            spaceAfter=4,
        )
        self.subtitle_style = ParagraphStyle(
            "ReportSubtitle",
            parent=self.styles["Normal"],
            fontSize=10,
            leading=13,
            textColor=colors.HexColor("#4A5568"),
            spaceAfter=10,
        )
        self.alert_style = ParagraphStyle(
            "AlertBanner",
            parent=self.styles["Normal"],
            fontSize=9,
            leading=12,
            textColor=colors.HexColor("#742A2A"),
            backColor=colors.HexColor("#FFF5F5"),
            borderColor=colors.HexColor("#FEB2B2"),
            borderWidth=1,
            borderPadding=6,
            spaceAfter=12,
        )
        self.cell_style = ParagraphStyle(
            "ReportCell",
            parent=self.styles["Normal"],
            fontSize=8,
            leading=10,
        )
        self.header_cell_style = ParagraphStyle(
            "ReportHeaderCell",
            parent=self.styles["Normal"],
            fontSize=8,
            leading=10,
            textColor=colors.whitesmoke,
            fontName="Helvetica-Bold",
        )

    def generate_pdf(
        self,
        lot_id: str,
        lot_summary: Dict[str, Any],
        onions: List[Dict[str, Any]],
        annotated_image: Optional[np.ndarray] = None,
        image_sha256: str = "",
        overrides: Optional[List[Dict[str, Any]]] = None,
        operator_name: str = "Automated System / Auditor",
    ) -> Tuple[Path, str]:
        """Generates a PDF report and returns (file_path, report_sha256)."""
        timestamp_str = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        filename = f"Inspection_Report_{lot_id}_{timestamp_str}.pdf"
        report_path = self.output_dir / filename

        doc = SimpleDocTemplate(
            str(report_path),
            pagesize=letter,
            rightMargin=36,
            leftMargin=36,
            topMargin=36,
            bottomMargin=36,
        )

        story: List[Any] = []

        # 1. Header & Title
        story.append(Paragraph("Onion Lot Quality & Grading Certificate", self.title_style))
        now_formatted = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
        story.append(
            Paragraph(
                f"Lot ID: <b>{lot_id}</b> &nbsp;|&nbsp; Inspected: {now_formatted} &nbsp;|&nbsp; Inspector: {operator_name}",
                self.subtitle_style,
            )
        )

        # 2. Placeholder Disclaimer Banner
        if lot_summary.get("is_placeholder", True):
            disclaimer_text = (
                "⚠️ <b>NOTICE - PLACEHOLDER SPECIFICATION:</b> "
                f"Grading rules version <i>{lot_summary.get('rules_version', '0.1')}</i> uses benchmark "
                "parameters (Grade A: 45-65mm, URS: 35-45mm). Confirm against official procurement tender."
            )
            story.append(Paragraph(disclaimer_text, self.alert_style))

        # 3. Lot Summary Metric Cards
        tot = lot_summary.get("total_count", 0)
        ga_pct = lot_summary.get("grade_a_pct", 0.0)
        urs_pct = lot_summary.get("urs_pct", 0.0)
        rej_pct = lot_summary.get("reject_pct", 0.0)
        rev_pct = lot_summary.get("needs_review_pct", 0.0)
        is_cal = lot_summary.get("is_calibrated", True)
        avg_diam = lot_summary.get("avg_diameter_mm")

        if is_cal and avg_diam is not None:
            diam_cell_text = f"<font size=12><b>{float(avg_diam):.1f} mm</b></font>"
        else:
            diam_cell_text = "<font size=10 color='#742A2A'><b>Calibration unavailable</b></font>"

        if not is_cal:
            cal_alert = (
                "⚠️ <b>SCALE NOTICE:</b> ArUco scale marker (DICT_4X4_50, ID 0) was not detected. "
                "Metric dimensions (mm) are uncalibrated and size-based grading is held pending calibration."
            )
            story.append(Paragraph(cal_alert, self.alert_style))

        summary_data = [
            [
                Paragraph("<b>Total Sampled</b>", self.cell_style),
                Paragraph("<b>Grade A</b>", self.cell_style),
                Paragraph("<b>URS (Relaxed)</b>", self.cell_style),
                Paragraph("<b>Reject</b>", self.cell_style),
                Paragraph("<b>Needs Review</b>", self.cell_style),
                Paragraph("<b>Mean Diameter</b>", self.cell_style),
            ],
            [
                Paragraph(f"<font size=12><b>{tot}</b></font>", self.cell_style),
                Paragraph(f"<font size=12 color='#22543D'><b>{ga_pct:.1f}%</b></font><br/>({lot_summary.get('grade_a_count', 0)})", self.cell_style),
                Paragraph(f"<font size=12 color='#744210'><b>{urs_pct:.1f}%</b></font><br/>({lot_summary.get('urs_count', 0)})", self.cell_style),
                Paragraph(f"<font size=12 color='#742A2A'><b>{rej_pct:.1f}%</b></font><br/>({lot_summary.get('reject_count', 0)})", self.cell_style),
                Paragraph(f"<font size=12 color='#2C5282'><b>{rev_pct:.1f}%</b></font><br/>({lot_summary.get('needs_review_count', 0)})", self.cell_style),
                Paragraph(diam_cell_text, self.cell_style),
            ],
        ]

        summary_table = Table(summary_data, colWidths=[90, 90, 90, 90, 90, 90])
        summary_table.setStyle(
            TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#EDF2F7")),
                ("BACKGROUND", (0, 1), (-1, 1), colors.HexColor("#F7FAFC")),
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E0")),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ])
        )
        story.append(summary_table)
        story.append(Spacer(1, 10))

        # 4. Embedded Annotated Image (if available)
        if annotated_image is not None and annotated_image.size > 0:
            # Save annotated image temporarily as PNG bytes
            temp_img_path = self.output_dir / f"temp_{lot_id}.png"
            # Resize for PDF display (fit within 540 width, 260 height max)
            h, w = annotated_image.shape[:2]
            aspect = h / w
            display_w = 480
            display_h = int(display_w * aspect)
            if display_h > 240:
                display_h = 240
                display_w = int(display_h / aspect)

            cv2.imwrite(str(temp_img_path), annotated_image)
            story.append(KeepTogether([
                Paragraph("<b>Visual Evidence (Computer Vision Segmentation & Calibration Overlay)</b>", self.subtitle_style),
                RLImage(str(temp_img_path), width=display_w, height=display_h),
                Spacer(1, 10),
            ]))

        # 5. Human Overrides Audit Log (if any)
        if overrides and len(overrides) > 0:
            story.append(Paragraph("<b>Audit Trail: Human Expert Overrides</b>", self.subtitle_style))
            override_data = [[
                Paragraph("Onion #", self.header_cell_style),
                Paragraph("Original", self.header_cell_style),
                Paragraph("Overridden", self.header_cell_style),
                Paragraph("Mandatory Reason", self.header_cell_style),
                Paragraph("Operator", self.header_cell_style),
                Paragraph("Timestamp", self.header_cell_style),
            ]]
            for o in overrides:
                override_data.append([
                    Paragraph(str(o.get("onion_id")), self.cell_style),
                    Paragraph(str(o.get("old_grade")), self.cell_style),
                    Paragraph(f"<b>{o.get('new_grade')}</b>", self.cell_style),
                    Paragraph(str(o.get("reason")), self.cell_style),
                    Paragraph(str(o.get("operator", "Auditor")), self.cell_style),
                    Paragraph(str(o.get("timestamp", ""))[:19], self.cell_style),
                ])
            override_table = Table(override_data, colWidths=[45, 65, 75, 175, 80, 100])
            override_table.setStyle(
                TableStyle([
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2C5282")),
                    ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E0")),
                    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.whitesmoke, colors.HexColor("#FFFBEB")]),
                    ("TOPPADDING", (0, 0), (-1, -1), 4),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ])
            )
            story.append(override_table)
            story.append(Spacer(1, 10))

        # 6. Detailed Per-Onion Breakdown (first 25 in table to keep PDF compact)
        story.append(Paragraph(f"<b>Individual Onion Measurement Records (Sample {min(len(onions), 25)} of {len(onions)})</b>", self.subtitle_style))
        records_data = [[
            Paragraph("ID", self.header_cell_style),
            Paragraph("Minor Axis", self.header_cell_style),
            Paragraph("Defect Label", self.header_cell_style),
            Paragraph("Confidence", self.header_cell_style),
            Paragraph("Assigned Grade", self.header_cell_style),
            Paragraph("Grading Reason", self.header_cell_style),
        ]]
        for o in onions[:25]:
            grade_color = "#22543D" if o.get("final_grade") == "Grade A" else ("#744210" if o.get("final_grade") == "URS" else "#742A2A")
            diam_raw = o.get("diameter_mm")
            if diam_raw is not None and str(diam_raw).strip() != "" and float(diam_raw) > 0:
                diam_cell = f"{float(diam_raw):.1f} mm"
            else:
                diam_cell = "Calibration unavailable"

            records_data.append([
                Paragraph(str(o.get("onion_id")), self.cell_style),
                Paragraph(diam_cell, self.cell_style),
                Paragraph(str(o.get("defect_label")), self.cell_style),
                Paragraph(f"{float(o.get('confidence', 0.0)):.2f}", self.cell_style),
                Paragraph(f"<font color='{grade_color}'><b>{o.get('final_grade')}</b></font>", self.cell_style),
                Paragraph(str(o.get("initial_reason", "")), self.cell_style),
            ])

        records_table = Table(records_data, colWidths=[30, 65, 75, 55, 80, 235])
        records_table.setStyle(
            TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1A365D")),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E0")),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F7FAFC")]),
                ("TOPPADDING", (0, 0), (-1, -1), 3),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ])
        )
        story.append(records_table)
        story.append(Spacer(1, 12))

        # 7. Verification Seal & Tamper-Evidence Hashes
        story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#CBD5E0"), spaceAfter=6))
        hash_text = (
            f"<b>Tamper-Evidence Cryptographic Verification:</b><br/>"
            f"• Source Photo SHA-256: <code>{image_sha256 or 'N/A'}</code><br/>"
            f"• Model Version: <code>yolo-seg-v1 (ultralytics)</code> &nbsp;|&nbsp; "
            f"Rules Version: <code>{lot_summary.get('rules_version', '0.1-placeholder')}</code><br/>"
            "<i>Notice: Optical inspection detects surface defects. Internal decay requires destructive cross-sectioning.</i>"
        )
        story.append(Paragraph(hash_text, self.cell_style))

        # Build document
        doc.build(story)

        # Cleanup temp image if created
        temp_img = self.output_dir / f"temp_{lot_id}.png"
        if temp_img.exists():
            try:
                temp_img.unlink()
            except Exception:
                pass

        # Calculate final PDF SHA-256
        report_sha256 = calculate_sha256(report_path)

        return report_path, report_sha256
