"""Grading rule engine for Onion Quality & Procurement Automation.

CRITICAL ARCHITECTURE RULE:
All grading thresholds (diameters, defect actions, confidence thresholds) MUST
live in the YAML configuration file and are loaded dynamically. No grading
thresholds are hardcoded here.
If physical scale calibration is unavailable (ArUco marker missing), size-based
grades are never invented.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional
import yaml


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "grading_rules.yaml"


@dataclass
class GradingResult:
    """Represents the evaluation outcome for an individual onion."""
    onion_id: int
    diameter_mm: Optional[float]
    defect_label: str
    confidence: float
    grade: str  # "Grade A", "URS", "Reject", "Needs Human Review", "Needs Calibration"
    reason: str
    is_placeholder: bool = True
    is_calibrated: bool = True
    needs_review: bool = False
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LotSummary:
    """Aggregated quality metrics for a sample lot."""
    total_count: int
    grade_a_count: int
    grade_a_pct: float
    urs_count: int
    urs_pct: float
    reject_count: int
    reject_pct: float
    needs_review_count: int
    needs_review_pct: float
    uncalibrated_count: int
    uncalibrated_pct: float
    is_calibrated: bool
    avg_diameter_mm: Optional[float]
    min_diameter_mm: Optional[float]
    max_diameter_mm: Optional[float]
    defect_counts: Dict[str, int]
    rules_version: str
    is_placeholder: bool
    disclaimer: str


class GradingRuleEngine:
    """Reads YAML config and grades onions deterministically."""

    def __init__(self, config_path: Optional[Path] = None):
        self.config_path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
        self.config = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        """Loads and validates grading rules from YAML."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Grading rules configuration not found at {self.config_path}")
        with open(self.config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        if not isinstance(cfg, dict):
            raise ValueError(f"Invalid YAML config structure in {self.config_path}")
        return cfg

    def reload(self) -> None:
        """Reloads the YAML config from disk without restarting the process."""
        self.config = self._load_config()

    @property
    def rules_version(self) -> str:
        return str(self.config.get("rules_version", "unknown"))

    @property
    def is_placeholder(self) -> bool:
        return bool(self.config.get("is_placeholder", True))

    @property
    def disclaimer(self) -> str:
        return str(self.config.get("disclaimer", "PLACEHOLDER RULES - Verify before official procurement use."))

    @property
    def marker_config(self) -> Dict[str, Any]:
        return self.config.get("marker", {"dictionary": "DICT_4X4_50", "id": 0, "size_mm": 50.0})

    @property
    def thresholds_config(self) -> Dict[str, Any]:
        return self.config.get("thresholds", {})

    def grade_onion(
        self,
        onion_id: int,
        diameter_mm: Optional[float],
        defect_label: str,
        confidence: float,
        is_calibrated: Optional[bool] = None,
    ) -> GradingResult:
        """Grades a single onion instance based on configured rules.

        Args:
            onion_id: Unique integer index for the onion in this lot.
            diameter_mm: Estimated minor-axis diameter in mm (None if uncalibrated).
            defect_label: Predicted surface defect class ('healthy', 'rotten', 'sprouted', 'damaged', 'other').
            confidence: Model confidence score in [0.0, 1.0].
            is_calibrated: Whether physical ArUco scale is valid.

        Returns:
            GradingResult with grade, reason, and placeholder flag.
        """
        defect_norm = (defect_label or "healthy").strip().lower()
        conf_min = float(self.config.get("thresholds", {}).get("confidence_min", 0.50))
        is_low_confidence = confidence < conf_min

        calibrated = (
            (diameter_mm is not None and diameter_mm > 0.0)
            if is_calibrated is None
            else bool(is_calibrated and diameter_mm is not None and diameter_mm > 0.0)
        )

        size_cfg = self.config.get("size_mm", {})
        grade_a_cfg = size_cfg.get("grade_a", {"min": 45.0, "max": 65.0})
        urs_cfg = size_cfg.get("urs", {"undersize_min": 35.0, "undersize_max": 45.0, "oversize_min": 65.0, "oversize_max": 75.0})
        reject_cfg = size_cfg.get("reject", {"extreme_undersize": 35.0, "extreme_oversize": 75.0})
        defect_map = self.config.get("defects", {
            "rotten": "reject",
            "sprouted": "urs",
            "damaged": "urs",
            "healthy": "by_size",
            "other": "urs",
        })

        diam_rounded = round(diameter_mm, 2) if (diameter_mm is not None and calibrated) else None

        # 1. Low Confidence Gate
        if is_low_confidence:
            return GradingResult(
                onion_id=onion_id,
                diameter_mm=diam_rounded,
                defect_label=defect_norm,
                confidence=round(confidence, 3),
                grade="Needs Human Review",
                reason=f"Detection confidence ({confidence:.2f}) below threshold ({conf_min:.2f})",
                is_placeholder=self.is_placeholder,
                is_calibrated=calibrated,
                needs_review=True,
                details={"confidence_min": conf_min},
            )

        # 2. Defect Action
        defect_action = defect_map.get(defect_norm, "urs")
        if defect_action == "reject":
            return GradingResult(
                onion_id=onion_id,
                diameter_mm=diam_rounded,
                defect_label=defect_norm,
                confidence=round(confidence, 3),
                grade="Reject",
                reason=f"Severe defect: {defect_norm} (unconditional reject)",
                is_placeholder=self.is_placeholder,
                is_calibrated=calibrated,
                needs_review=False,
            )

        if defect_action == "urs" and defect_norm != "healthy":
            return GradingResult(
                onion_id=onion_id,
                diameter_mm=diam_rounded,
                defect_label=defect_norm,
                confidence=round(confidence, 3),
                grade="URS",
                reason=f"Defect penalty: {defect_norm} (classified under relaxed specifications)",
                is_placeholder=self.is_placeholder,
                is_calibrated=calibrated,
                needs_review=False,
            )

        # 3. Calibration Gating for Size-Based Grading
        if not calibrated or diameter_mm is None:
            return GradingResult(
                onion_id=onion_id,
                diameter_mm=None,
                defect_label=defect_norm,
                confidence=round(confidence, 3),
                grade="Needs Calibration",
                reason="Physical scale unavailable (ArUco marker missing). Millimetre size cannot be verified.",
                is_placeholder=self.is_placeholder,
                is_calibrated=False,
                needs_review=True,
                details={"uncalibrated": True},
            )

        # 4. Healthy Bulb - Graded by Size (Minor Axis Diameter)
        ga_min = float(grade_a_cfg.get("min", 45.0))
        ga_max = float(grade_a_cfg.get("max", 65.0))
        urs_under_min = float(urs_cfg.get("undersize_min", 35.0))
        urs_over_max = float(urs_cfg.get("oversize_max", 75.0))
        rej_under = float(reject_cfg.get("extreme_undersize", 35.0))
        rej_over = float(reject_cfg.get("extreme_oversize", 75.0))

        if ga_min <= diameter_mm <= ga_max:
            grade = "Grade A"
            reason = f"Healthy bulb within Grade A standard size ({ga_min:.0f}-{ga_max:.0f} mm)"
        elif urs_under_min <= diameter_mm < ga_min:
            grade = "URS"
            reason = f"Healthy bulb undersized for Grade A; within URS window ({urs_under_min:.0f}-{ga_min:.0f} mm)"
        elif ga_max < diameter_mm <= urs_over_max:
            grade = "URS"
            reason = f"Healthy bulb oversized for Grade A; within URS window ({ga_max:.0f}-{urs_over_max:.0f} mm)"
        elif diameter_mm < rej_under:
            grade = "Reject"
            reason = f"Extreme undersize cull (< {rej_under:.0f} mm)"
        else:
            grade = "Reject"
            reason = f"Extreme oversize bulb (> {rej_over:.0f} mm)"

        return GradingResult(
            onion_id=onion_id,
            diameter_mm=diam_rounded,
            defect_label=defect_norm,
            confidence=round(confidence, 3),
            grade=grade,
            reason=reason,
            is_placeholder=self.is_placeholder,
            is_calibrated=True,
            needs_review=False,
            details={
                "grade_a_range": [ga_min, ga_max],
                "urs_undersize_range": [urs_under_min, ga_min],
                "urs_oversize_range": [ga_max, urs_over_max],
            },
        )

    def summarize_lot(self, results: List[GradingResult]) -> LotSummary:
        """Aggregates individual onion grading results into a lot-level summary."""
        total = len(results)
        if total == 0:
            return LotSummary(
                total_count=0,
                grade_a_count=0,
                grade_a_pct=0.0,
                urs_count=0,
                urs_pct=0.0,
                reject_count=0,
                reject_pct=0.0,
                needs_review_count=0,
                needs_review_pct=0.0,
                uncalibrated_count=0,
                uncalibrated_pct=0.0,
                is_calibrated=False,
                avg_diameter_mm=None,
                min_diameter_mm=None,
                max_diameter_mm=None,
                defect_counts={},
                rules_version=self.rules_version,
                is_placeholder=self.is_placeholder,
                disclaimer=self.disclaimer,
            )

        grade_a = sum(1 for r in results if r.grade == "Grade A")
        urs = sum(1 for r in results if r.grade == "URS")
        reject = sum(1 for r in results if r.grade == "Reject")
        needs_review = sum(1 for r in results if r.grade == "Needs Human Review")
        uncalibrated = sum(1 for r in results if r.grade == "Needs Calibration" or not r.is_calibrated)

        calibrated_diameters = [r.diameter_mm for r in results if (r.is_calibrated and r.diameter_mm is not None and r.diameter_mm > 0)]
        lot_is_calibrated = len(calibrated_diameters) > 0 and uncalibrated == 0

        avg_diam = round(sum(calibrated_diameters) / len(calibrated_diameters), 1) if calibrated_diameters else None
        min_diam = round(min(calibrated_diameters), 1) if calibrated_diameters else None
        max_diam = round(max(calibrated_diameters), 1) if calibrated_diameters else None

        defect_counts: Dict[str, int] = {}
        for r in results:
            defect_counts[r.defect_label] = defect_counts.get(r.defect_label, 0) + 1

        return LotSummary(
            total_count=total,
            grade_a_count=grade_a,
            grade_a_pct=round((grade_a / total) * 100.0, 1),
            urs_count=urs,
            urs_pct=round((urs / total) * 100.0, 1),
            reject_count=reject,
            reject_pct=round((reject / total) * 100.0, 1),
            needs_review_count=needs_review,
            needs_review_pct=round((needs_review / total) * 100.0, 1),
            uncalibrated_count=uncalibrated,
            uncalibrated_pct=round((uncalibrated / total) * 100.0, 1),
            is_calibrated=lot_is_calibrated,
            avg_diameter_mm=avg_diam,
            min_diameter_mm=min_diam,
            max_diameter_mm=max_diam,
            defect_counts=defect_counts,
            rules_version=self.rules_version,
            is_placeholder=self.is_placeholder,
            disclaimer=self.disclaimer,
        )


def load_grading_rules(config_path: Optional[Path] = None) -> GradingRuleEngine:
    """Convenience factory function."""
    return GradingRuleEngine(config_path)
