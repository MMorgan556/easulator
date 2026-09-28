"""End-to-end analysis: ingest -> detect -> post-process -> report."""

from __future__ import annotations

import hashlib

from .config import Settings
from .detection import Detector
from .ingest import load_image
from .postprocess import build_findings
from .report import ReportGenerationError, ReportGenerator, TemplateReportGenerator
from .schemas import AnalysisResult, ImageModality


class Pipeline:
    def __init__(self, settings: Settings, detector: Detector, reporter: ReportGenerator):
        self.settings = settings
        self.detector = detector
        self.reporter = reporter

    def analyze(self, data: bytes, with_report: bool = True) -> AnalysisResult:
        loaded = load_image(data)
        info = loaded.info
        warnings: list[str] = []

        if self.detector.name == "demo":
            warnings.append("DEMO MODE: findings are synthetic placeholders, not an analysis of this image.")
        if info.modality != ImageModality.PANORAMIC:
            warnings.append(
                f"FDI numbering is only estimated on panoramic images; this looks like a {info.modality.value}."
            )

        detections = self.detector.detect(loaded.pixels)
        teeth, findings = build_findings(
            detections,
            image_width=info.width,
            modality=info.modality,
            min_confidence=self.settings.min_confidence,
            review_confidence=self.settings.review_confidence,
        )

        report = None
        if with_report:
            try:
                report = self.reporter.generate(info, teeth, findings, warnings)
            except ReportGenerationError as exc:
                if self.settings.report_backend == "claude":
                    raise
                warnings.append(f"Claude report unavailable ({exc}); used template report instead.")
                report = TemplateReportGenerator().generate(info, teeth, findings, warnings)

        return AnalysisResult(
            # Content hash, so the same upload always maps to the same ID without storing any patient data.
            study_id=hashlib.sha256(data).hexdigest()[:16],
            image=info,
            detector=self.detector.name,
            teeth=teeth,
            findings=findings,
            report=report,
            warnings=warnings,
        )
