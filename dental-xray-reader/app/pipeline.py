"""End-to-end analysis: ingest -> detect -> post-process -> report.

The work is split into an image stage (``examine``: memory-heavy, needs the upload) and a
report stage (``write_report``: slow, but only needs the small structured findings), so the
API can bound each stage separately and release the image before waiting on Claude.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import io
from dataclasses import dataclass

import numpy as np
from PIL import Image

from .config import Settings
from .detection import Detector
from .ingest import IngestError, load_image
from .postprocess import build_findings
from .report import ReportGenerationError, ReportGenerator, TemplateReportGenerator
from .schemas import AnalysisResult, Finding, ImageInfo, ImageModality, Report, Tooth

PREVIEW_MAX_SIDE = 1600


def encode_preview(pixels: np.ndarray, max_side: int = PREVIEW_MAX_SIDE) -> str:
    """Downscaled PNG of the grayscale image, base64-encoded for JSON transport."""
    img = Image.fromarray(pixels)
    img.thumbnail((max_side, max_side), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return base64.b64encode(buf.getvalue()).decode("ascii")


@dataclass
class Examination:
    """Everything learned from the image; holds no pixel data."""

    study_id: str
    info: ImageInfo
    teeth: list[Tooth]
    findings: list[Finding]
    warnings: list[str]
    preview_png: str | None


class Pipeline:
    def __init__(self, settings: Settings, detector: Detector, reporter: ReportGenerator):
        self.settings = settings
        self.detector = detector
        self.reporter = reporter
        # Admission limits for the API. Waiting on these holds no thread and no upload body.
        self.image_slots = asyncio.Semaphore(max(1, settings.max_concurrent_images))
        self.report_slots = asyncio.Semaphore(max(1, settings.max_concurrent_reports))

    def examine(self, data: bytes, with_preview: bool = False) -> Examination:
        # Content hash, so the same upload always maps to the same ID without storing any patient data.
        study_id = hashlib.sha256(data).hexdigest()[:16]
        loaded = load_image(data)
        detections = self.detector.detect(loaded.pixels)
        preview = encode_preview(loaded.pixels) if with_preview else None
        info = loaded.info
        del loaded  # the full-size image is not needed past this point

        warnings: list[str] = []
        if self.detector.name == "demo":
            warnings.append("DEMO MODE: findings are synthetic placeholders, not an analysis of this image.")
        if info.modality != ImageModality.PANORAMIC:
            warnings.append(
                f"FDI numbering is only estimated on panoramic images; this looks like a {info.modality.value}."
            )

        teeth, findings = build_findings(
            detections,
            image_width=info.width,
            image_height=info.height,
            modality=info.modality,
            min_confidence=self.settings.min_confidence,
            review_confidence=self.settings.review_confidence,
        )
        return Examination(study_id, info, teeth, findings, warnings, preview)

    def examine_or_reason(self, data: bytes, with_preview: bool = False) -> Examination | str:
        """``examine``, returning why the upload was rejected instead of raising.

        For callers on another thread: an exception carried back across the thread boundary
        keeps its traceback, and with it every frame's locals (the upload and the decoded
        arrays), alive for as long as the worker or future holds on to it.
        """
        try:
            return self.examine(data, with_preview=with_preview)
        except IngestError as exc:
            reason = str(exc)
        return reason  # outside the except block, so the traceback has already been released

    def write_report(self, exam: Examination) -> Report:
        """Report from the configured generator; in auto mode, a template if Claude fails."""
        try:
            return self.reporter.generate(exam.info, exam.teeth, exam.findings, exam.warnings)
        except ReportGenerationError as exc:
            if self.settings.report_backend == "claude":
                raise
            exam.warnings.append(f"Claude report unavailable ({exc}); used template report instead.")
            return TemplateReportGenerator().generate(exam.info, exam.teeth, exam.findings, exam.warnings)

    def result(self, exam: Examination, report: Report | None) -> AnalysisResult:
        return AnalysisResult(
            study_id=exam.study_id,
            image=exam.info,
            detector=self.detector.name,
            teeth=exam.teeth,
            findings=exam.findings,
            report=report,
            warnings=exam.warnings,
            preview_png=exam.preview_png,
        )

    def analyze(self, data: bytes, with_report: bool = True, with_preview: bool = False) -> AnalysisResult:
        """Both stages in one call, for scripts and tests."""
        exam = self.examine(data, with_preview=with_preview)
        return self.result(exam, self.write_report(exam) if with_report else None)
