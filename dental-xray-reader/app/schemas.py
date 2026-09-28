"""Structured data passed between pipeline stages and returned by the API."""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class ImageModality(str, Enum):
    BITEWING = "bitewing"
    PERIAPICAL = "periapical"
    PANORAMIC = "panoramic"
    UNKNOWN = "unknown"


class FindingType(str, Enum):
    TOOTH = "tooth"
    CARIES = "caries"
    PERIAPICAL_LESION = "periapical_lesion"
    RESTORATION = "restoration"
    CROWN = "crown"
    ROOT_CANAL_TREATMENT = "root_canal_treatment"
    IMPLANT = "implant"
    IMPACTED_TOOTH = "impacted_tooth"
    BONE_LOSS = "bone_loss"
    CALCULUS = "calculus"


# Findings that indicate disease, as opposed to existing dental work or anatomy.
PATHOLOGY_TYPES = {
    FindingType.CARIES,
    FindingType.PERIAPICAL_LESION,
    FindingType.IMPACTED_TOOTH,
    FindingType.BONE_LOSS,
    FindingType.CALCULUS,
}


class BoundingBox(BaseModel):
    """Pixel coordinates in the original image (x1, y1 top-left; x2, y2 bottom-right)."""

    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def area(self) -> float:
        return max(0.0, self.x2 - self.x1) * max(0.0, self.y2 - self.y1)

    @property
    def center(self) -> tuple[float, float]:
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)

    def intersection(self, other: BoundingBox) -> float:
        w = min(self.x2, other.x2) - max(self.x1, other.x1)
        h = min(self.y2, other.y2) - max(self.y1, other.y1)
        return max(0.0, w) * max(0.0, h)

    def iou(self, other: BoundingBox) -> float:
        inter = self.intersection(other)
        union = self.area + other.area - inter
        return inter / union if union > 0 else 0.0


class Detection(BaseModel):
    """Raw detector output before post-processing."""

    type: FindingType
    confidence: float = Field(ge=0.0, le=1.0)
    box: BoundingBox


class Finding(BaseModel):
    id: str
    type: FindingType
    confidence: float = Field(ge=0.0, le=1.0)
    box: BoundingBox
    tooth: str | None = Field(default=None, description="FDI tooth number the finding sits on, if known")
    needs_review: bool = False


class Tooth(BaseModel):
    fdi: str
    confidence: float
    box: BoundingBox


class ImageInfo(BaseModel):
    width: int
    height: int
    modality: ImageModality
    source_format: str
    pixel_spacing_mm: tuple[float, float] | None = None


class Report(BaseModel):
    text: str
    generator: str
    disclaimer: str = (
        "AI-generated draft for clinical decision support only. "
        "All findings must be reviewed and confirmed by a licensed dentist."
    )


class AnalysisResult(BaseModel):
    study_id: str
    image: ImageInfo
    detector: str
    teeth: list[Tooth]
    findings: list[Finding]
    report: Report | None = None
    warnings: list[str] = Field(default_factory=list)
    preview_png: str | None = Field(
        default=None,
        description="Base64 PNG of the processed image, downscaled; box coordinates still refer to the full image",
    )
