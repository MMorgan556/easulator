"""Runtime settings, read from environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass, field


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    return float(value) if value else default


@dataclass(frozen=True)
class Settings:
    # "demo" runs without model weights; "yolo" loads an Ultralytics model from YOLO_WEIGHTS.
    detector_backend: str = field(default_factory=lambda: os.getenv("DETECTOR_BACKEND", "demo"))
    yolo_weights: str = field(default_factory=lambda: os.getenv("YOLO_WEIGHTS", "weights/dental-yolo.pt"))
    # "auto" uses Claude when credentials are configured, "claude" requires it, "template" never calls it.
    report_backend: str = field(default_factory=lambda: os.getenv("REPORT_BACKEND", "auto"))
    claude_model: str = field(default_factory=lambda: os.getenv("CLAUDE_MODEL", "claude-opus-5"))
    # Findings below this confidence are dropped entirely.
    min_confidence: float = field(default_factory=lambda: _env_float("MIN_CONFIDENCE", 0.25))
    # Findings between min_confidence and this value are kept but flagged for dentist review.
    review_confidence: float = field(default_factory=lambda: _env_float("REVIEW_CONFIDENCE", 0.6))
    model_input_size: int = field(default_factory=lambda: int(os.getenv("MODEL_INPUT_SIZE", "1024")))
    max_upload_bytes: int = field(default_factory=lambda: int(os.getenv("MAX_UPLOAD_BYTES", str(64 * 1024 * 1024))))


def get_settings() -> Settings:
    return Settings()
