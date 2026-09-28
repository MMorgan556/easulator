"""Runtime settings, read from environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass, field

MIN_ACCESS_CODE_LENGTH = 12


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    return float(value) if value else default


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    return value.strip().lower() in ("1", "true", "yes", "on") if value else default


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
    max_upload_bytes: int = field(default_factory=lambda: int(os.getenv("MAX_UPLOAD_BYTES", str(32 * 1024 * 1024))))
    # Uploads read and decoded at the same time; later ones wait without holding memory or a thread.
    max_concurrent_images: int = field(default_factory=lambda: int(os.getenv("MAX_CONCURRENT_IMAGES", "1")))
    # Reports written at the same time (each holds a worker thread while waiting on Claude).
    max_concurrent_reports: int = field(default_factory=lambda: int(os.getenv("MAX_CONCURRENT_REPORTS", "8")))
    # When set, /analyze requires this code in the X-Access-Code header.
    access_code: str = field(default_factory=lambda: os.getenv("ACCESS_CODE", "").strip())
    # Fail closed: with this on, /analyze refuses all requests until a strong ACCESS_CODE is configured.
    require_access_code: bool = field(default_factory=lambda: _env_bool("REQUIRE_ACCESS_CODE", False))

    def access_code_problem(self) -> str | None:
        """Why the access-code configuration is unsafe to serve, or None if it is fine."""
        if not self.require_access_code:
            return None
        if not self.access_code:
            return "ACCESS_CODE is not set"
        if len(self.access_code) < MIN_ACCESS_CODE_LENGTH:
            return f"ACCESS_CODE must be at least {MIN_ACCESS_CODE_LENGTH} characters"
        return None


def get_settings() -> Settings:
    return Settings()
