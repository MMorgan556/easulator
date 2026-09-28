"""Contrast enhancement and letterbox resizing for model input."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image

from .ingest import to_uint8


def equalize_contrast(pixels: np.ndarray, clip_limit: float = 0.01) -> np.ndarray:
    """Clipped global histogram equalization.

    A dependency-free stand-in for CLAHE; swap in ``cv2.createCLAHE`` when OpenCV
    is installed for better local contrast on enamel/dentin boundaries.
    """
    hist = np.bincount(pixels.ravel(), minlength=256).astype(np.float64)
    limit = max(1.0, clip_limit * pixels.size)
    excess = np.clip(hist - limit, 0, None).sum()
    hist = np.minimum(hist, limit) + excess / 256
    cdf = hist.cumsum()
    cdf = (cdf - cdf.min()) / (cdf.max() - cdf.min() or 1.0)
    lut = (cdf * 255).round().astype(np.uint8)
    return lut[pixels]


@dataclass
class Letterbox:
    """Maps coordinates between the model input and the original image."""

    scale: float
    pad_x: int
    pad_y: int

    def to_original(self, x: float, y: float) -> tuple[float, float]:
        return (x - self.pad_x) / self.scale, (y - self.pad_y) / self.scale


def letterbox(pixels: np.ndarray, size: int) -> tuple[np.ndarray, Letterbox]:
    """Resize keeping aspect ratio and pad to a ``size`` x ``size`` square."""
    h, w = pixels.shape
    scale = size / max(h, w)
    new_w, new_h = max(1, round(w * scale)), max(1, round(h * scale))
    resized = np.asarray(Image.fromarray(pixels).resize((new_w, new_h), Image.BILINEAR))
    canvas = np.zeros((size, size), dtype=np.uint8)
    pad_x, pad_y = (size - new_w) // 2, (size - new_h) // 2
    canvas[pad_y : pad_y + new_h, pad_x : pad_x + new_w] = resized
    return canvas, Letterbox(scale=scale, pad_x=pad_x, pad_y=pad_y)


def prepare(pixels: np.ndarray, size: int) -> tuple[np.ndarray, Letterbox]:
    # Stretch first so faint films use the full range, then equalize the histogram.
    return letterbox(equalize_contrast(to_uint8(pixels)), size)
