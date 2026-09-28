"""Load dental X-rays from DICOM or common image formats into a normalized grayscale array.

No patient-identifying DICOM tags are read or returned: only pixel data and the
geometry needed for measurement leave this module.
"""

from __future__ import annotations

import io
from dataclasses import dataclass

import numpy as np
from PIL import Image, UnidentifiedImageError

from .schemas import ImageInfo, ImageModality

DICOM_MAGIC_OFFSET = 128
DICOM_MAGIC = b"DICM"


class IngestError(ValueError):
    """The upload could not be decoded as a dental X-ray."""


@dataclass
class LoadedImage:
    pixels: np.ndarray  # uint8, shape (H, W)
    info: ImageInfo


def is_dicom(data: bytes) -> bool:
    return data[DICOM_MAGIC_OFFSET : DICOM_MAGIC_OFFSET + 4] == DICOM_MAGIC


def to_uint8(arr: np.ndarray) -> np.ndarray:
    """Scale an arbitrary-range array to 0-255 using robust percentiles."""
    arr = arr.astype(np.float32)
    lo, hi = np.percentile(arr, (0.5, 99.5))
    if hi <= lo:
        lo, hi = float(arr.min()), float(arr.max())
    if hi <= lo:
        return np.zeros(arr.shape, dtype=np.uint8)
    scaled = np.clip((arr - lo) / (hi - lo), 0.0, 1.0)
    return (scaled * 255).round().astype(np.uint8)


def guess_modality(width: int, height: int, hint: str | None = None) -> ImageModality:
    """Best-effort guess of the radiograph type from DICOM hints or image shape."""
    if hint:
        text = hint.lower()
        if "pan" in text:
            return ImageModality.PANORAMIC
        if "bitewing" in text or "bwx" in text:
            return ImageModality.BITEWING
        if "periapical" in text or "pa " in f"{text} ":
            return ImageModality.PERIAPICAL
    aspect = width / height if height else 0
    if aspect >= 1.7:
        return ImageModality.PANORAMIC
    if aspect >= 1.15:
        return ImageModality.BITEWING
    if aspect > 0:
        return ImageModality.PERIAPICAL
    return ImageModality.UNKNOWN


def _load_dicom(data: bytes) -> LoadedImage:
    import pydicom
    from pydicom.pixels import apply_modality_lut, apply_voi_lut

    try:
        ds = pydicom.dcmread(io.BytesIO(data))
        arr = ds.pixel_array
    except Exception as exc:  # pydicom raises many exception types for bad files
        raise IngestError(f"Could not decode DICOM pixel data: {exc}") from exc

    if arr.ndim == 3 and ds.get("SamplesPerPixel", 1) == 1:
        arr = arr[0]  # multi-frame: use the first frame
    if arr.ndim == 3:
        arr = arr.mean(axis=-1)  # colour data: collapse to grayscale
    if arr.ndim != 2:
        raise IngestError(f"Unsupported DICOM pixel array shape {arr.shape}")

    arr = apply_modality_lut(arr, ds)
    if "WindowCenter" in ds or "VOILUTSequence" in ds:
        arr = apply_voi_lut(arr, ds)
    pixels = to_uint8(arr)
    if ds.get("PhotometricInterpretation") == "MONOCHROME1":
        pixels = 255 - pixels  # MONOCHROME1 stores bright = low values

    spacing = ds.get("PixelSpacing") or ds.get("ImagerPixelSpacing")
    pixel_spacing = (float(spacing[0]), float(spacing[1])) if spacing else None
    hint = " ".join(str(ds.get(tag, "")) for tag in ("SeriesDescription", "StudyDescription", "BodyPartExamined"))

    height, width = pixels.shape
    return LoadedImage(
        pixels=pixels,
        info=ImageInfo(
            width=width,
            height=height,
            modality=guess_modality(width, height, hint),
            source_format="dicom",
            pixel_spacing_mm=pixel_spacing,
        ),
    )


def _load_raster(data: bytes) -> LoadedImage:
    try:
        img = Image.open(io.BytesIO(data))
        img.load()
    except (UnidentifiedImageError, OSError) as exc:
        raise IngestError("Upload is neither DICOM nor a readable image file") from exc

    fmt = (img.format or "unknown").lower()
    if img.mode in ("I;16", "I;16B", "I", "F"):
        pixels = to_uint8(np.asarray(img))
    else:
        pixels = np.asarray(img.convert("L"), dtype=np.uint8)

    height, width = pixels.shape
    return LoadedImage(
        pixels=pixels,
        info=ImageInfo(width=width, height=height, modality=guess_modality(width, height), source_format=fmt),
    )


def load_image(data: bytes) -> LoadedImage:
    if not data:
        raise IngestError("Empty upload")
    loaded = _load_dicom(data) if is_dicom(data) else _load_raster(data)
    if min(loaded.pixels.shape) < 32:
        raise IngestError(f"Image too small to analyze ({loaded.info.width}x{loaded.info.height})")
    return loaded
