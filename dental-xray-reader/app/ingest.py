"""Load dental X-rays from DICOM or common image formats into a normalized grayscale array.

No patient-identifying DICOM tags are read or returned: only pixel data and the
geometry needed for measurement leave this module.
"""

from __future__ import annotations

import io
import re
from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError
from pydicom.multival import MultiValue

from .schemas import ImageInfo, ImageModality

DICOM_MAGIC_OFFSET = 128
DICOM_MAGIC = b"DICM"
# Largest image accepted (panoramic sensors are typically under 10 MP).
# Sized so decoding stays well inside a 512 MB container.
MAX_PIXELS = 20_000_000


class IngestError(ValueError):
    """The upload could not be decoded as a dental X-ray."""


@dataclass
class LoadedImage:
    pixels: np.ndarray  # uint8, shape (H, W)
    info: ImageInfo


def is_dicom(data: bytes) -> bool:
    return data[DICOM_MAGIC_OFFSET : DICOM_MAGIC_OFFSET + 4] == DICOM_MAGIC


def to_uint8(arr: np.ndarray, *, overwrite: bool = False) -> np.ndarray:
    """Scale an arbitrary-range array to 0-255 using robust percentiles.

    With ``overwrite=True`` a float32 input is used as the working buffer instead of being
    copied; pass it only for arrays nothing else will read afterwards.
    """
    # Percentiles from a 1-in-16 sample: same result on real images, without a full-size copy.
    lo, hi = (float(v) for v in np.percentile(arr[::4, ::4], (0.5, 99.5)))
    if hi <= lo:
        lo, hi = float(arr.min()), float(arr.max())
    if hi <= lo:
        return np.zeros(arr.shape, dtype=np.uint8)
    out = arr.astype(np.float32, copy=not overwrite)  # one working buffer, updated in place below
    out -= lo
    out *= 255.0 / (hi - lo)
    np.clip(out, 0.0, 255.0, out=out)
    np.rint(out, out=out)
    return out.astype(np.uint8)


def guess_modality(width: int, height: int, hint: str | None = None) -> ImageModality:
    """Best-effort guess of the radiograph type from DICOM hints or image shape."""
    if hint:
        words = re.findall(r"[a-z]+", hint.lower())
        if any(w.startswith("pan") or w in ("opg", "opt", "dpt") for w in words):
            return ImageModality.PANORAMIC
        if any(w.startswith("bitewing") or w in ("bw", "bwx") for w in words):
            return ImageModality.BITEWING
        if any(w.startswith("periapical") or w in ("pa", "iopa") for w in words):
            return ImageModality.PERIAPICAL
    aspect = width / height if height else 0
    if aspect >= 1.7:
        return ImageModality.PANORAMIC
    if aspect >= 1.15:
        return ImageModality.BITEWING
    if aspect > 0:
        return ImageModality.PERIAPICAL
    return ImageModality.UNKNOWN


def _first_value(value) -> float:
    """DICOM window tags may hold several values; the first is the default window."""
    return float(value[0] if isinstance(value, (list, tuple, MultiValue)) else value)


def _linear_window(ds) -> tuple[float, float] | None:
    """(center, width) of a usable linear DICOM window, or None."""
    # An empty LUT sequence counts as absent, here and below (as in pydicom's own checks).
    if ds.get("VOILUTSequence") or "WindowCenter" not in ds or "WindowWidth" not in ds:
        return None
    if str(ds.get("VOILUTFunction", "LINEAR") or "LINEAR").upper() != "LINEAR":
        return None
    try:
        center, width = _first_value(ds.WindowCenter), _first_value(ds.WindowWidth)
    except (TypeError, ValueError, IndexError):
        return None  # empty or malformed window values: fall back to a percentile stretch
    if not (np.isfinite(center) and np.isfinite(width)) or width <= 1:
        return None
    return center, width


def _window_to_uint8(arr: np.ndarray, center: float, width: float) -> np.ndarray:
    """Apply a linear DICOM window straight to 0-255 in float32."""
    # DICOM PS3.3 C.11.2.1.2 linear window: [c - 0.5 - (w-1)/2, c - 0.5 + (w-1)/2] maps to [0, 255].
    low = center - 0.5 - (width - 1) / 2
    out = arr.astype(np.float32, copy=False)  # arr is a private working array, safe to modify
    out -= low
    out *= 255.0 / (width - 1)
    np.clip(out, 0.0, 255.0, out=out)
    np.rint(out, out=out)
    return out.astype(np.uint8)


def _dicom_display_pixels(arr: np.ndarray, ds, samples: int) -> np.ndarray:
    """Stored DICOM pixel values to an 8-bit display image (rescale, window, stretch)."""
    from pydicom.pixels import apply_modality_lut

    if arr.ndim == 3 and samples > 1:
        arr = arr.mean(axis=-1, dtype=np.float32)  # colour data: collapse to grayscale
    if arr.ndim != 2:
        raise IngestError(f"Unsupported DICOM pixel array shape {arr.shape}")

    if ds.get("ModalityLUTSequence"):
        arr = apply_modality_lut(arr, ds)
    else:
        # Linear rescale done in float32 in place (pydicom's version allocates float64).
        slope = float(ds.get("RescaleSlope", 1) or 1)
        intercept = float(ds.get("RescaleIntercept", 0) or 0)
        if slope != 1 or intercept != 0:
            arr = arr.astype(np.float32)
            arr *= slope
            arr += intercept

    window = _linear_window(ds)
    if window is not None:
        return _window_to_uint8(arr, *window)
    if ds.get("VOILUTSequence") or str(ds.get("VOILUTFunction", "LINEAR") or "LINEAR").upper() != "LINEAR":
        # Lookup-table or non-linear VOI. If it is malformed, show the image without it rather
        # than reject it: the percentile stretch below still gives a usable picture.
        try:
            arr = _apply_voi(arr, ds)
        except Exception:
            pass
    return to_uint8(arr, overwrite=True)  # arr is this function's private working array


def _apply_voi(arr: np.ndarray, ds) -> np.ndarray:
    """pydicom.apply_voi_lut's choice (a complete LUT item wins, else the window), with our LUT lookup."""
    from pydicom.pixels import apply_voi_lut

    sequence = ds.get("VOILUTSequence")
    if sequence and sequence[0].get("LUTDescriptor") is not None and sequence[0].get("LUTData") is not None:
        return _voi_lut_sequence(arr, ds)
    return apply_voi_lut(arr, ds)


def _voi_lut_sequence(arr: np.ndarray, ds) -> np.ndarray:
    """First VOI LUT Sequence item applied like pydicom.apply_voi, with a correct index type.

    pydicom builds the lookup indices in the LUT's own dtype, so for 8-bit LUTs every index
    wraps modulo 256 and the image collapses to a few gray levels.
    """
    item = ds.VOILUTSequence[0]
    entries, first_map, bits = (int(v) for v in item.LUTDescriptor)
    entries = entries or 65536
    if bits == 8:
        dtype = np.uint8
    elif 10 <= bits <= 16:
        dtype = np.uint16
    else:
        raise NotImplementedError(f"{bits} bits per VOI LUT entry is not supported")
    element = item["LUTData"]
    if element.VR == "OW":
        # The dataset's own encoding also covers files without file meta information.
        order = ">" if ds.original_encoding[1] is False else "<"
        raw = np.frombuffer(element.value, dtype=f"{order}u2", count=entries)
    else:
        raw = np.asarray(element.value, dtype=np.int64)[:entries]
    if raw.size < entries or raw.max(initial=0) > np.iinfo(dtype).max:
        raise ValueError("VOI LUT data does not match its descriptor")
    lut = raw.astype(dtype)
    # Values below the first mapped value take entry 0; above the table, the last entry.
    values = arr.astype(np.float64)  # no integer overflow whatever first_map and the pixel dtype are
    index = np.where(values >= first_map, values - first_map, 0)
    index = np.clip(index, 0, entries - 1).astype(np.int64)  # truncates, like pydicom's cast
    return lut[index]


def _pixel_spacing(ds) -> tuple[float, float] | None:
    """Row/column spacing in mm, or None when absent or malformed (it is optional metadata)."""
    spacing = ds.get("PixelSpacing") or ds.get("ImagerPixelSpacing")
    try:
        rows, cols = float(spacing[0]), float(spacing[1])
    except (TypeError, ValueError, IndexError):
        return None
    if not (np.isfinite(rows) and np.isfinite(cols)) or rows <= 0 or cols <= 0:
        return None
    return rows, cols


def _load_dicom(data: bytes) -> LoadedImage:
    import pydicom
    from pydicom.pixels import pixel_array

    try:
        # Header only: size checks run before any pixel data is decoded.
        ds = pydicom.dcmread(io.BytesIO(data), stop_before_pixels=True)
        samples = int(ds.get("SamplesPerPixel", 1) or 1)
        width, height = int(ds.get("Columns", 0) or 0), int(ds.get("Rows", 0) or 0)
        frames = int(ds.get("NumberOfFrames", 1) or 1)
    except Exception as exc:  # pydicom raises many exception types for bad files
        raise IngestError(f"Could not read DICOM file: {exc}") from exc
    _check_size(width, height)
    try:
        # Decoding from the source keeps no second copy in a Dataset; multi-frame files decode frame 0 only.
        arr = pixel_array(io.BytesIO(data), index=0 if frames > 1 else None)
    except AttributeError as exc:
        raise IngestError("DICOM file contains no image") from exc
    except Exception as exc:
        raise IngestError(f"Could not decode DICOM pixel data: {exc}") from exc

    try:
        pixels = _dicom_display_pixels(arr, ds, samples)
    except IngestError:
        raise
    except Exception as exc:  # malformed rescale/LUT values in an untrusted file
        raise IngestError(f"Could not process DICOM image: {exc}") from exc
    del arr
    if ds.get("PhotometricInterpretation") == "MONOCHROME1":
        pixels = 255 - pixels  # MONOCHROME1 stores bright = low values

    pixel_spacing = _pixel_spacing(ds)
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


def _check_size(width: int, height: int) -> None:
    if width * height > MAX_PIXELS:
        raise IngestError(f"Image too large ({width}x{height}); the limit is {MAX_PIXELS:,} pixels")


def _load_raster(data: bytes) -> LoadedImage:
    try:
        img = Image.open(io.BytesIO(data))
    except Image.DecompressionBombError as exc:
        raise IngestError("Image too large to process") from exc
    except (UnidentifiedImageError, OSError) as exc:
        raise IngestError("Upload is neither DICOM nor a readable image file") from exc
    fmt = (img.format or "unknown").lower()  # read before exif_transpose, which drops it
    # Image.open only reads the header, so this runs before decoding.
    _check_size(*img.size)
    try:
        img.load()
        # Honour EXIF rotation (phone photos of films), so boxes match what viewers display.
        ImageOps.exif_transpose(img, in_place=True)
    except OSError as exc:
        raise IngestError(f"Image file is damaged or truncated: {exc}") from exc

    if img.mode in ("I;16", "I;16B", "I", "F"):
        pixels = to_uint8(np.asarray(img))
    else:
        # 8-bit scans get the same percentile stretch as DICOM and 16-bit input, so faint films
        # reach the model (and exported training images) with a consistent contrast range.
        pixels = to_uint8(np.asarray(img.convert("L"), dtype=np.uint8))

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
