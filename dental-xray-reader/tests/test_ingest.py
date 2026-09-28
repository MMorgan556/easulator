import numpy as np
import pytest

from app.ingest import IngestError, guess_modality, load_image
from app.preprocess import letterbox, prepare
from app.schemas import ImageModality

from .conftest import dicom_bytes, png_bytes


def test_loads_png_as_grayscale(panoramic_png):
    loaded = load_image(panoramic_png)
    assert loaded.pixels.dtype == np.uint8
    assert loaded.pixels.shape == (500, 1000)
    assert loaded.info.source_format == "png"
    assert loaded.info.modality == ImageModality.PANORAMIC


def test_loads_dicom_with_spacing_and_no_patient_data(panoramic_dicom):
    loaded = load_image(panoramic_dicom)
    assert loaded.info.source_format == "dicom"
    assert loaded.info.pixel_spacing_mm == (0.1, 0.1)
    assert loaded.info.modality == ImageModality.PANORAMIC
    assert loaded.pixels.max() == 255
    dumped = loaded.info.model_dump_json()
    assert "SECRET-123" not in dumped and "Test^Patient" not in dumped


def test_monochrome1_is_inverted_to_match_monochrome2():
    normal = load_image(dicom_bytes(photometric="MONOCHROME2")).pixels.astype(int)
    inverted = load_image(dicom_bytes(photometric="MONOCHROME1")).pixels.astype(int)
    assert np.abs(normal - inverted).mean() < 2


def test_dicom_description_overrides_shape():
    loaded = load_image(dicom_bytes(width=800, height=400, description="Bitewing right"))
    assert loaded.info.modality == ImageModality.BITEWING


@pytest.mark.parametrize(
    ("w", "h", "expected"),
    [(2000, 1000, ImageModality.PANORAMIC), (1300, 1000, ImageModality.BITEWING), (800, 1000, ImageModality.PERIAPICAL)],
)
def test_guess_modality_from_shape(w, h, expected):
    assert guess_modality(w, h) == expected


@pytest.mark.parametrize("data", [b"", b"not an image at all"])
def test_rejects_invalid_uploads(data):
    with pytest.raises(IngestError):
        load_image(data)


def test_rejects_tiny_images():
    with pytest.raises(IngestError, match="too small"):
        load_image(png_bytes(20, 20))


def test_letterbox_round_trips_coordinates():
    pixels = np.zeros((500, 1000), dtype=np.uint8)
    out, lb = letterbox(pixels, 640)
    assert out.shape == (640, 640)
    # A point in model space maps back to the original image.
    x, y = lb.to_original(lb.pad_x + 320 * lb.scale * 2, lb.pad_y + 100 * lb.scale)
    assert x == pytest.approx(640, abs=1)
    assert y == pytest.approx(100, abs=1)


def test_prepare_stretches_contrast():
    # Low-contrast film: every value squeezed into 100-130.
    pixels = np.tile(np.linspace(100, 130, 100).astype(np.uint8), (100, 1))
    out, _ = prepare(pixels, 100)
    assert int(out.max()) - int(out.min()) > 100


@pytest.mark.parametrize("syntax", ["JPEG2000Lossless", "RLELossless"])
def test_loads_compressed_dicom(syntax):
    import io

    import pydicom
    import pydicom.uid

    ds = pydicom.dcmread(io.BytesIO(dicom_bytes()))
    ds.compress(getattr(pydicom.uid, syntax))
    buf = io.BytesIO()
    ds.save_as(buf)

    compressed = load_image(buf.getvalue())
    reference = load_image(dicom_bytes())
    assert compressed.info.source_format == "dicom"
    assert np.array_equal(compressed.pixels, reference.pixels)


def test_rejects_dicom_without_pixels():
    import io

    import pydicom

    ds = pydicom.dcmread(io.BytesIO(dicom_bytes()))
    del ds.PixelData
    buf = io.BytesIO()
    ds.save_as(buf)
    with pytest.raises(IngestError, match="no image"):
        load_image(buf.getvalue())


def test_rejects_oversized_image(monkeypatch):
    import app.ingest

    monkeypatch.setattr(app.ingest, "MAX_PIXELS", 1000)
    with pytest.raises(IngestError, match="too large"):
        load_image(png_bytes(100, 50))
    with pytest.raises(IngestError, match="too large"):
        load_image(dicom_bytes(100, 50))


@pytest.mark.parametrize(
    ("hint", "expected"),
    [
        ("OPG", ImageModality.PANORAMIC),
        ("Panoramic X-ray", ImageModality.PANORAMIC),
        ("BW left", ImageModality.BITEWING),
        ("IOPA 46", ImageModality.PERIAPICAL),
        ("Optional spa notes", ImageModality.PERIAPICAL),  # no false match; falls back to shape (800x1000)
    ],
)
def test_guess_modality_from_hint(hint, expected):
    assert guess_modality(800, 1000, hint) == expected
