import numpy as np
import pytest

from app.ingest import IngestError, guess_modality, load_image, to_uint8
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


def test_to_uint8_stretches_low_contrast_16_bit():
    # Faint 16-bit film: every value squeezed into 1000-1300.
    arr = np.tile(np.linspace(1000, 1300, 400).astype(np.uint16), (100, 1))
    out = to_uint8(arr)
    assert out.dtype == np.uint8
    assert int(out.max()) - int(out.min()) >= 250


def test_to_uint8_flat_image_is_black():
    assert not to_uint8(np.full((50, 50), 7, dtype=np.uint16)).any()


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


def _dicom_with(values: np.ndarray, **tags) -> bytes:
    import io

    import pydicom

    ds = pydicom.dcmread(io.BytesIO(dicom_bytes(width=values.shape[1], height=values.shape[0])))
    ds.PixelData = values.astype(np.uint16).tobytes()
    for key, value in tags.items():
        setattr(ds, key, value)
    buf = io.BytesIO()
    ds.save_as(buf)
    return buf.getvalue()


def test_linear_window_follows_dicom_formula():
    # Stored 0..3999 with slope 1, intercept -1000: modality values -1000..2999.
    values = np.tile(np.arange(0, 4000, 10, dtype=np.uint16), (40, 1))[:, :400]
    data = _dicom_with(values, RescaleSlope=1, RescaleIntercept=-1000, WindowCenter=500, WindowWidth=1001)
    pixels = load_image(data).pixels[0]
    modality = values[0].astype(float) - 1000
    assert (pixels[modality <= 0] == 0).all()  # at or below c - 0.5 - (w-1)/2 = 0
    assert (pixels[modality >= 1000] == 255).all()  # at or above c - 0.5 + (w-1)/2 = 1000
    mid = np.argmin(np.abs(modality - 500))
    assert pixels[mid] == 128  # (500 - 0) / 1000 * 255 = 127.5, which rounds to 128


def test_multivalue_window_uses_first_window():
    values = np.tile(np.arange(0, 4000, 10, dtype=np.uint16), (40, 1))[:, :400]
    first = load_image(_dicom_with(values, WindowCenter=[1000, 3000], WindowWidth=[500, 500])).pixels
    single = load_image(_dicom_with(values, WindowCenter=1000, WindowWidth=500)).pixels
    assert np.array_equal(first, single)


def test_multiframe_dicom_uses_first_frame():
    import io

    import pydicom

    frame0 = np.zeros((64, 128), dtype=np.uint16)
    frame0[:, 64:] = 4000
    frame1 = 4000 - frame0
    ds = pydicom.dcmread(io.BytesIO(dicom_bytes(width=128, height=64)))
    ds.NumberOfFrames = 2
    ds.PixelData = np.stack([frame0, frame1]).tobytes()
    buf = io.BytesIO()
    ds.save_as(buf)
    pixels = load_image(buf.getvalue()).pixels
    assert pixels[:, :64].mean() < 10 and pixels[:, 64:].mean() > 245


def test_exif_rotation_is_applied():
    import io

    from PIL import Image

    img = Image.new("L", (200, 100))
    exif = Image.Exif()
    exif[0x0112] = 6  # rotate 90 degrees clockwise when displayed
    buf = io.BytesIO()
    img.save(buf, format="JPEG", exif=exif)
    loaded = load_image(buf.getvalue())
    assert (loaded.info.width, loaded.info.height) == (100, 200)
    assert loaded.info.source_format == "jpeg"


def test_colour_phone_photo_size_is_accepted():
    import io

    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (4032, 3024), (40, 40, 40)).save(buf, format="JPEG")  # 12 MP, 36M samples
    loaded = load_image(buf.getvalue())
    assert (loaded.info.width, loaded.info.height) == (4032, 3024)


def test_faint_8_bit_scan_is_stretched():
    import io

    from PIL import Image

    faint = np.tile(np.linspace(100, 130, 400).astype(np.uint8), (100, 1))
    buf = io.BytesIO()
    Image.fromarray(faint).save(buf, format="PNG")
    pixels = load_image(buf.getvalue()).pixels
    assert int(pixels.max()) - int(pixels.min()) >= 250


@pytest.mark.parametrize(
    "tags",
    [
        {"WindowCenter": 500, "WindowWidth": 0},  # degenerate width
        {"WindowCenter": 500, "WindowWidth": 1},  # step function; not handled by the linear path
        {"WindowCenter": "", "WindowWidth": ""},  # empty values
        {"WindowCenter": 500, "WindowWidth": 0, "VOILUTFunction": "SIGMOID"},  # pydicom rejects this
        {"WindowCenter": 500},  # width missing
    ],
)
def test_unusable_window_falls_back_to_percentile_stretch(tags):
    values = np.tile(np.arange(0, 4000, 10, dtype=np.uint16), (40, 1))[:, :400]
    pixels = load_image(_dicom_with(values, **tags)).pixels
    assert pixels.min() == 0 and pixels.max() == 255


def test_malformed_pixel_spacing_is_ignored():
    values = np.zeros((64, 64), dtype=np.uint16)
    values[:, 32:] = 1000
    loaded = load_image(_dicom_with(values, PixelSpacing=[0.1]))
    assert loaded.info.pixel_spacing_mm is None


def test_exif_transpose_does_not_break_untagged_images(panoramic_png):
    loaded = load_image(panoramic_png)
    assert (loaded.info.width, loaded.info.height, loaded.info.source_format) == (1000, 500, "png")


def test_to_uint8_overwrite_matches_copy():
    arr = np.random.default_rng(1).normal(1000, 200, (64, 64)).astype(np.float32)
    expected = to_uint8(arr.copy())
    assert np.array_equal(to_uint8(arr, overwrite=True), expected)
    original = np.random.default_rng(1).normal(1000, 200, (64, 64)).astype(np.float32)
    kept = original.copy()
    to_uint8(original)
    assert np.array_equal(original, kept)  # default never modifies the input
