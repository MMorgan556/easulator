import io

import numpy as np
import pytest
from PIL import Image
from pydicom.dataset import Dataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, SecondaryCaptureImageStorage, generate_uid


def synthetic_xray(width: int, height: int) -> np.ndarray:
    """Gradient background with bright 'teeth' columns, roughly X-ray shaped."""
    y, x = np.mgrid[0:height, 0:width]
    img = 40 + 30 * (y / height)
    for cx in np.linspace(width * 0.1, width * 0.9, 12):
        img[np.abs(x - cx) < width * 0.025] += 150
    return np.clip(img, 0, 255).astype(np.uint8)


def png_bytes(width: int = 1000, height: int = 500) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(synthetic_xray(width, height)).save(buf, format="PNG")
    return buf.getvalue()


def dicom_bytes(
    width: int = 800,
    height: int = 400,
    photometric: str = "MONOCHROME2",
    description: str = "PANORAMIC",
) -> bytes:
    pixels = synthetic_xray(width, height).astype(np.uint16) * 16  # 12-bit range
    if photometric == "MONOCHROME1":
        pixels = 4095 - pixels

    meta = FileMetaDataset()
    meta.MediaStorageSOPClassUID = SecondaryCaptureImageStorage
    meta.MediaStorageSOPInstanceUID = generate_uid()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian

    ds = Dataset()
    ds.file_meta = meta
    ds.SOPClassUID = meta.MediaStorageSOPClassUID
    ds.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
    ds.PatientName = "Test^Patient"
    ds.PatientID = "SECRET-123"
    ds.SeriesDescription = description
    ds.Rows, ds.Columns = height, width
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = photometric
    ds.BitsAllocated, ds.BitsStored, ds.HighBit = 16, 12, 11
    ds.PixelRepresentation = 0
    ds.PixelSpacing = [0.1, 0.1]
    ds.PixelData = pixels.tobytes()

    buf = io.BytesIO()
    ds.save_as(buf, enforce_file_format=True)
    return buf.getvalue()


@pytest.fixture
def panoramic_png() -> bytes:
    return png_bytes()


@pytest.fixture
def panoramic_dicom() -> bytes:
    return dicom_bytes()


@pytest.fixture(autouse=True)
def isolated_env(monkeypatch):
    """Keep tests independent of whatever is configured in the developer's shell."""
    for name in (
        "ACCESS_CODE",
        "REQUIRE_ACCESS_CODE",
        "REPORT_BACKEND",
        "DETECTOR_BACKEND",
        "MAX_UPLOAD_BYTES",
        "MIN_CONFIDENCE",
        "REVIEW_CONFIDENCE",
    ):
        monkeypatch.delenv(name, raising=False)
