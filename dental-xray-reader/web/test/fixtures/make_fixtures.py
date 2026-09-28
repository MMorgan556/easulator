"""Generate DICOM/image fixtures and the exact 8-bit images app/ingest.py derives from them.

The browser code must reproduce these byte-for-byte (JPEG baseline within a small tolerance,
since browsers and libjpeg may round differently), so an AI model trained on images from
scripts/prepare_dataset.py sees identical input in the web app.

Run from dental-xray-reader/:  python web/test/fixtures/make_fixtures.py
Needs: pydicom, pylibjpeg(-libjpeg, -openjpeg), pyjpegls, python-gdcm, pillow, numpy.
"""

from __future__ import annotations

import io
import json
import sys
from pathlib import Path

import numpy as np
import pydicom
import pydicom.uid as uid
from PIL import Image
from pydicom.dataset import Dataset, FileMetaDataset

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from app.ingest import load_image  # noqa: E402

OUT = Path(__file__).resolve().parent
W, H = 96, 48


def pattern(bits: int = 12, signed: bool = False, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:H, 0:W]
    top = 2**bits - 1
    base = (x / W) * top * 0.6 + (y / H) * top * 0.2
    base[:, ::7] += top * 0.15  # "teeth"
    base += rng.normal(0, top * 0.02, base.shape)
    arr = np.clip(base, 0, top)
    if signed:
        arr = arr - 2 ** (bits - 1)
        return arr.astype(np.int16)
    return arr.astype(np.uint16 if bits > 8 else np.uint8)


def dataset(pixels: np.ndarray, **tags) -> Dataset:
    meta = FileMetaDataset()
    meta.MediaStorageSOPClassUID = uid.SecondaryCaptureImageStorage
    meta.MediaStorageSOPInstanceUID = uid.generate_uid()
    meta.TransferSyntaxUID = uid.ExplicitVRLittleEndian
    ds = Dataset()
    ds.file_meta = meta
    ds.SOPClassUID = meta.MediaStorageSOPClassUID
    ds.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
    ds.PatientName = "Fixture^Patient"
    frames = pixels.shape[0] if pixels.ndim == 3 and tags.get("NumberOfFrames") else 1
    rows, cols = (pixels.shape[1:3] if frames > 1 else pixels.shape[:2])
    ds.Rows, ds.Columns = rows, cols
    samples = pixels.shape[-1] if (pixels.ndim == 3 and frames == 1) or pixels.ndim == 4 else 1
    ds.SamplesPerPixel = samples
    ds.PhotometricInterpretation = "RGB" if samples == 3 else "MONOCHROME2"
    if samples == 3:
        ds.PlanarConfiguration = 0
    bits = 8 if pixels.dtype.itemsize == 1 else 16
    ds.BitsAllocated = bits
    ds.BitsStored = tags.pop("BitsStored", 12 if bits == 16 else 8)
    ds.HighBit = ds.BitsStored - 1
    ds.PixelRepresentation = 1 if pixels.dtype.kind == "i" else 0
    for key, value in tags.items():
        setattr(ds, key, value)
    ds.PixelData = pixels.tobytes()
    return ds


def save(name: str, ds: Dataset, compress=None, cases=None, tolerance=0, **write) -> None:
    if compress is not None:
        ds.compress(compress)
    buf = io.BytesIO()
    ds.save_as(buf, enforce_file_format=True, **write)
    data = buf.getvalue()
    write_case(name, data, "dcm", cases, tolerance)


def write_case(name, data, ext, cases, tolerance=0):
    loaded = load_image(data)
    if ext != "dcm":  # what the browser's image decoder hands the JavaScript
        (OUT / f"{name}.rgba.bin").write_bytes(np.asarray(Image.open(io.BytesIO(data)).convert("RGBA")).tobytes())
    (OUT / f"{name}.{ext}").write_bytes(data)
    (OUT / f"{name}.expected.bin").write_bytes(loaded.pixels.tobytes())
    cases.append(
        {
            "name": name,
            "file": f"{name}.{ext}",
            "width": loaded.info.width,
            "height": loaded.info.height,
            "pixel_spacing_mm": loaded.info.pixel_spacing_mm,
            "modality": loaded.info.modality.value,
            "tolerance": tolerance,
        }
    )


def gdcm_compress(ds: Dataset, syntax: str) -> Dataset:
    """Compress with GDCM (pydicom has no JPEG baseline / JPEG lossless encoder)."""
    import gdcm

    buf = io.BytesIO()
    ds.save_as(buf, enforce_file_format=True)
    reader = gdcm.ImageReader()
    tmp_in = OUT / "_tmp_in.dcm"
    tmp_out = OUT / "_tmp_out.dcm"
    tmp_in.write_bytes(buf.getvalue())
    reader.SetFileName(str(tmp_in))
    assert reader.Read()
    change = gdcm.ImageChangeTransferSyntax()
    change.SetTransferSyntax(gdcm.TransferSyntax(getattr(gdcm.TransferSyntax, syntax)))
    change.SetInput(reader.GetImage())
    assert change.Change()
    writer = gdcm.ImageWriter()
    writer.SetFileName(str(tmp_out))
    writer.SetFile(reader.GetFile())
    writer.SetImage(change.GetOutput())
    assert writer.Write()
    out = pydicom.dcmread(tmp_out)
    tmp_in.unlink()
    tmp_out.unlink()
    return out


def main() -> None:
    for old in OUT.glob("*"):
        if old.suffix in (".dcm", ".bin", ".png", ".jpg") or old.name == "cases.json":
            old.unlink()
    cases: list[dict] = []
    p12 = pattern(12)

    save("native16", dataset(p12, PixelSpacing=[0.1, 0.12], SeriesDescription="PANORAMIC"), cases=cases)
    save("monochrome1", dataset(p12, PhotometricInterpretation="MONOCHROME1"), cases=cases)
    save("rescale_window", dataset(p12, RescaleSlope=2, RescaleIntercept=-1024, WindowCenter=2000, WindowWidth=3000), cases=cases)
    save("window_multivalue", dataset(p12, WindowCenter=[1500, 3000], WindowWidth=[1200, 500]), cases=cases)
    save("window_degenerate", dataset(p12, WindowCenter=500, WindowWidth=0), cases=cases)
    save("window_sigmoid", dataset(p12, WindowCenter=1500, WindowWidth=1200, VOILUTFunction="SIGMOID"), cases=cases)
    save("signed16", dataset(pattern(12, signed=True), RescaleSlope=1, RescaleIntercept=-1000), cases=cases)
    save("native8", dataset(pattern(8), BitsStored=8), cases=cases)
    rgb = np.stack([pattern(8, seed=s) for s in (1, 2, 3)], axis=-1)
    save("rgb8", dataset(rgb, BitsStored=8), cases=cases)
    ds = dataset(rgb, BitsStored=8)
    ds.PlanarConfiguration = 1
    ds.PixelData = np.ascontiguousarray(rgb.transpose(2, 0, 1)).tobytes()
    save("rgb8_planar", ds, cases=cases)
    frames = np.stack([p12, 4095 - p12])
    save("multiframe", dataset(frames, NumberOfFrames=2), cases=cases)
    ds = dataset(p12)
    ds.file_meta.TransferSyntaxUID = uid.ExplicitVRBigEndian
    ds.PixelData = p12.astype(">u2").tobytes()
    save("big_endian16", ds, cases=cases, implicit_vr=False, little_endian=False)

    save("rle_multiframe", dataset(frames, NumberOfFrames=2), compress=uid.RLELossless, cases=cases)
    save("j2k_multiframe", dataset(frames, NumberOfFrames=2), compress=uid.JPEG2000Lossless, cases=cases)
    save("rle16", dataset(p12), compress=uid.RLELossless, cases=cases)
    save("rle8", dataset(pattern(8), BitsStored=8), compress=uid.RLELossless, cases=cases)
    save("rle_rgb", dataset(rgb, BitsStored=8), compress=uid.RLELossless, cases=cases)
    save("j2k16", dataset(p12, WindowCenter=1800, WindowWidth=2500), compress=uid.JPEG2000Lossless, cases=cases)
    save("j2k8", dataset(pattern(8), BitsStored=8), compress=uid.JPEG2000Lossless, cases=cases)
    save("jls16", dataset(p12), compress=uid.JPEGLSLossless, cases=cases)
    save("jpeg_lossless16", gdcm_compress(dataset(p12), "JPEGLosslessProcess14_1"), cases=cases)
    save("jpeg_lossless8", gdcm_compress(dataset(pattern(8), BitsStored=8), "JPEGLosslessProcess14_1"), cases=cases)
    save("jpeg_baseline8", gdcm_compress(dataset(pattern(8), BitsStored=8), "JPEGBaselineProcess1"), cases=cases, tolerance=2)

    # Edge cases: unused high bits, signed data stored without sign extension, planar RLE,
    # colour JPEG 2000 / JPEG-LS, LINEAR_EXACT, VOI and Modality LUT Sequences.
    ds = dataset(p12)
    ds.PixelData = (p12 | 0xF000).astype(np.uint16).tobytes()
    save("native_high_bits", ds, cases=cases)
    ds = dataset(p12)
    ds.PixelData = (p12 | 0xF000).astype(np.uint16).tobytes()
    save("rle_high_bits", ds, compress=uid.RLELossless, cases=cases)
    s12 = pattern(12, signed=True)
    ds = dataset(s12)
    ds.PixelData = (s12.astype(np.int32) & 0xFFF).astype(np.uint16).tobytes()
    save("rle_signed_unextended", ds, compress=uid.RLELossless, cases=cases)
    ds = dataset(rgb, BitsStored=8)
    ds.compress(uid.RLELossless)
    ds.PlanarConfiguration = 1
    save("rle_rgb_planar1", ds, cases=cases)
    save("j2k_rgb", dataset(rgb, BitsStored=8), compress=uid.JPEG2000Lossless, cases=cases)
    save("jls_rgb", dataset(rgb, BitsStored=8), compress=uid.JPEGLSLossless, cases=cases)
    save("window_linear_exact", dataset(p12, RescaleSlope=1, RescaleIntercept=0, WindowCenter=1500, WindowWidth=1200, VOILUTFunction="LINEAR_EXACT"), cases=cases)
    for name, bits, table in [
        ("voi_lut16", 16, (np.linspace(0, 1, 4096) ** 0.5 * 65535).astype(np.uint16)),
        ("voi_lut8", 8, (np.linspace(0, 1, 2048) ** 2 * 255).astype(np.uint16)),
    ]:
        ds = dataset(p12, WindowCenter=100, WindowWidth=50)  # the LUT must win over the window
        item = Dataset()
        item.LUTDescriptor = [len(table), 1000 if bits == 8 else 0, bits]
        if bits == 16:
            item.add_new(0x00283006, "US", table.tolist())  # LUT Data as US values
        else:
            item.add_new(0x00283006, "OW", table.astype("<u2").tobytes())  # ... and as OW bytes
        ds.VOILUTSequence = [item]
        save(name, ds, cases=cases)
    ds = dataset(p12, WindowCenter=30000, WindowWidth=40000)
    item = Dataset()
    item.LUTDescriptor = [4096, 0, 16]
    item.add_new(0x00283006, "OW", (np.arange(4096) * 16 + 7).astype("<u2").tobytes())
    ds.ModalityLUTSequence = [item]
    save("modality_lut", ds, cases=cases)
    ds = dataset(p12)
    item = Dataset()
    item.LUTDescriptor = [4096, 0, 9]  # unsupported depth: pydicom raises, ingest skips the VOI
    item.add_new(0x00283006, "OW", np.arange(4096, dtype="<u2").tobytes())
    ds.VOILUTSequence = [item]
    save("voi_lut_invalid", ds, cases=cases)

    # Raster uploads (decoded by the browser, then Pillow's luma + percentile stretch).
    gray = Image.fromarray(pattern(8, seed=4))
    buf = io.BytesIO()
    gray.save(buf, format="PNG")
    write_case("gray_png", buf.getvalue(), "png", cases)
    color = Image.fromarray(rgb)
    buf = io.BytesIO()
    color.save(buf, format="PNG")
    write_case("rgb_png", buf.getvalue(), "png", cases)

    # 16-bit grayscale PNGs (browsers only give 8 bits, so the app decodes these itself).
    import png

    g16 = (pattern(12, seed=5).astype(np.uint32) * 3 + 20000).astype(np.uint16)
    for name, interlace in (("gray16_png", False), ("gray16_interlaced_png", True)):
        buf = io.BytesIO()
        png.Writer(W, H, greyscale=True, bitdepth=16, interlace=interlace).write(buf, g16.tolist())
        write_case(name, buf.getvalue(), "png", cases)

    (OUT / "cases.json").write_text(json.dumps(cases, indent=1) + "\n")
    print(f"wrote {len(cases)} cases to {OUT}")


if __name__ == "__main__":
    main()
