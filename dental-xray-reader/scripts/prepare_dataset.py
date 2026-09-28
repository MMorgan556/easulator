"""Export radiographs to PNG exactly as the service sees them, for annotation and training.

Usage:
    python scripts/prepare_dataset.py raw_xrays/ datasets/dental/images/train/

Every DICOM or image file under the source folder is decoded with app.ingest (DICOM
windowing, MONOCHROME1 inversion, 16-bit contrast stretch, EXIF rotation) and written
as an 8-bit grayscale PNG, keeping the folder structure. Annotate the exported PNGs,
not the originals, so label coordinates match what the model is given in production.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.ingest import IngestError, load_image  # noqa: E402


def export(src: Path, dst: Path) -> tuple[int, list[str]]:
    written, failures = 0, []
    claimed: set[Path] = set()
    for path in sorted(p for p in src.rglob("*") if p.is_file() and not p.name.startswith(".")):
        out = dst / path.relative_to(src).with_suffix(".png")
        if out in claimed:
            failures.append(f"{path}: another file already exports to {out.name}")
            continue
        try:
            loaded = load_image(path.read_bytes())
        except IngestError as exc:
            failures.append(f"{path}: {exc}")
            continue
        claimed.add(out)
        out.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(loaded.pixels).save(out)
        written += 1
    return written, failures


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("src", type=Path, help="Folder of DICOM/PNG/JPEG/TIFF radiographs")
    parser.add_argument("dst", type=Path, help="Output folder for PNGs")
    args = parser.parse_args()
    if not args.src.is_dir():
        sys.exit(f"{args.src} is not a folder")

    written, failures = export(args.src, args.dst)
    for line in failures:
        print(f"skipped {line}", file=sys.stderr)
    print(f"Exported {written} image(s) to {args.dst}; skipped {len(failures)}.")


if __name__ == "__main__":
    main()
