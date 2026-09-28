"""Train the dental detector with Ultralytics YOLO.

Usage:
    pip install -r requirements-ml.txt
    python scripts/train_yolo.py --data configs/dental.yaml --model yolo11m.pt --epochs 150

Train on images exported with scripts/prepare_dataset.py so they are normalized exactly
as the service normalizes uploads. The best checkpoint is copied to weights/dental-yolo.pt,
which DETECTOR_BACKEND=yolo loads.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.schemas import FindingType  # noqa: E402


def resolve_dataset_config(data_yaml: Path, out_dir: Path) -> Path:
    """Validate the dataset config and write a copy whose ``path`` is absolute.

    Ultralytics resolves a relative ``path`` against its own datasets directory, not the
    config file, so a relative path would silently point somewhere else.
    """
    import yaml

    config = yaml.safe_load(data_yaml.read_text())
    names = config["names"]
    names = list(names.values()) if isinstance(names, dict) else list(names)
    unknown = set(names) - {t.value for t in FindingType}
    if unknown:
        sys.exit(f"{data_yaml}: classes {sorted(unknown)} are not FindingType values and would be ignored at inference")

    root = Path(config.get("path", "."))
    if not root.is_absolute():
        root = (data_yaml.resolve().parent / root).resolve()
    if not root.is_dir():
        sys.exit(f"Dataset folder {root} does not exist (from 'path' in {data_yaml})")
    config["path"] = str(root)

    out_dir.mkdir(parents=True, exist_ok=True)
    resolved = out_dir / "dataset.resolved.yaml"
    resolved.write_text(yaml.safe_dump(config, sort_keys=False))
    return resolved


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, default=ROOT / "configs" / "dental.yaml")
    parser.add_argument("--model", default="yolo11m.pt", help="Pretrained checkpoint to fine-tune")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default=None, help="e.g. 0, 0,1 or cpu")
    args = parser.parse_args()

    data_config = resolve_dataset_config(args.data, ROOT / "runs")

    from ultralytics import YOLO

    model = YOLO(args.model)
    model.train(
        data=str(data_config),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=str(ROOT / "runs"),
        name="dental",
        # X-rays are grayscale and orientation matters for FDI numbering:
        # no colour jitter and no horizontal flips (they swap patient left/right).
        hsv_h=0.0,
        hsv_s=0.0,
        fliplr=0.0,
        degrees=5.0,
        scale=0.3,
        patience=30,
    )
    metrics = model.val()
    print(f"mAP50-95: {metrics.box.map:.3f}  mAP50: {metrics.box.map50:.3f}")

    best = Path(model.trainer.best)
    out = ROOT / "weights" / "dental-yolo.pt"
    out.parent.mkdir(exist_ok=True)
    shutil.copy(best, out)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
