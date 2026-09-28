"""Random detection scenes with app/postprocess.py's results, replayed by test/findings.test.mjs.

Run from dental-xray-reader/:  python web/test/fixtures/make_findings_cases.py
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from app.ingest import guess_modality  # noqa: E402
from app.postprocess import build_findings  # noqa: E402
from app.schemas import BoundingBox, Detection, FindingType, ImageModality  # noqa: E402

OUT = Path(__file__).resolve().parent / "findings_cases.json"


def scene(rng: random.Random) -> dict:
    width, height = rng.choice([(1600, 800), (1000, 500), (800, 1000), (1300, 1000)])
    modality = rng.choice([ImageModality.PANORAMIC, ImageModality.PANORAMIC, ImageModality.BITEWING])
    dets = []
    arches = rng.choice([("upper", "lower"), ("upper",), ("lower",), ()])
    per_side = rng.randint(3, 9)
    tw = (width / 2) / (per_side + 1)
    for arch in arches:
        y1, y2 = (height * 0.12, height * 0.47) if arch == "upper" else (height * 0.53, height * 0.88)
        for i in range(per_side):
            for side in (-1, 1):
                if rng.random() < 0.1:
                    continue  # missing tooth
                jitter = rng.uniform(-tw * 0.1, tw * 0.1)
                x1 = width / 2 - 5 - (i + 1) * tw + 10 + jitter if side < 0 else width / 2 + 5 + i * tw + jitter
                box = [x1, y1 + rng.uniform(-8, 8), x1 + tw - 10, y2 + rng.uniform(-8, 8)]
                dets.append(("tooth", round(rng.uniform(0.2, 0.99), 4), box))
                if rng.random() < 0.15:  # duplicate detection of the same tooth
                    dets.append(("tooth", round(rng.uniform(0.2, 0.99), 4), [v + rng.uniform(-3, 3) for v in box]))
    for _ in range(rng.randint(0, 12)):
        kind = rng.choice([t.value for t in FindingType if t.value != "tooth"])
        x = rng.uniform(0, width - 60)
        y = rng.uniform(0, height - 60)
        w, h = rng.uniform(5, 60), rng.uniform(5, 60)
        dets.append((kind, round(rng.uniform(0.05, 0.99), 4), [x, y, x + w, y + h]))
    rng.shuffle(dets)
    teeth, findings = build_findings(
        [Detection(type=t, confidence=c, box=BoundingBox(x1=b[0], y1=b[1], x2=b[2], y2=b[3])) for t, c, b in dets],
        width,
        height,
        modality,
        0.25,
        0.6,
    )
    return {
        "width": width,
        "height": height,
        "modality": modality.value,
        "detections": [{"type": t, "confidence": c, "box": dict(zip(("x1", "y1", "x2", "y2"), b))} for t, c, b in dets],
        "teeth": [{"fdi": t.fdi, "confidence": t.confidence, "box": t.box.model_dump()} for t in teeth],
        "findings": [
            {"id": f.id, "type": f.type.value, "confidence": f.confidence, "tooth": f.tooth, "needsReview": f.needs_review}
            for f in findings
        ],
    }


def main() -> None:
    rng = random.Random(1234)
    scenes = [scene(rng) for _ in range(200)]
    hints = ["OPG", "Panoramic X-ray", "BW left", "IOPA 46", "Optional spa notes", "", "pa", "DPT"]
    shapes = [(2000, 1000), (1300, 1000), (800, 1000), (1000, 1000), (1699, 1000)]
    modalities = [{"width": w, "height": h, "hint": hint, "expected": guess_modality(w, h, hint).value} for hint in hints for w, h in shapes]
    OUT.write_text(json.dumps({"scenes": scenes, "modalities": modalities}) + "\n")
    print(f"wrote {len(scenes)} scenes, {len(modalities)} modality cases")


if __name__ == "__main__":
    main()
