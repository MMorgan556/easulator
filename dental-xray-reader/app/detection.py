"""Detection backends. Each returns boxes in original-image pixel coordinates."""

from __future__ import annotations

from typing import Protocol

import numpy as np

from .config import Settings
from .preprocess import prepare
from .schemas import BoundingBox, Detection, FindingType


class Detector(Protocol):
    name: str

    def detect(self, pixels: np.ndarray) -> list[Detection]: ...


class DemoDetector:
    """Deterministic placeholder output so the full pipeline runs without trained weights.

    It does NOT analyze the image. Every result it produces is fake and the API
    marks it as such.
    """

    name = "demo"

    def detect(self, pixels: np.ndarray) -> list[Detection]:
        h, w = pixels.shape
        detections: list[Detection] = []
        teeth_per_arch = 8
        tooth_w = w / (teeth_per_arch + 1)
        for arch_top, arch_bottom in ((0.12, 0.48), (0.52, 0.88)):
            for i in range(teeth_per_arch):
                x1 = tooth_w * (i + 0.5)
                detections.append(
                    Detection(
                        type=FindingType.TOOTH,
                        confidence=0.95,
                        box=BoundingBox(x1=x1, y1=h * arch_top, x2=x1 + tooth_w * 0.9, y2=h * arch_bottom),
                    )
                )
        # One confident caries on the first upper tooth, one uncertain lesion on the last lower tooth.
        first, last = detections[0].box, detections[-1].box
        detections.append(
            Detection(
                type=FindingType.CARIES,
                confidence=0.82,
                box=BoundingBox(x1=first.x1 + 5, y1=first.y2 - 40, x2=first.x1 + 35, y2=first.y2 - 10),
            )
        )
        detections.append(
            Detection(
                type=FindingType.PERIAPICAL_LESION,
                confidence=0.41,
                box=BoundingBox(x1=last.x1 + 10, y1=last.y2 - 30, x2=last.x2 - 10, y2=last.y2 - 2),
            )
        )
        return detections


class YoloDetector:
    """Ultralytics YOLO (v8/v11) model trained on the classes in ``FindingType``.

    Class names in the weights must match ``FindingType`` values; unknown classes are ignored.
    """

    name = "yolo"

    def __init__(self, weights: str, input_size: int, min_confidence: float):
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError("DETECTOR_BACKEND=yolo requires: pip install -r requirements-ml.txt") from exc
        self.model = YOLO(weights)
        self.input_size = input_size
        self.min_confidence = min_confidence
        valid = {t.value for t in FindingType}
        self.class_map = {idx: FindingType(n) for idx, n in self.model.names.items() if n in valid}

    def detect(self, pixels: np.ndarray) -> list[Detection]:
        model_input, lb = prepare(pixels, self.input_size)
        rgb = np.repeat(model_input[..., None], 3, axis=-1)
        result = self.model.predict(rgb, imgsz=self.input_size, conf=self.min_confidence, verbose=False)[0]
        h, w = pixels.shape
        detections: list[Detection] = []
        for xyxy, conf, cls in zip(
            result.boxes.xyxy.tolist(), result.boxes.conf.tolist(), result.boxes.cls.tolist()
        ):
            finding_type = self.class_map.get(int(cls))
            if finding_type is None:
                continue
            x1, y1 = lb.to_original(xyxy[0], xyxy[1])
            x2, y2 = lb.to_original(xyxy[2], xyxy[3])
            detections.append(
                Detection(
                    type=finding_type,
                    confidence=float(conf),
                    box=BoundingBox(
                        x1=float(np.clip(x1, 0, w)),
                        y1=float(np.clip(y1, 0, h)),
                        x2=float(np.clip(x2, 0, w)),
                        y2=float(np.clip(y2, 0, h)),
                    ),
                )
            )
        return detections


def build_detector(settings: Settings) -> Detector:
    if settings.detector_backend == "demo":
        return DemoDetector()
    if settings.detector_backend == "yolo":
        return YoloDetector(settings.yolo_weights, settings.model_input_size, settings.min_confidence)
    raise ValueError(f"Unknown DETECTOR_BACKEND {settings.detector_backend!r} (expected 'demo' or 'yolo')")
