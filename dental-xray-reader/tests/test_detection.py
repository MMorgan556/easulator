from types import SimpleNamespace

import numpy as np

from app.detection import DemoDetector, YoloDetector
from app.schemas import FindingType


class FakeTensor(list):
    def tolist(self):
        return list(self)


class FakeYolo:
    """Stands in for ultralytics.YOLO so the adapter is tested without torch."""

    names = {0: "tooth", 1: "caries", 2: "not_a_dental_class"}

    def __init__(self):
        self.calls = []

    def predict(self, image, **kwargs):
        self.calls.append((image, kwargs))
        boxes = SimpleNamespace(
            xyxy=FakeTensor([[10, 20, 60, 90], [-5, -5, 30, 500], [1, 1, 2, 2]]),
            conf=FakeTensor([0.9, 0.7, 0.99]),
            cls=FakeTensor([0.0, 1.0, 2.0]),
        )
        return [SimpleNamespace(boxes=boxes)]


def test_yolo_adapter_passes_ingested_image_unchanged():
    model = FakeYolo()
    detector = YoloDetector("unused.pt", input_size=1024, min_confidence=0.25, model=model)
    pixels = np.random.default_rng(0).integers(0, 256, (300, 400), dtype=np.uint8)

    detections = detector.detect(pixels)

    image, kwargs = model.calls[0]
    assert image.shape == (300, 400, 3)
    assert all(np.array_equal(image[..., c], pixels) for c in range(3))
    assert kwargs == {"imgsz": 1024, "conf": 0.25, "verbose": False}
    # Unknown class dropped; boxes clipped to the image.
    assert [d.type for d in detections] == [FindingType.TOOTH, FindingType.CARIES]
    assert detections[1].box.model_dump() == {"x1": 0.0, "y1": 0.0, "x2": 30.0, "y2": 300.0}


def test_demo_detector_scales_to_image():
    detections = DemoDetector().detect(np.zeros((500, 1000), dtype=np.uint8))
    assert sum(d.type == FindingType.TOOTH for d in detections) == 16
    assert all(0 <= d.box.x1 < d.box.x2 <= 1000 and 0 <= d.box.y1 < d.box.y2 <= 500 for d in detections)
