"""A stand-in ONNX model with an Ultralytics YOLO export's interface, for testing inference.

Input "images" [1, 3, 64, 64]; output "output0" [1, 4 + 10, 5]: fixed predictions (plus
0 x mean(input), so the graph really consumes the input). Run from dental-xray-reader/.
"""

from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

OUT = Path(__file__).resolve().parent / "test_model.onnx"
CLASSES = 10
# anchors: cx, cy, w, h then per-class scores (tooth=0, caries=1, implant=6)
anchors = [
    ((32, 32, 16, 8), {0: 0.9}),   # tooth
    ((32, 32, 16, 8), {1: 0.8}),   # caries at the same place (other class: kept)
    ((33, 32, 16, 8), {0: 0.85}),  # duplicate tooth, IoU > 0.7: suppressed
    ((10, 20, 4, 4), {0: 0.1}),    # below confidence: dropped
    ((50, 40, 10, 6), {6: 0.6}),   # implant
]
pred = np.zeros((1, 4 + CLASSES, len(anchors)), dtype=np.float32)
for i, (box, scores) in enumerate(anchors):
    pred[0, :4, i] = box
    for c, s in scores.items():
        pred[0, 4 + c, i] = s

graph = helper.make_graph(
    [
        helper.make_node("ReduceMean", ["images"], ["mean"], keepdims=0),
        helper.make_node("Mul", ["mean", "zero"], ["nothing"]),
        helper.make_node("Add", ["pred", "nothing"], ["output0"]),
    ],
    "stand_in_yolo",
    [helper.make_tensor_value_info("images", TensorProto.FLOAT, [1, 3, 64, 64])],
    [helper.make_tensor_value_info("output0", TensorProto.FLOAT, [1, 4 + CLASSES, len(anchors)])],
    [numpy_helper.from_array(pred, "pred"), numpy_helper.from_array(np.array(0, dtype=np.float32), "zero")],
)
model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
model.ir_version = 8
onnx.checker.check_model(model)
OUT.write_bytes(model.SerializeToString())
print(f"wrote {OUT}")
