import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { join } from "node:path";
import * as ort from "onnxruntime-web";
import { decodeYolo, letterboxGeometry, preprocess } from "../js/ai.js";
import { FINDING_TYPES } from "../js/findings.js";
import { fixtures } from "./helpers.mjs";

test("letterbox geometry matches Ultralytics for a 2:1 image", () => {
  // 128x64 into 64: r=0.5, 64x32 content, 16 px padding top and bottom.
  assert.deepEqual(letterboxGeometry(128, 64, 64), { r: 0.5, newW: 64, newH: 32, left: 0, top: 16, padX: 0, padY: 16 });
});

test("preprocess pads with gray 114 and keeps 3 identical channels", () => {
  const display = new Uint8Array(128 * 64).fill(255);
  const { tensor } = preprocess(display, 128, 64, 64);
  const plane = 64 * 64;
  assert.ok(Math.abs(tensor[0] - 114 / 255) < 1e-7, "padding");
  assert.equal(tensor[32 * 64 + 10], 1, "content");
  assert.equal(tensor[plane + 32 * 64 + 10], tensor[32 * 64 + 10]);
  assert.equal(tensor[2 * plane + 32 * 64 + 10], tensor[32 * 64 + 10]);
});

test("stand-in YOLO model runs through onnxruntime-web and decodes to image pixels", async () => {
  ort.env.wasm.numThreads = 1;
  const session = await ort.InferenceSession.create(new Uint8Array(readFileSync(join(fixtures, "test_model.onnx"))));
  const display = new Uint8Array(128 * 64).fill(100);
  const { tensor, geometry } = preprocess(display, 128, 64, 64);
  const out = await session.run({ images: new ort.Tensor("float32", tensor, [1, 3, 64, 64]) });
  const output = out.output0;
  const detections = decodeYolo(output.data, output.dims, { classes: FINDING_TYPES, geometry, width: 128, height: 64 });

  assert.deepEqual(
    detections.map((d) => [d.type, Math.round(d.confidence * 100) / 100]),
    [["tooth", 0.9], ["caries", 0.8], ["implant", 0.6]],
  );
  // Input box cx=32, cy=32, 16x8 -> x 24..40, y 28..36; minus 16 px top padding, / 0.5.
  assert.deepEqual(detections[0].box, { x1: 48, y1: 24, x2: 80, y2: 40 });
});

test("channels-last and end-to-end output layouts decode the same boxes", () => {
  const geometry = letterboxGeometry(64, 64, 64);
  const classes = ["tooth", "caries"];
  const opts = { classes, geometry, width: 64, height: 64 };
  // channels-last [1, N, 4+nc]: one tooth at cx=20, cy=20, 10x10
  const last = decodeYolo(new Float32Array([20, 20, 10, 10, 0.9, 0.1]), [1, 1, 6], { ...opts, classes: ["tooth", "caries"] });
  assert.deepEqual(last[0].box, { x1: 15, y1: 15, x2: 25, y2: 25 });
  // end-to-end [1, N, 6] with 10 classes: x1,y1,x2,y2,score,class
  const e2e = decodeYolo(new Float32Array([15, 15, 25, 25, 0.9, 1]), [1, 1, 6], { ...opts, classes: FINDING_TYPES });
  assert.deepEqual(e2e.map((d) => d.type), ["caries"]);
  assert.throws(() => decodeYolo(new Float32Array(10), [1, 10, 1], opts), /does not match/);
});
