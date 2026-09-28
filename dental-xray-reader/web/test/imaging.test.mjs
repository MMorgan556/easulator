import { test } from "node:test";
import assert from "node:assert/strict";
import { clahe, histogram, percentileSorted, rint, rgbaToLuma, toUint8, windowToUint8 } from "../js/imaging.js";

test("rint rounds half to even like numpy", () => {
  assert.deepEqual([0.5, 1.5, 2.5, 3.5, 127.5, 128.4, 128.6].map(rint), [0, 2, 2, 4, 128, 128, 129]);
});

test("percentile matches numpy's linear method", () => {
  // np.percentile([1, 2, 3, 4, 10], [0.5, 50, 99.5]) -> [1.02, 3.0, 9.88]
  const s = [1, 2, 3, 4, 10];
  assert.ok(Math.abs(percentileSorted(s, 0.5) - 1.02) < 1e-12);
  assert.equal(percentileSorted(s, 50), 3);
  assert.ok(Math.abs(percentileSorted(s, 99.5) - 9.88) < 1e-12);
});

test("flat image stretches to black", () => {
  assert.equal(toUint8(new Float32Array(64).fill(7), 8, 8).reduce((a, b) => a + b, 0), 0);
});

test("DICOM linear window endpoints", () => {
  // c=500, w=1001: values <= 0 -> 0, >= 1000 -> 255, 500 -> 128 (127.5 rounds to even)
  assert.deepEqual(Array.from(windowToUint8(new Float32Array([-5, 0, 500, 1000, 2000]), 500, 1001)), [0, 0, 128, 255, 255]);
});

test("luma uses Pillow's fixed-point weights", () => {
  // Pillow: Image.new("RGB", (1, 1), (200, 100, 50)).convert("L") -> 124
  assert.equal(rgbaToLuma(new Uint8Array([200, 100, 50, 255]))[0], 124);
});

test("CLAHE with a high clip limit spreads a low-contrast image", () => {
  const w = 64, h = 64;
  const img = new Uint8Array(w * h);
  for (let i = 0; i < img.length; i++) img[i] = 100 + ((i % w) >> 3); // 100..107
  const out = clahe(img, w, h, { tiles: 1, clipLimit: 40 });
  const hist = histogram(out);
  const used = hist.reduce((n, v) => n + (v > 0 ? 1 : 0), 0);
  assert.ok(Math.max(...out) - Math.min(...out) > 30, "contrast increased");
  assert.ok(used >= 8, "levels preserved");
});

test("CLAHE keeps a uniform image uniform", () => {
  const out = clahe(new Uint8Array(32 * 32).fill(90), 32, 32);
  assert.equal(new Set(out).size, 1);
});

test("CLAHE output is monotonic in input within a tile region", () => {
  const w = 16, h = 16;
  const img = new Uint8Array(w * h);
  for (let i = 0; i < img.length; i++) img[i] = (i * 7) % 256;
  const out = clahe(img, w, h, { tiles: 1 });
  const pairs = Array.from(img, (v, i) => [v, out[i]]).sort((a, b) => a[0] - b[0]);
  for (let i = 1; i < pairs.length; i++) assert.ok(pairs[i][1] >= pairs[i - 1][1]);
});
