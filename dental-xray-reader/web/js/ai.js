// AI detection with an Ultralytics YOLO model exported to ONNX (scripts/train_yolo.py
// --export-web). Pre- and post-processing reproduce Ultralytics' predict(): letterbox with
// gray (114) padding, class-aware NMS at IoU 0.7, and boxes scaled back to the image.
// Inference runs in a Web Worker (ai-worker.js) so the page stays responsive.

import { rint } from "./imaging.js";

export const PAD_VALUE = 114;

/** Letterbox geometry, as ultralytics.data.augment.LetterBox(auto=False, center=True). */
export function letterboxGeometry(width, height, size) {
  const r = Math.min(size / height, size / width);
  const newW = rint(width * r);
  const newH = rint(height * r);
  const dw = (size - newW) / 2;
  const dh = (size - newH) / 2;
  return {
    r,
    newW,
    newH,
    left: rint(dw - 0.1),
    top: rint(dh - 0.1),
    // ultralytics.utils.ops.scale_boxes derives the padding from the unrounded size.
    padX: rint((size - width * r) / 2 - 0.1),
    padY: rint((size - height * r) / 2 - 0.1),
  };
}

/** Bilinear resize of an 8-bit image with half-pixel centres (cv2.INTER_LINEAR). */
export function resizeBilinear(src, width, height, newW, newH) {
  const out = new Uint8Array(newW * newH);
  const sx = width / newW;
  const sy = height / newH;
  for (let y = 0; y < newH; y++) {
    let fy = (y + 0.5) * sy - 0.5;
    if (fy < 0) fy = 0;
    const y0 = Math.min(height - 1, Math.floor(fy));
    const y1 = Math.min(height - 1, y0 + 1);
    const wy = fy - y0;
    for (let x = 0; x < newW; x++) {
      let fx = (x + 0.5) * sx - 0.5;
      if (fx < 0) fx = 0;
      const x0 = Math.min(width - 1, Math.floor(fx));
      const x1 = Math.min(width - 1, x0 + 1);
      const wx = fx - x0;
      const top = src[y0 * width + x0] * (1 - wx) + src[y0 * width + x1] * wx;
      const bottom = src[y1 * width + x0] * (1 - wx) + src[y1 * width + x1] * wx;
      out[y * newW + x] = Math.round(top * (1 - wy) + bottom * wy);
    }
  }
  return out;
}

/** Model input: grayscale in 3 identical channels, NCHW float32 scaled to 0-1. */
export function preprocess(display, width, height, size) {
  const geo = letterboxGeometry(width, height, size);
  const resized = resizeBilinear(display, width, height, geo.newW, geo.newH);
  const plane = size * size;
  const tensor = new Float32Array(3 * plane).fill(PAD_VALUE / 255);
  for (let y = 0; y < geo.newH; y++) {
    const dst = (y + geo.top) * size + geo.left;
    for (let x = 0; x < geo.newW; x++) {
      const v = resized[y * geo.newW + x] / 255;
      tensor[dst + x] = v;
      tensor[plane + dst + x] = v;
      tensor[2 * plane + dst + x] = v;
    }
  }
  return { tensor, geometry: geo };
}

function boxIou(a, b) {
  const w = Math.min(a[2], b[2]) - Math.max(a[0], b[0]);
  const h = Math.min(a[3], b[3]) - Math.max(a[1], b[1]);
  const inter = Math.max(0, w) * Math.max(0, h);
  const union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter;
  return union > 0 ? inter / union : 0;
}

/**
 * Decode raw YOLO output into detections in original-image pixels.
 * Supports the standard export ([1, 4+nc, N] or [1, N, 4+nc], boxes as cx,cy,w,h) and the
 * end-to-end export ([1, N, 6] as x1,y1,x2,y2,score,class, already NMS-ed).
 */
export function decodeYolo(data, dims, { classes, geometry, width, height, confidence = 0.25, iouThreshold = 0.7, maxDetections = 300 }) {
  const nc = classes.length;
  const candidates = [];
  if (dims.length === 3 && dims[2] === 6 && dims[1] !== 4 + nc && nc + 4 !== 6) {
    for (let i = 0; i < dims[1]; i++) {
      const o = i * 6;
      const score = data[o + 4];
      if (score > confidence) candidates.push({ box: [data[o], data[o + 1], data[o + 2], data[o + 3]], score, cls: Math.round(data[o + 5]) });
    }
  } else {
    let n;
    let at;
    if (dims[1] === 4 + nc) {
      n = dims[2];
      at = (anchor, channel) => data[channel * n + anchor];
    } else if (dims[2] === 4 + nc) {
      n = dims[1];
      at = (anchor, channel) => data[anchor * (4 + nc) + channel];
    } else {
      throw new Error(`Model output shape [${dims}] does not match ${nc} classes`);
    }
    for (let a = 0; a < n; a++) {
      let best = -1;
      let score = -Infinity;
      for (let c = 0; c < nc; c++) {
        const s = at(a, 4 + c);
        if (s > score) {
          score = s;
          best = c;
        }
      }
      if (!(score > confidence)) continue;
      const cx = at(a, 0), cy = at(a, 1), w = at(a, 2), h = at(a, 3);
      candidates.push({ box: [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], score, cls: best });
    }
    // Class-aware NMS (torchvision.ops.nms semantics: drop boxes with IoU above the threshold).
    candidates.sort((p, q) => q.score - p.score);
    const kept = [];
    for (const c of candidates) {
      if (kept.length >= maxDetections) break;
      if (kept.every((k) => k.cls !== c.cls || boxIou(k.box, c.box) <= iouThreshold)) kept.push(c);
    }
    candidates.length = 0;
    candidates.push(...kept);
  }

  const { r, padX: left, padY: top } = geometry;
  const clip = (v, max) => Math.min(Math.max(v, 0), max);
  return candidates
    .filter((c) => c.cls >= 0 && c.cls < nc)
    .map((c) => ({
      type: classes[c.cls],
      confidence: c.score,
      box: {
        x1: clip((c.box[0] - left) / r, width),
        y1: clip((c.box[1] - top) / r, height),
        x2: clip((c.box[2] - left) / r, width),
        y2: clip((c.box[3] - top) / r, height),
      },
    }));
}

/** Read model/model.json; null when no model is installed. */
export async function loadManifest(baseUrl) {
  let resp;
  try {
    resp = await fetch(new URL("model.json", baseUrl), { cache: "no-cache" });
  } catch {
    return null;
  }
  if (!resp.ok) return null;
  const manifest = await resp.json();
  if (!manifest.available) return null;
  if (!Array.isArray(manifest.classes) || !manifest.file || !Number.isInteger(manifest.input_size)) {
    throw new Error("model/model.json is missing classes, file or input_size");
  }
  return { ...manifest, url: new URL(manifest.file, baseUrl).href };
}

/** Runs the model in a worker. One analysis at a time. */
export class Detector {
  constructor(manifest, workerUrl) {
    this.manifest = manifest;
    this.worker = new Worker(workerUrl, { type: "module" });
    this.pending = new Map();
    this.nextId = 1;
    this.failed = null;
    this.worker.onmessage = ({ data }) => {
      const entry = this.pending.get(data.id);
      if (!entry) return;
      this.pending.delete(data.id);
      if (data.error) entry.reject(new Error(data.error));
      else entry.resolve(data);
    };
    // A worker that fails to load or crashes never answers: fail everything waiting on it.
    const fail = (reason) => {
      this.failed = new Error(reason);
      for (const entry of this.pending.values()) entry.reject(this.failed);
      this.pending.clear();
    };
    this.worker.onerror = (e) => {
      e.preventDefault?.();
      fail(`The AI engine could not start${e.message ? ` (${e.message})` : ""}`);
    };
    this.worker.onmessageerror = () => fail("The AI engine sent an unreadable message");
    this.ready = this.call({ type: "load", url: manifest.url });
    this.ready.catch((err) => {
      this.failed ||= err; // without a loaded model the engine is unusable: start over next time
    });
  }

  terminate() {
    this.worker.terminate();
    for (const entry of this.pending.values()) entry.reject(new Error("AI engine stopped"));
    this.pending.clear();
  }

  call(message, transfer = []) {
    if (this.failed) return Promise.reject(this.failed);
    const id = this.nextId++;
    return new Promise((resolve, reject) => {
      this.pending.set(id, { resolve, reject });
      this.worker.postMessage({ ...message, id }, transfer);
    });
  }

  async detect(image) {
    await this.ready;
    const size = this.manifest.input_size;
    const { tensor, geometry } = preprocess(image.display, image.width, image.height, size);
    const started = performance.now();
    const out = await this.call({ type: "run", input: tensor, size }, [tensor.buffer]);
    const detections = decodeYolo(out.data, out.dims, {
      classes: this.manifest.classes,
      geometry,
      width: image.width,
      height: image.height,
      confidence: this.manifest.confidence ?? 0.25,
    });
    return { detections, milliseconds: performance.now() - started };
  }
}
