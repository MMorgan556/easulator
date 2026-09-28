// Pixel mapping shared by the model input and the viewer.
//
// These functions mirror app/ingest.py so the browser feeds an AI model exactly what the
// Python tooling (scripts/prepare_dataset.py) produced for training: the same percentile
// stretch, DICOM linear window and float32 rounding. test/parity.test.mjs checks them
// against images exported by the Python code.

const f32 = Math.fround;

/** Round half to even, like numpy.rint. */
export function rint(x) {
  const r = Math.round(x);
  const even = Math.abs(x % 1) === 0.5 && r % 2 !== 0 ? r - 1 : r;
  return even === 0 ? 0 : even; // never -0
}

/** numpy.percentile (linear method) of already-sorted values. */
export function percentileSorted(sorted, p) {
  const n = sorted.length;
  const q = p / 100;
  const virtual = n * q - q; // numpy's virtual index for the linear method
  const lower = Math.floor(virtual);
  const gamma = virtual - lower;
  const a = sorted[Math.min(Math.max(lower, 0), n - 1)];
  const b = sorted[Math.min(Math.max(lower + 1, 0), n - 1)];
  const diff = b - a;
  // numpy's _lerp: interpolate from whichever end is closer.
  return gamma >= 0.5 ? b - diff * (1 - gamma) : a + diff * gamma;
}

/** Every 4th row and column, like arr[::4, ::4]. */
export function subsample4(values, width, height) {
  const cols = Math.ceil(width / 4);
  const rows = Math.ceil(height / 4);
  const out = new Float64Array(rows * cols);
  let k = 0;
  for (let y = 0; y < height; y += 4) {
    const row = y * width;
    for (let x = 0; x < width; x += 4) out[k++] = values[row + x];
  }
  return out;
}

/**
 * Map values linearly to 0-255 as ``(x - low) * scale`` with float32 steps, clipped and
 * rounded half to even (numpy float32 in-place arithmetic with weak Python scalars).
 */
export function mapLinear(values, low, scale, out = new Uint8Array(values.length)) {
  const lo = f32(low);
  const s = f32(scale);
  for (let i = 0; i < values.length; i++) {
    let v = f32(f32(f32(values[i]) - lo) * s);
    if (v <= 0) v = 0;
    else if (v >= 255) v = 255;
    out[i] = rint(v);
  }
  return out;
}

/** Robust percentile stretch to 0-255 (app.ingest.to_uint8). Returns the mapping used. */
export function stretchRange(values, width, height) {
  const sample = subsample4(values, width, height).sort();
  let lo = percentileSorted(sample, 0.5);
  let hi = percentileSorted(sample, 99.5);
  if (hi <= lo) {
    lo = Infinity;
    hi = -Infinity;
    for (let i = 0; i < values.length; i++) {
      const v = values[i];
      if (v < lo) lo = v;
      if (v > hi) hi = v;
    }
  }
  return { lo, hi };
}

export function toUint8(values, width, height) {
  const { lo, hi } = stretchRange(values, width, height);
  if (!(hi > lo)) return new Uint8Array(values.length);
  return mapLinear(values, lo, 255 / (hi - lo));
}

/** DICOM PS3.3 C.11.2.1.2 linear window straight to 0-255. */
export function windowToUint8(values, center, width) {
  const low = center - 0.5 - (width - 1) / 2;
  return mapLinear(values, low, 255 / (width - 1));
}

/** Window for interactive display: centre/width in the image's value units. */
export function windowFromRange(lo, hi) {
  const width = hi - lo + 1;
  return { center: lo + 0.5 + (width - 1) / 2, width };
}

/** RGBA bytes to 8-bit luma with Pillow's "L" conversion (ITU-R 601-2, fixed point). */
export function rgbaToLuma(rgba) {
  const n = rgba.length >> 2;
  const out = new Uint8Array(n);
  for (let i = 0, j = 0; i < n; i++, j += 4) {
    out[i] = (rgba[j] * 19595 + rgba[j + 1] * 38470 + rgba[j + 2] * 7471 + 0x8000) >> 16;
  }
  return out;
}

export function invert(u8, out = new Uint8Array(u8.length)) {
  for (let i = 0; i < u8.length; i++) out[i] = 255 - u8[i];
  return out;
}

/**
 * Contrast-limited adaptive histogram equalization, for display only (the model never
 * sees it). Tiles are interpolated bilinearly, as in OpenCV's CLAHE.
 */
export function clahe(u8, width, height, { tiles = 8, clipLimit = 2.0 } = {}) {
  const tilesX = Math.max(1, Math.min(tiles, width));
  const tilesY = Math.max(1, Math.min(tiles, height));
  const tileW = width / tilesX;
  const tileH = height / tilesY;
  const luts = new Array(tilesX * tilesY);
  const hist = new Uint32Array(256);

  for (let ty = 0; ty < tilesY; ty++) {
    for (let tx = 0; tx < tilesX; tx++) {
      hist.fill(0);
      const x0 = Math.floor(tx * tileW), x1 = Math.floor((tx + 1) * tileW);
      const y0 = Math.floor(ty * tileH), y1 = Math.floor((ty + 1) * tileH);
      for (let y = y0; y < y1; y++) {
        const row = y * width;
        for (let x = x0; x < x1; x++) hist[u8[row + x]]++;
      }
      const count = Math.max(1, (x1 - x0) * (y1 - y0));
      const limit = Math.max(1, Math.floor((clipLimit * count) / 256));
      let excess = 0;
      for (let i = 0; i < 256; i++) {
        if (hist[i] > limit) {
          excess += hist[i] - limit;
          hist[i] = limit;
        }
      }
      const share = Math.floor(excess / 256);
      let remainder = excess - share * 256;
      for (let i = 0; i < 256; i++) {
        hist[i] += share;
        if (remainder > 0 && i % Math.max(1, Math.floor(256 / remainder)) === 0) {
          hist[i]++;
          remainder--;
        }
      }
      const lut = new Uint8Array(256);
      let cdf = 0;
      for (let i = 0; i < 256; i++) {
        cdf += hist[i];
        lut[i] = Math.min(255, Math.round((cdf * 255) / count));
      }
      luts[ty * tilesX + tx] = lut;
    }
  }

  const out = new Uint8Array(u8.length);
  for (let y = 0; y < height; y++) {
    const gy = (y + 0.5) / tileH - 0.5;
    const ty0 = Math.max(0, Math.min(tilesY - 1, Math.floor(gy)));
    const ty1 = Math.min(tilesY - 1, ty0 + 1);
    const fy = Math.max(0, Math.min(1, gy - ty0));
    for (let x = 0; x < width; x++) {
      const gx = (x + 0.5) / tileW - 0.5;
      const tx0 = Math.max(0, Math.min(tilesX - 1, Math.floor(gx)));
      const tx1 = Math.min(tilesX - 1, tx0 + 1);
      const fx = Math.max(0, Math.min(1, gx - tx0));
      const v = u8[y * width + x];
      const top = luts[ty0 * tilesX + tx0][v] * (1 - fx) + luts[ty0 * tilesX + tx1][v] * fx;
      const bottom = luts[ty1 * tilesX + tx0][v] * (1 - fx) + luts[ty1 * tilesX + tx1][v] * fx;
      out[y * width + x] = Math.round(top * (1 - fy) + bottom * fy);
    }
  }
  return out;
}

export function histogram(u8) {
  const hist = new Uint32Array(256);
  for (let i = 0; i < u8.length; i++) hist[u8[i]]++;
  return hist;
}

/**
 * Image from browser-decoded RGBA (PNG/JPEG/WebP/BMP): Pillow luma then the percentile
 * stretch, like app.ingest._load_raster for 8-bit input.
 */
export function rasterToImage(rgba, width, height) {
  return grayToImage(rgbaToLuma(rgba), width, height);
}

/** Grayscale samples (8- or 16-bit) to an image with the percentile-stretched display. */
export function grayToImage(values, width, height) {
  const { lo, hi } = stretchRange(values, width, height);
  const display = hi > lo ? mapLinear(values, lo, 255 / (hi - lo)) : new Uint8Array(values.length);
  return {
    width,
    height,
    values,
    display,
    defaultWindow: hi > lo ? windowFromRange(lo, hi) : { center: lo, width: 2 },
    inverted: false,
    format: "image",
    compression: null,
    pixelSpacing: null,
    hint: "",
  };
}
