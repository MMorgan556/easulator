// Decoder for 16-bit grayscale PNGs. Browsers reduce every PNG to 8 bits per channel, which
// would posterize sensor images that use a narrow part of the 16-bit range; Pillow (and so
// the Python tooling) keeps all 16 bits. Other PNGs return null and use the browser decoder.

const SIGNATURE = [137, 80, 78, 71, 13, 10, 26, 10];
const ADAM7 = [
  [0, 0, 8, 8],
  [4, 0, 8, 8],
  [0, 4, 4, 8],
  [2, 0, 4, 4],
  [0, 2, 2, 4],
  [1, 0, 2, 2],
  [0, 1, 1, 2],
];

export class PngError extends Error {}

function readChunks(bytes) {
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  let offset = 8;
  let header = null;
  const idat = [];
  while (offset + 8 <= bytes.length) {
    const length = view.getUint32(offset);
    const type = String.fromCharCode(...bytes.subarray(offset + 4, offset + 8));
    const start = offset + 8;
    if (start + length + 4 > bytes.length) throw new PngError("PNG file is truncated");
    if (type === "IHDR") {
      header = {
        width: view.getUint32(start),
        height: view.getUint32(start + 4),
        bitDepth: bytes[start + 8],
        colorType: bytes[start + 9],
        interlace: bytes[start + 12],
      };
    } else if (type === "IDAT") {
      idat.push(bytes.subarray(start, start + length));
    } else if (type === "IEND") {
      break;
    }
    offset = start + length + 4;
  }
  if (!header) throw new PngError("PNG file has no header");
  return { header, idat };
}

/** zlib-inflate the IDAT data, refusing to produce more than `limit` bytes (zip bombs). */
async function inflate(parts, limit) {
  const reader = new Blob(parts).stream().pipeThrough(new DecompressionStream("deflate")).getReader();
  const chunks = [];
  let total = 0;
  try {
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      total += value.length;
      if (total > limit) {
        await reader.cancel().catch(() => {});
        throw new PngError("image data is larger than its header says");
      }
      chunks.push(value);
    }
  } catch (err) {
    if (err instanceof PngError) throw err;
    throw new PngError("image data is corrupt or incomplete");
  }
  const out = new Uint8Array(total);
  let offset = 0;
  for (const c of chunks) {
    out.set(c, offset);
    offset += c.length;
  }
  return out;
}

function paeth(a, b, c) {
  const p = a + b - c;
  const pa = Math.abs(p - a);
  const pb = Math.abs(p - b);
  const pc = Math.abs(p - c);
  return pa <= pb && pa <= pc ? a : pb <= pc ? b : c;
}

/** Undo PNG row filters for one (sub)image; returns the filtered-free bytes. */
function unfilter(data, offset, width, height, bpp) {
  const stride = width * bpp;
  const out = new Uint8Array(stride * height);
  let pos = offset;
  for (let y = 0; y < height; y++) {
    if (pos + 1 + stride > data.length) throw new PngError("PNG image data is truncated");
    const filter = data[pos++];
    const row = y * stride;
    const prev = row - stride;
    for (let x = 0; x < stride; x++) {
      const raw = data[pos++];
      const a = x >= bpp ? out[row + x - bpp] : 0;
      const b = y > 0 ? out[prev + x] : 0;
      const c = x >= bpp && y > 0 ? out[prev + x - bpp] : 0;
      let v;
      switch (filter) {
        case 0: v = raw; break;
        case 1: v = raw + a; break;
        case 2: v = raw + b; break;
        case 3: v = raw + ((a + b) >> 1); break;
        case 4: v = raw + paeth(a, b, c); break;
        default: throw new PngError(`Invalid PNG filter type ${filter}`);
      }
      out[row + x] = v & 0xff;
    }
  }
  return { pixels: out, next: pos };
}

function passes(header) {
  const list = header.interlace === 1 ? ADAM7 : [[0, 0, 1, 1]];
  return list
    .map(([x0, y0, dx, dy]) => ({ x0, y0, dx, dy, w: Math.ceil((header.width - x0) / dx), h: Math.ceil((header.height - y0) / dy) }))
    .filter((p) => p.w > 0 && p.h > 0);
}

/**
 * Decode a 16-bit grayscale PNG to { width, height, values: Uint16Array }, or null if the
 * file is another kind of PNG. `maxPixels` is checked before any image data is expanded;
 * `onTooLarge(width, height)` may throw its own error instead of the PngError.
 */
export async function decodePng16(bytes, { maxPixels = Infinity, onTooLarge } = {}) {
  if (bytes.length < 8 || SIGNATURE.some((v, i) => bytes[i] !== v)) return null;
  const { header, idat } = readChunks(bytes);
  if (header.bitDepth !== 16 || header.colorType !== 0) return null;
  const { width, height } = header;
  if (!width || !height) throw new PngError("image has no pixels");
  if (width * height > maxPixels) {
    onTooLarge?.(width, height);
    throw new PngError(`image is too large (${width}x${height})`);
  }
  if (!idat.length) throw new PngError("image data is missing");
  const layout = passes(header);
  const expected = layout.reduce((sum, p) => sum + p.h * (1 + p.w * 2), 0);
  const data = await inflate(idat, expected);
  const values = new Uint16Array(width * height);
  let offset = 0;
  for (const { x0, y0, dx, dy, w, h } of layout) {
    const { pixels, next } = unfilter(data, offset, w, h, 2);
    offset = next;
    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) {
        const i = (y * w + x) * 2;
        values[(y0 + y * dy) * width + x0 + x * dx] = (pixels[i] << 8) | pixels[i + 1];
      }
    }
  }
  return { width, height, values };
}
