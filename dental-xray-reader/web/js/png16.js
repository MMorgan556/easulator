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

async function inflate(parts) {
  const stream = new Blob(parts).stream().pipeThrough(new DecompressionStream("deflate"));
  return new Uint8Array(await new Response(stream).arrayBuffer());
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

/**
 * Decode a 16-bit grayscale PNG to { width, height, values: Uint16Array }, or null if the
 * file is another kind of PNG.
 */
export async function decodePng16(bytes) {
  if (bytes.length < 8 || SIGNATURE.some((v, i) => bytes[i] !== v)) return null;
  const { header, idat } = readChunks(bytes);
  if (header.bitDepth !== 16 || header.colorType !== 0) return null;
  const { width, height } = header;
  const data = await inflate(idat);
  const values = new Uint16Array(width * height);
  const passes = header.interlace === 1 ? ADAM7 : [[0, 0, 1, 1]];
  let offset = 0;
  for (const [x0, y0, dx, dy] of passes) {
    const w = Math.ceil((width - x0) / dx);
    const h = Math.ceil((height - y0) / dy);
    if (w <= 0 || h <= 0) continue;
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
