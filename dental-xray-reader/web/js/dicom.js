// DICOM decoding in the browser. Mirrors app/ingest.py: frame 0 only, linear rescale,
// DICOM linear window or percentile stretch, MONOCHROME1 inversion, 20 MP limit.
//
// Codecs are passed in so the same code runs in the browser (vendored wasm builds) and in
// Node tests. Each codec getter returns a promise.

import { mapLinear, toUint8, windowToUint8, invert, stretchRange } from "./imaging.js";

export const MAX_PIXELS = 20_000_000;

export class ImageError extends Error {}

const TS = {
  IMPLICIT_LE: "1.2.840.10008.1.2",
  EXPLICIT_LE: "1.2.840.10008.1.2.1",
  DEFLATED_LE: "1.2.840.10008.1.2.1.99",
  EXPLICIT_BE: "1.2.840.10008.1.2.2",
  JPEG_BASELINE: "1.2.840.10008.1.2.4.50",
  JPEG_EXTENDED: "1.2.840.10008.1.2.4.51",
  JPEG_LOSSLESS: "1.2.840.10008.1.2.4.57",
  JPEG_LOSSLESS_SV1: "1.2.840.10008.1.2.4.70",
  JPEG_LS_LOSSLESS: "1.2.840.10008.1.2.4.80",
  JPEG_LS_NEAR: "1.2.840.10008.1.2.4.81",
  J2K_LOSSLESS: "1.2.840.10008.1.2.4.90",
  J2K: "1.2.840.10008.1.2.4.91",
  RLE: "1.2.840.10008.1.2.5",
};

const TS_NAMES = {
  [TS.IMPLICIT_LE]: "Implicit VR Little Endian",
  [TS.EXPLICIT_LE]: "Explicit VR Little Endian",
  [TS.DEFLATED_LE]: "Deflated Explicit VR Little Endian",
  [TS.EXPLICIT_BE]: "Explicit VR Big Endian",
  [TS.JPEG_BASELINE]: "JPEG Baseline",
  [TS.JPEG_EXTENDED]: "JPEG Extended",
  [TS.JPEG_LOSSLESS]: "JPEG Lossless",
  [TS.JPEG_LOSSLESS_SV1]: "JPEG Lossless SV1",
  [TS.JPEG_LS_LOSSLESS]: "JPEG-LS Lossless",
  [TS.JPEG_LS_NEAR]: "JPEG-LS Near-lossless",
  [TS.J2K_LOSSLESS]: "JPEG 2000 Lossless",
  [TS.J2K]: "JPEG 2000",
  [TS.RLE]: "RLE Lossless",
};

export function isDicom(bytes) {
  return bytes.length > 132 && bytes[128] === 0x44 && bytes[129] === 0x49 && bytes[130] === 0x43 && bytes[131] === 0x4d;
}

function numbers(ds, tag) {
  const text = ds.string(tag);
  if (text === undefined) return [];
  return text
    .split("\\")
    .map((s) => Number.parseFloat(s))
    .filter((v) => Number.isFinite(v));
}

function firstNumber(ds, tag, fallback) {
  const values = numbers(ds, tag);
  return values.length ? values[0] : fallback;
}

/** Header fields needed for decoding and display. No patient data is read. */
export function readHeader(dicomParser, bytes) {
  let ds;
  try {
    ds = dicomParser.parseDicom(bytes, { untilTag: "x7fe00010" });
  } catch (err) {
    throw new ImageError(`This DICOM file is damaged or incomplete and could not be read (${err?.exception || err?.message || err})`);
  }
  const u16 = (tag, fallback) => {
    const el = ds.elements[tag];
    return el && el.length >= 2 ? ds.uint16(tag) : fallback;
  };
  const header = {
    transferSyntax: (ds.string("x00020010") || TS.IMPLICIT_LE).replace(/\0/g, "").trim(),
    rows: u16("x00280010", 0),
    columns: u16("x00280011", 0),
    samples: u16("x00280002", 1) || 1,
    planar: u16("x00280006", 0),
    bitsAllocated: u16("x00280100", 16),
    bitsStored: u16("x00280101", u16("x00280100", 16)),
    pixelRepresentation: u16("x00280103", 0),
    photometric: (ds.string("x00280004") || "MONOCHROME2").trim(),
    frames: Math.max(1, Math.trunc(firstNumber(ds, "x00280008", 1))),
    rescaleSlope: firstNumber(ds, "x00281053", 1) || 1,
    rescaleIntercept: firstNumber(ds, "x00281052", 0),
    windowCenter: numbers(ds, "x00281050"),
    windowWidth: numbers(ds, "x00281051"),
    voiFunction: (ds.string("x00281056") || "LINEAR").trim().toUpperCase(),
    hasVoiLutSequence: Boolean(ds.elements.x00283010),
    hasModalityLutSequence: Boolean(ds.elements.x00283000),
    pixelSpacing: null,
    hint: ["x0008103e", "x00081030", "x00180015"].map((t) => ds.string(t) || "").join(" "),
  };
  for (const tag of ["x00280030", "x00181164"]) {
    const spacing = numbers(ds, tag);
    if (spacing.length >= 2 && spacing[0] > 0 && spacing[1] > 0) {
      header.pixelSpacing = [spacing[0], spacing[1]];
      break;
    }
  }
  return header;
}

// ---------------------------------------------------------------- native and RLE data

function storedArray(bytes, offset, count, bitsAllocated, signed, bigEndian) {
  if (bitsAllocated === 8) {
    const view = bytes.subarray(offset, offset + count);
    return signed ? new Int8Array(view.buffer, view.byteOffset, count).slice() : view.slice();
  }
  if (bitsAllocated === 16) {
    const dv = new DataView(bytes.buffer, bytes.byteOffset + offset, count * 2);
    const out = signed ? new Int16Array(count) : new Uint16Array(count);
    for (let i = 0; i < count; i++) out[i] = signed ? dv.getInt16(i * 2, !bigEndian) : dv.getUint16(i * 2, !bigEndian);
    return out;
  }
  if (bitsAllocated === 32) {
    const dv = new DataView(bytes.buffer, bytes.byteOffset + offset, count * 4);
    const out = signed ? new Int32Array(count) : new Uint32Array(count);
    for (let i = 0; i < count; i++) out[i] = signed ? dv.getInt32(i * 4, !bigEndian) : dv.getUint32(i * 4, !bigEndian);
    return out;
  }
  throw new ImageError(`Unsupported Bits Allocated: ${bitsAllocated}`);
}

/** Keep only Bits Stored bits (the rest may hold overlays) and sign-extend. */
function maskBitsStored(arr, header) {
  const { bitsStored, bitsAllocated, pixelRepresentation } = header;
  if (bitsStored >= bitsAllocated || bitsStored <= 0) return arr;
  const mask = 2 ** bitsStored - 1;
  const signBit = 2 ** (bitsStored - 1);
  const out = pixelRepresentation ? new Int32Array(arr.length) : arr;
  for (let i = 0; i < arr.length; i++) {
    let v = arr[i] & mask;
    if (pixelRepresentation && v >= signBit) v -= 2 ** bitsStored;
    out[i] = v;
  }
  return out;
}

function packBits(src, start, end, out, outOffset, stride, limit) {
  // DICOM RLE segment (PS3.5 G.3.1): PackBits into every `stride`-th output byte.
  let i = start;
  let o = 0;
  while (i < end && o < limit) {
    const n = src[i++];
    if (n < 128) {
      for (let k = 0; k <= n && o < limit; k++) out[outOffset + o++ * stride] = src[i++];
    } else if (n > 128) {
      const value = src[i++];
      for (let k = 0; k < 257 - n && o < limit; k++) out[outOffset + o++ * stride] = value;
    }
    // n === 128 is a no-op
  }
  if (o < limit) throw new ImageError("RLE segment ended early; the file is damaged");
}

/** Decode one DICOM RLE frame to little-endian, sample-interleaved bytes. */
export function decodeRle(frame, header) {
  const dv = new DataView(frame.buffer, frame.byteOffset, frame.byteLength);
  const segments = dv.getUint32(0, true);
  const bytesPerSample = header.bitsAllocated / 8;
  const pixels = header.rows * header.columns;
  if (segments !== bytesPerSample * header.samples) {
    throw new ImageError(`RLE frame has ${segments} segments, expected ${bytesPerSample * header.samples}`);
  }
  const offsets = [];
  for (let s = 0; s < segments; s++) offsets.push(dv.getUint32(4 + s * 4, true));
  const out = new Uint8Array(pixels * header.samples * bytesPerSample);
  const stride = header.samples * bytesPerSample;
  for (let sample = 0; sample < header.samples; sample++) {
    for (let b = 0; b < bytesPerSample; b++) {
      const s = sample * bytesPerSample + b; // segments are most significant byte first
      const start = offsets[s];
      const end = s + 1 < segments ? offsets[s + 1] : frame.length;
      const byteInSample = bytesPerSample - 1 - b; // write little-endian
      packBits(frame, start, end, out, sample * bytesPerSample + byteInSample, stride, pixels);
    }
  }
  return out;
}

// ---------------------------------------------------------------- compressed codecs

async function decodeWasm(factoryPromise, className, frame) {
  const codec = await factoryPromise;
  const decoder = new codec[className]();
  try {
    decoder.getEncodedBuffer(frame.length).set(frame);
    decoder.decode();
    const info = decoder.getFrameInfo();
    const buffer = decoder.getDecodedBuffer().slice();
    return { info, buffer };
  } finally {
    decoder.delete();
  }
}

function typedFromDecoded(buffer, bitsPerSample, signed) {
  if (bitsPerSample <= 8) return signed ? new Int8Array(buffer.buffer, buffer.byteOffset, buffer.length) : buffer;
  const count = buffer.length / 2;
  return signed ? new Int16Array(buffer.buffer, buffer.byteOffset, count) : new Uint16Array(buffer.buffer, buffer.byteOffset, count);
}

async function decodeCompressed(header, frame, codecs) {
  const ts = header.transferSyntax;
  const signed = header.pixelRepresentation === 1;
  if (ts === TS.RLE) {
    const raw = decodeRle(frame, { ...header });
    return storedArray(raw, 0, header.rows * header.columns * header.samples, header.bitsAllocated, signed, false);
  }
  if (ts === TS.J2K_LOSSLESS || ts === TS.J2K) {
    const { info, buffer } = await decodeWasm(codecs.openjpeg(), "J2KDecoder", frame);
    return typedFromDecoded(buffer, info.bitsPerSample, info.isSigned || signed);
  }
  if (ts === TS.JPEG_LS_LOSSLESS || ts === TS.JPEG_LS_NEAR) {
    const { info, buffer } = await decodeWasm(codecs.charls(), "JpegLSDecoder", frame);
    return typedFromDecoded(buffer, info.bitsPerSample, signed);
  }
  if (ts === TS.JPEG_LOSSLESS || ts === TS.JPEG_LOSSLESS_SV1) {
    const Decoder = await codecs.jpegLossless();
    const decoded = new Decoder().decode(frame, 0, frame.length, header.bitsAllocated > 8 ? 2 : 1);
    if (header.bitsAllocated <= 8) return signed ? new Int8Array(decoded.buffer, decoded.byteOffset, decoded.length) : new Uint8Array(decoded.buffer, decoded.byteOffset, decoded.length);
    return signed ? new Int16Array(decoded.buffer, decoded.byteOffset, decoded.length) : new Uint16Array(decoded.buffer, decoded.byteOffset, decoded.length);
  }
  if (ts === TS.JPEG_BASELINE || ts === TS.JPEG_EXTENDED) {
    if (header.bitsStored > 8) throw new ImageError("12-bit JPEG Extended DICOM is not supported in the browser");
    const { rgba } = await codecs.jpegBaseline(frame);
    const n = header.rows * header.columns;
    if (rgba.length !== n * 4) throw new ImageError("JPEG frame size does not match the DICOM header");
    if (header.samples === 1) {
      const out = new Uint8Array(n);
      for (let i = 0; i < n; i++) out[i] = rgba[i * 4];
      return out;
    }
    const out = new Uint8Array(n * 3);
    for (let i = 0; i < n; i++) {
      out[i * 3] = rgba[i * 4];
      out[i * 3 + 1] = rgba[i * 4 + 1];
      out[i * 3 + 2] = rgba[i * 4 + 2];
    }
    header.photometric = "RGB"; // the browser returns RGB whatever the stored colour space
    return out;
  }
  throw new ImageError(`Unsupported DICOM compression: ${TS_NAMES[ts] || ts}`);
}

// ---------------------------------------------------------------- decode + display

function toGrayscale(stored, header) {
  const n = header.rows * header.columns;
  if (header.samples === 1) return stored;
  if (header.samples !== 3) throw new ImageError(`Unsupported Samples per Pixel: ${header.samples}`);
  const out = new Float32Array(n);
  const planar = header.planar === 1 && !header.transferSyntax.startsWith("1.2.840.10008.1.2.4");
  for (let i = 0; i < n; i++) {
    const [r, g, b] = planar ? [stored[i], stored[i + n], stored[i + 2 * n]] : [stored[i * 3], stored[i * 3 + 1], stored[i * 3 + 2]];
    out[i] = Math.fround(Math.fround(Math.fround(r + g) + b) / 3);
  }
  return out;
}

function rescale(values, header) {
  const { rescaleSlope: slope, rescaleIntercept: intercept } = header;
  if (slope === 1 && intercept === 0) return values;
  const s = Math.fround(slope);
  const b = Math.fround(intercept);
  const out = new Float32Array(values.length);
  for (let i = 0; i < values.length; i++) out[i] = Math.fround(Math.fround(Math.fround(values[i]) * s) + b);
  return out;
}

/** The DICOM linear window to use, or null (same rules as app.ingest._linear_window). */
export function linearWindow(header) {
  if (header.hasVoiLutSequence || header.voiFunction !== "LINEAR") return null;
  if (!header.windowCenter.length || !header.windowWidth.length) return null;
  const center = header.windowCenter[0];
  const width = header.windowWidth[0];
  if (!Number.isFinite(center) || !Number.isFinite(width) || width <= 1) return null;
  return { center, width };
}

/** DICOM SIGMOID VOI function (PS3.3 C.11.2.1.3.1), as pydicom applies it; null if not used. */
function sigmoidWindow(header, values) {
  if (header.hasVoiLutSequence || header.voiFunction !== "SIGMOID") return null;
  const center = header.windowCenter[0];
  const width = header.windowWidth[0];
  if (!Number.isFinite(center) || !Number.isFinite(width) || width <= 0) return null;
  const yMax = 2 ** header.bitsStored - 1;
  const out = new Float64Array(values.length);
  for (let i = 0; i < values.length; i++) out[i] = yMax / (1 + Math.exp((-4 * (values[i] - center)) / width));
  return out;
}

/**
 * Decode a DICOM file. Returns modality values (for interactive windowing), the default
 * 8-bit display image (identical to the Python service's model input) and metadata.
 */
export async function decodeDicom(bytes, { dicomParser, codecs }) {
  const header = readHeader(dicomParser, bytes);
  const { rows, columns } = header;
  if (!rows || !columns) throw new ImageError("DICOM file contains no image");
  if (rows * columns > MAX_PIXELS) {
    throw new ImageError(`Image too large (${columns}x${rows}); the limit is ${MAX_PIXELS.toLocaleString("en-US")} pixels`);
  }
  if (header.transferSyntax === TS.DEFLATED_LE) throw new ImageError("Deflated DICOM files are not supported");

  let ds;
  try {
    ds = dicomParser.parseDicom(bytes);
  } catch (err) {
    throw new ImageError(`This DICOM file is damaged or incomplete and could not be read (${err?.exception || err?.message || err})`);
  }
  const pixelElement = ds.elements.x7fe00010;
  if (!pixelElement) throw new ImageError("DICOM file contains no image");
  const count = rows * columns * header.samples;

  let stored;
  try {
    if (pixelElement.encapsulatedPixelData) {
      let frame;
      if (pixelElement.basicOffsetTable?.length) {
        frame = dicomParser.readEncapsulatedImageFrame(ds, pixelElement, 0);
      } else if (header.frames === 1) {
        // No offset table: a single frame is all fragments joined together.
        frame = dicomParser.readEncapsulatedPixelDataFromFragments(ds, pixelElement, 0, pixelElement.fragments.length);
      } else if (pixelElement.fragments.length === header.frames) {
        // Multi-frame without an offset table, one fragment per frame (the usual layout).
        frame = dicomParser.readEncapsulatedPixelDataFromFragments(ds, pixelElement, 0, 1);
      } else {
        // Frames split over several fragments: find boundaries from JPEG end-of-image markers.
        const table = dicomParser.createJPEGBasicOffsetTable(ds, pixelElement);
        frame = dicomParser.readEncapsulatedImageFrame(ds, pixelElement, 0, table);
      }
      stored = await decodeCompressed(header, frame, codecs);
    } else {
      const bytesNeeded = (count * header.bitsAllocated) / 8;
      if (pixelElement.length < bytesNeeded) throw new ImageError("DICOM pixel data is shorter than the header says; the file is damaged or truncated");
      stored = storedArray(bytes, pixelElement.dataOffset, count, header.bitsAllocated, header.pixelRepresentation === 1, header.transferSyntax === TS.EXPLICIT_BE);
      stored = maskBitsStored(stored, header);
    }
  } catch (err) {
    if (err instanceof ImageError) throw err;
    throw new ImageError(`Could not decode DICOM pixel data: ${err?.message || err}`);
  }
  if (stored.length < count) throw new ImageError("Decoded image is smaller than the DICOM header says");

  const gray = toGrayscale(stored, header);
  const values = rescale(gray, header);

  const window = linearWindow(header);
  let display;
  let displayWindow;
  if (window) {
    display = windowToUint8(values, window.center, window.width);
    displayWindow = window;
  } else {
    const sigmoid = sigmoidWindow(header, values);
    display = toUint8(sigmoid || values, columns, rows);
    const { lo, hi } = stretchRange(values, columns, rows);
    displayWindow = hi > lo ? { center: lo + 0.5 + (hi - lo) / 2, width: hi - lo + 1 } : { center: lo, width: 2 };
  }
  const inverted = header.photometric === "MONOCHROME1";
  if (inverted) display = invert(display);

  return {
    width: columns,
    height: rows,
    values,
    display,
    defaultWindow: displayWindow,
    inverted,
    format: "dicom",
    compression: TS_NAMES[header.transferSyntax] || header.transferSyntax,
    pixelSpacing: header.pixelSpacing,
    hint: header.hint,
    bitsStored: header.bitsStored,
  };
}

/** Interactive window/level over modality values (display only). */
export function renderWindow(image, center, width) {
  const low = center - 0.5 - (width - 1) / 2;
  const u8 = mapLinear(image.values, low, 255 / Math.max(1e-6, width - 1));
  return image.inverted ? invert(u8) : u8;
}
