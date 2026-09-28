// Opens a user-selected file entirely in the browser. Codecs are fetched lazily from vendor/.

import { decodeDicom, ImageError, isDicom, MAX_PIXELS } from "./dicom.js";
import { grayToImage, rasterToImage } from "./imaging.js";
import { decodePng16, PngError } from "./png16.js";

const vendor = new URL("../vendor/", import.meta.url);

const scripts = new Map();
function loadScript(path, globalName) {
  if (!scripts.has(path)) {
    scripts.set(
      path,
      new Promise((resolve, reject) => {
        const el = document.createElement("script");
        el.src = new URL(path, vendor).href;
        el.onload = () => (globalThis[globalName] ? resolve(globalThis[globalName]) : reject(new Error(`${path} did not load`)));
        el.onerror = () => reject(new Error(`Could not load ${path}`));
        document.head.append(el);
      }),
    );
  }
  return scripts.get(path);
}

function wasmCodec(script, globalName) {
  let codec;
  return () => {
    codec ||= loadScript(script, globalName).then((factory) =>
      factory({ locateFile: (file) => new URL(`${script.split("/")[0]}/${file}`, vendor).href, print: () => {}, printErr: () => {} }),
    );
    return codec;
  };
}

async function decodeToRgba(blob) {
  let bitmap;
  try {
    // No colour management (Pillow ignores ICC profiles too); EXIF rotation applied.
    bitmap = await createImageBitmap(blob, { colorSpaceConversion: "none", imageOrientation: "from-image", premultiplyAlpha: "none" });
  } catch {
    throw new ImageError("This file is neither DICOM nor an image format the browser can open (use DICOM, PNG, JPEG, WebP or BMP)");
  }
  const { width, height } = bitmap;
  if (width * height > MAX_PIXELS) {
    bitmap.close();
    throw new ImageError(`Image too large (${width}x${height}); the limit is ${MAX_PIXELS.toLocaleString("en-US")} pixels`);
  }
  const canvas = new OffscreenCanvas(width, height);
  const ctx = canvas.getContext("2d", { willReadFrequently: true });
  ctx.drawImage(bitmap, 0, 0);
  bitmap.close();
  return { rgba: ctx.getImageData(0, 0, width, height).data, width, height };
}

export const codecs = {
  openjpeg: wasmCodec("openjpeg/openjpegwasm_decode.js", "OpenJPEGWASM"),
  charls: wasmCodec("charls/charlswasm_decode.js", "CharLSWASM"),
  jpegLossless: async () => (await import(new URL("jpeg-lossless.js", vendor).href)).Decoder,
  jpegBaseline: (frame) => decodeToRgba(new Blob([frame], { type: "image/jpeg" })),
};

async function decodeFile(file, bytes) {
  if (isDicom(bytes)) {
    const dicomParser = await loadScript("dicomParser.min.js", "dicomParser");
    return decodeDicom(bytes, { dicomParser, codecs });
  }
  if (/\.(tif|tiff)$/i.test(file.name)) {
    throw new ImageError("TIFF can't be opened in most browsers; export the X-ray as DICOM or PNG");
  }
  let png16;
  try {
    png16 = await decodePng16(bytes);
  } catch (err) {
    if (err instanceof PngError) throw new ImageError(`This PNG file is damaged: ${err.message}`);
    throw err;
  }
  if (png16) {
    if (png16.width * png16.height > MAX_PIXELS) {
      throw new ImageError(`Image too large (${png16.width}x${png16.height}); the limit is ${MAX_PIXELS.toLocaleString("en-US")} pixels`);
    }
    return grayToImage(png16.values, png16.width, png16.height);
  }
  const { rgba, width, height } = await decodeToRgba(new Blob([bytes], { type: file.type || "application/octet-stream" }));
  return rasterToImage(rgba, width, height);
}

export async function openFile(file) {
  const bytes = new Uint8Array(await file.arrayBuffer());
  const image = await decodeFile(file, bytes);
  if (Math.min(image.width, image.height) < 32) {
    throw new ImageError(`Image too small to analyze (${image.width}x${image.height})`);
  }
  return image;
}

export { ImageError };
