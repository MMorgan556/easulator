// Node equivalents of the codecs the browser loads from vendor/.
import { createRequire } from "node:module";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const require = createRequire(import.meta.url);
const here = dirname(fileURLToPath(import.meta.url));
const modules = join(here, "..", "node_modules");

export const dicomParser = require("dicom-parser");

function wasm(pkg, file) {
  let promise;
  return () => {
    promise ||= require(join(modules, pkg, "dist", file))({
      locateFile: (f) => join(modules, pkg, "dist", f),
      print: () => {},
      printErr: () => {},
    });
    return promise;
  };
}

export const codecs = {
  openjpeg: wasm("@cornerstonejs/codec-openjpeg", "openjpegwasm_decode.js"),
  charls: wasm("@cornerstonejs/codec-charls", "charlswasm_decode.js"),
  jpegLossless: async () => (await import("jpeg-lossless-decoder-js")).Decoder,
  jpegBaseline: async () => {
    throw new Error("JPEG baseline is decoded by the browser; covered by the browser test");
  },
};

export const fixtures = join(here, "fixtures");
