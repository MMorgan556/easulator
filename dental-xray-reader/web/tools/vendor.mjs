// Copies the browser libraries from node_modules into vendor/ (served as static files).
// Run after `npm ci`. vendor/ is generated, not committed.
import { cpSync, existsSync, mkdirSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const root = join(dirname(fileURLToPath(import.meta.url)), "..");
const modules = join(root, "node_modules");
const vendor = join(root, "vendor");

const files = {
  "dicom-parser": [["dist/dicomParser.min.js", "dicomParser.min.js"]],
  "@cornerstonejs/codec-openjpeg": [
    ["dist/openjpegwasm_decode.js", "openjpeg/openjpegwasm_decode.js"],
    ["dist/openjpegwasm_decode.wasm", "openjpeg/openjpegwasm_decode.wasm"],
  ],
  "@cornerstonejs/codec-charls": [
    ["dist/charlswasm_decode.js", "charls/charlswasm_decode.js"],
    ["dist/charlswasm_decode.wasm", "charls/charlswasm_decode.wasm"],
  ],
  "jpeg-lossless-decoder-js": [["release/lossless.js", "jpeg-lossless.js"]],
  "onnxruntime-web": [
    ["dist/ort.wasm.min.mjs", "onnxruntime-web/ort.wasm.min.mjs"],
    ["dist/ort-wasm-simd-threaded.mjs", "onnxruntime-web/ort-wasm-simd-threaded.mjs"],
    ["dist/ort-wasm-simd-threaded.wasm", "onnxruntime-web/ort-wasm-simd-threaded.wasm"],
  ],
};

const bundledLibraries = {
  "@cornerstonejs/codec-openjpeg": "Contains OpenJPEG (BSD 2-Clause), https://github.com/uclouvain/openjpeg",
  "@cornerstonejs/codec-charls": "Contains CharLS (BSD 3-Clause), https://github.com/team-charls/charls",
};

rmSync(vendor, { recursive: true, force: true });
const notices = ["Third-party software served with the Dental X-ray Reader web app.", ""];
for (const [pkg, list] of Object.entries(files)) {
  const pkgDir = join(modules, pkg);
  const meta = JSON.parse(readFileSync(join(pkgDir, "package.json"), "utf8"));
  for (const [from, to] of list) {
    const dest = join(vendor, to);
    mkdirSync(dirname(dest), { recursive: true });
    cpSync(join(pkgDir, from), dest);
  }
  notices.push(`${"=".repeat(72)}\n${pkg} ${meta.version} (${meta.license})`);
  if (bundledLibraries[pkg]) notices.push(bundledLibraries[pkg]);
  const licenseFile = ["LICENSE", "LICENSE.md", "LICENSE.txt"].map((f) => join(pkgDir, f)).find(existsSync);
  notices.push(licenseFile ? readFileSync(licenseFile, "utf8").trim() : `Licensed under ${meta.license}; see ${meta.homepage || meta.repository?.url || "the package"}.`);
  notices.push("");
}
writeFileSync(join(vendor, "THIRD_PARTY_NOTICES.txt"), notices.join("\n"));
console.log(`vendor/ ready (${Object.keys(files).length} packages)`);
