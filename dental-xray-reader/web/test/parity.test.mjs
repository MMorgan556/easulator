// The browser must derive exactly the image app/ingest.py derives (the AI model's input).
import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { join } from "node:path";
import { decodeDicom, isDicom } from "../js/dicom.js";
import { codecs, dicomParser, fixtures } from "./helpers.mjs";

const cases = JSON.parse(readFileSync(join(fixtures, "cases.json"), "utf8"));

function compare(actual, expected, tolerance) {
  assert.equal(actual.length, expected.length, "pixel count");
  let worst = 0;
  let differing = 0;
  for (let i = 0; i < actual.length; i++) {
    const d = Math.abs(actual[i] - expected[i]);
    if (d) differing++;
    if (d > worst) worst = d;
  }
  assert.ok(worst <= tolerance, `max difference ${worst} > ${tolerance} (${differing} pixels differ)`);
}

for (const c of cases.filter((c) => c.file.endsWith(".dcm") && c.name !== "jpeg_baseline8")) {
  test(`DICOM parity: ${c.name}`, async () => {
    const bytes = new Uint8Array(readFileSync(join(fixtures, c.file)));
    assert.ok(isDicom(bytes));
    const image = await decodeDicom(bytes, { dicomParser, codecs });
    assert.equal(image.width, c.width);
    assert.equal(image.height, c.height);
    assert.deepEqual(image.pixelSpacing, c.pixel_spacing_mm ? c.pixel_spacing_mm : null);
    const expected = new Uint8Array(readFileSync(join(fixtures, `${c.name}.expected.bin`)));
    compare(image.display, expected, c.name === "window_sigmoid" ? 1 : c.tolerance);
  });
}

for (const c of cases.filter((c) => !c.file.endsWith(".dcm"))) {
  test(`image parity: ${c.name}`, async () => {
    const { grayToImage, rasterToImage } = await import("../js/imaging.js");
    const { decodePng16 } = await import("../js/png16.js");
    const png16 = await decodePng16(new Uint8Array(readFileSync(join(fixtures, c.file))));
    const rgba = new Uint8Array(readFileSync(join(fixtures, `${c.name}.rgba.bin`)));
    const image = png16 ? grayToImage(png16.values, png16.width, png16.height) : rasterToImage(rgba, c.width, c.height);
    assert.equal(Boolean(png16), c.name.startsWith("gray16"), "16-bit decoder used exactly for 16-bit PNGs");
    const expected = new Uint8Array(readFileSync(join(fixtures, `${c.name}.expected.bin`)));
    compare(image.display, expected, c.tolerance);
  });
}

test("default window reproduces the default display exactly", async () => {
  const { renderWindow } = await import("../js/dicom.js");
  for (const c of cases.filter((c) => c.file.endsWith(".dcm") && c.name !== "jpeg_baseline8")) {
    const image = await decodeDicom(new Uint8Array(readFileSync(join(fixtures, c.file))), { dicomParser, codecs });
    const { center, width } = image.defaultWindow;
    const rendered = renderWindow(image, center, width);
    compare(rendered, image.display, 0);
  }
});
