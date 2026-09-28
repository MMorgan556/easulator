import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { join } from "node:path";
import { deflateSync } from "node:zlib";
import { decodePng16, PngError } from "../js/png16.js";
import { fixtures } from "./helpers.mjs";

function chunk(type, data) {
  const out = Buffer.alloc(12 + data.length);
  out.writeUInt32BE(data.length, 0);
  out.write(type, 4, "latin1");
  data.copy(out, 8);
  return out; // CRC is not checked by the decoder
}

function png16(width, height, idat) {
  const ihdr = Buffer.alloc(13);
  ihdr.writeUInt32BE(width, 0);
  ihdr.writeUInt32BE(height, 4);
  ihdr[8] = 16; // bit depth
  ihdr[9] = 0; // grayscale
  return new Uint8Array(Buffer.concat([Buffer.from([137, 80, 78, 71, 13, 10, 26, 10]), chunk("IHDR", ihdr), ...(idat ? [chunk("IDAT", idat)] : []), chunk("IEND", Buffer.alloc(0))]));
}

test("8-bit and colour PNGs are left to the browser", async () => {
  assert.equal(await decodePng16(new Uint8Array(readFileSync(join(fixtures, "gray_png.png")))), null);
  assert.equal(await decodePng16(new Uint8Array(readFileSync(join(fixtures, "rgb_png.png")))), null);
});

test("interlaced and progressive 16-bit PNGs decode to the same pixels", async () => {
  const a = await decodePng16(new Uint8Array(readFileSync(join(fixtures, "gray16_png.png"))));
  const b = await decodePng16(new Uint8Array(readFileSync(join(fixtures, "gray16_interlaced_png.png"))));
  assert.deepEqual(Array.from(a.values), Array.from(b.values));
});

test("size limit is checked before any image data is expanded", async () => {
  let reported = null;
  const bomb = png16(60000, 60000, deflateSync(Buffer.alloc(1024)));
  await assert.rejects(decodePng16(bomb, { maxPixels: 20_000_000, onTooLarge: (w, h) => (reported = [w, h]) }), PngError);
  assert.deepEqual(reported, [60000, 60000]);
});

test("data larger than the header allows is refused", async () => {
  const bomb = png16(4, 4, deflateSync(Buffer.alloc(10_000_000)));
  await assert.rejects(decodePng16(bomb), /larger than its header/);
});

test("corrupt, truncated or missing image data gives a PngError", async () => {
  const good = deflateSync(Buffer.alloc(4 * (1 + 8)));
  await assert.rejects(decodePng16(png16(4, 4, Buffer.from("not zlib at all"))), /corrupt or incomplete/);
  await assert.rejects(decodePng16(png16(4, 4, good.subarray(0, good.length - 6))), PngError);
  await assert.rejects(decodePng16(png16(4, 4, null)), /missing/);
});
