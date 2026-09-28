import { test } from "node:test";
import assert from "node:assert/strict";
import { buildReport } from "../js/report.js";

const study = { modality: "panoramic", width: 2800, height: 1400, format: "dicom", compression: "JPEG 2000 Lossless", pixelSpacing: [0.1, 0.1], date: "2026-09-28" };

test("report includes confirmed and manual findings, never pending or rejected AI", () => {
  const text = buildReport(study, [
    { type: "caries", tooth: "36", source: "ai", status: "confirmed", confidence: 0.82 },
    { type: "periapical_lesion", tooth: "46", source: "ai", status: "pending", confidence: 0.41 },
    { type: "bone_loss", tooth: null, source: "ai", status: "rejected", confidence: 0.3 },
    { type: "crown", tooth: "16", source: "manual", status: "confirmed", note: "porcelain" },
  ], { measurements: [{ label: "Distance 1", value: "4.2 mm" }], impression: "Recall in 6 months." });
  assert.match(text, /Tooth 36: caries \(AI 82%\)/);
  assert.match(text, /Tooth 16: crown \(porcelain\)/);
  assert.doesNotMatch(text, /periapical/);
  assert.doesNotMatch(text, /bone loss/);
  assert.match(text, /1 AI suggestion\(s\) not yet reviewed/);
  assert.match(text, /Distance 1: 4\.2 mm/);
  assert.match(text, /IMPRESSION\n  Recall in 6 months\./);
  assert.match(text, /Panoramic radiograph, 2800 x 1400 px \(DICOM, JPEG 2000 Lossless\)/);
});

test("empty report says so plainly", () => {
  const text = buildReport({ ...study, format: "image", pixelSpacing: null }, []);
  assert.match(text, /No pathology recorded/);
  assert.match(text, /None recorded/);
  assert.doesNotMatch(text, /NOTE:/);
  assert.doesNotMatch(text, /Pixel spacing/);
});

test("findings are ordered by tooth with unassigned last", () => {
  const text = buildReport(study, [
    { type: "caries", tooth: null, source: "manual", status: "confirmed" },
    { type: "caries", tooth: "46", source: "manual", status: "confirmed" },
    { type: "caries", tooth: "11", source: "manual", status: "confirmed" },
  ]);
  const order = [...text.matchAll(/- (Tooth \d+|Location not assigned)/g)].map((m) => m[1]);
  assert.deepEqual(order, ["Tooth 11", "Tooth 46", "Location not assigned"]);
});
