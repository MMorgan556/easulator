import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { join } from "node:path";
import { buildFindings, guessModality } from "../js/findings.js";
import { fixtures } from "./helpers.mjs";

const { scenes, modalities } = JSON.parse(readFileSync(join(fixtures, "findings_cases.json"), "utf8"));

test("tooth numbering and findings match app/postprocess.py on 200 random scenes", () => {
  scenes.forEach((s, n) => {
    const { teeth, findings } = buildFindings(s.detections, { width: s.width, height: s.height, modality: s.modality });
    assert.deepEqual(teeth.map((t) => [t.fdi, t.confidence, t.box]), s.teeth.map((t) => [t.fdi, t.confidence, t.box]), `scene ${n} teeth`);
    assert.deepEqual(
      findings.map((f) => [f.id, f.type, f.tooth, f.needsReview]),
      s.findings.map((f) => [f.id, f.type, f.tooth, f.needsReview]),
      `scene ${n} findings`,
    );
    findings.forEach((f, i) => assert.ok(Math.abs(f.confidence - s.findings[i].confidence) <= 0.0011, `scene ${n} confidence`));
  });
});

test("modality guess matches app/ingest.py", () => {
  for (const m of modalities) assert.equal(guessModality(m.width, m.height, m.hint), m.expected, JSON.stringify(m));
});
