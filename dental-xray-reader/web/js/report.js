// Structured draft report from the reviewed findings. Only findings the dentist confirmed or
// added go in as findings; unreviewed AI suggestions are counted, never reported as fact.

import { LABELS, PATHOLOGY } from "./findings.js";

const MODALITY = { panoramic: "Panoramic", bitewing: "Bitewing", periapical: "Periapical", unknown: "Dental" };

function toothSort(a, b) {
  if (a === b) return 0;
  if (a === null) return 1;
  if (b === null) return -1;
  return a < b ? -1 : 1;
}

function describe(f) {
  const where = f.tooth ? `Tooth ${f.tooth}` : "Location not assigned to a tooth";
  const extra = [];
  if (f.source === "ai" && f.confidence != null) extra.push(`AI ${Math.round(f.confidence * 100)}%`);
  if (f.note) extra.push(f.note);
  return `${where}: ${LABELS[f.type].toLowerCase()}${extra.length ? ` (${extra.join("; ")})` : ""}`;
}

/**
 * @param {object} study  { modality, width, height, format, compression, pixelSpacing, date }
 * @param {Array} findings  items with type, tooth, source ("ai"|"manual"), status, confidence, note
 * @param {object} extra  { measurements: [{label, value}], impression: string }
 */
export function buildReport(study, findings, { measurements = [], impression = "" } = {}) {
  const accepted = findings.filter((f) => f.status === "confirmed" || f.source === "manual");
  const pending = findings.filter((f) => f.source === "ai" && f.status === "pending");
  const pathology = accepted.filter((f) => PATHOLOGY.has(f.type)).sort((a, b) => toothSort(a.tooth, b.tooth));
  const work = accepted.filter((f) => !PATHOLOGY.has(f.type) && f.type !== "tooth").sort((a, b) => toothSort(a.tooth, b.tooth));

  const lines = [];
  lines.push("DENTAL RADIOGRAPH REPORT (DRAFT)");
  lines.push(`Date: ${study.date || new Date().toISOString().slice(0, 10)}`);
  const source = study.format === "dicom" ? `DICOM${study.compression ? `, ${study.compression}` : ""}` : "image file";
  lines.push(`Image: ${MODALITY[study.modality] || "Dental"} radiograph, ${study.width} x ${study.height} px (${source})`);
  if (study.pixelSpacing) lines.push(`Pixel spacing: ${study.pixelSpacing[0]} x ${study.pixelSpacing[1]} mm`);
  lines.push("");

  lines.push("FINDINGS");
  if (!pathology.length) lines.push("  No pathology recorded.");
  for (const f of pathology) lines.push(`  - ${describe(f)}`);
  lines.push("");

  lines.push("EXISTING DENTAL WORK");
  if (!work.length) lines.push("  None recorded.");
  for (const f of work) lines.push(`  - ${describe(f)}`);

  if (measurements.length) {
    lines.push("");
    lines.push("MEASUREMENTS");
    for (const m of measurements) lines.push(`  - ${m.label}: ${m.value}`);
  }

  if (impression.trim()) {
    lines.push("");
    lines.push("IMPRESSION");
    for (const line of impression.trim().split("\n")) lines.push(`  ${line}`);
  }

  if (pending.length) {
    lines.push("");
    lines.push(`NOTE: ${pending.length} AI suggestion(s) not yet reviewed and not included above.`);
  }
  lines.push("");
  lines.push("This draft supports, and does not replace, the clinical judgement of a licensed dentist.");
  return lines.join("\n");
}
