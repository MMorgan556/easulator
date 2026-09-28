// Application controller: file opening, tools, findings review, chart, report and exports.
// All state lives in memory on this device.

import { Detector, loadManifest } from "./ai.js";
import { renderWindow } from "./dicom.js";
import { assignTooth, buildFindings, FINDING_TYPES, guessModality, LABELS, PATHOLOGY, PERMANENT_TEETH } from "./findings.js";
import { clahe, invert } from "./imaging.js";
import { ImageError, openFile } from "./loader.js";
import { buildReport } from "./report.js";
import { Viewer } from "./viewer.js";

const $ = (id) => document.getElementById(id);
const el = (tag, props = {}, ...children) => {
  const node = document.createElement(tag);
  for (const [key, value] of Object.entries(props)) {
    if (key === "class") node.className = value;
    else if (key === "dataset") Object.assign(node.dataset, value);
    else if (key.startsWith("on")) node.addEventListener(key.slice(2), value);
    else if (value !== undefined && value !== null && value !== false) node.setAttribute(key, value === true ? "" : value);
  }
  for (const child of children.flat()) if (child !== null && child !== undefined) node.append(child);
  return node;
};

const state = {
  image: null,
  modality: "panoramic",
  toothDetections: [], // raw AI tooth boxes, re-numbered when the radiograph type changes
  otherDetections: [],
  teeth: [],
  findings: [],
  measurements: [],
  selected: null,
  selectedTooth: null,
  window: null,
  valueRange: [0, 255],
  invert: false,
  enhance: false,
  showTeeth: true,
  spacing: null,
  spacingSource: null,
  pendingBox: null,
  reportEdited: false,
  counters: { manual: 0, measure: 0 },
  ai: { manifest: null, detector: null, lastRun: null },
};

// ---------------------------------------------------------------- helpers

let toastTimer;
function toast(message, kind = "") {
  const t = $("toast");
  t.textContent = message;
  t.className = `toast ${kind}`;
  t.hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => (t.hidden = true), kind === "error" ? 7000 : 3500);
}

function busy(text) {
  $("busy-text").textContent = text || "";
  $("busy").hidden = !text;
}

const nextFrame = () => new Promise((resolve) => requestAnimationFrame(() => setTimeout(resolve, 0)));

function lengthText(line) {
  const dx = line.x2 - line.x1;
  const dy = line.y2 - line.y1;
  if (state.spacing) {
    const [rowMm, colMm] = state.spacing;
    return `${Math.hypot(dx * colMm, dy * rowMm).toFixed(1)} mm`;
  }
  return `${Math.round(Math.hypot(dx, dy))} px`;
}

function download(blob, name) {
  const url = URL.createObjectURL(blob);
  const a = el("a", { href: url, download: name });
  document.body.append(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 2000);
}

// Export names never reuse the uploaded file name, which often contains the patient's name.
const exportName = (ext) => `xray-analysis-${new Date().toISOString().slice(0, 16).replace(/[-:T]/g, "")}.${ext}`;

// ---------------------------------------------------------------- viewer

const viewer = new Viewer($("viewer"), {
  onWindow(dx, dy) {
    if (!state.window) return;
    const span = Math.max(1, state.valueRange[1] - state.valueRange[0]);
    const step = span / 400;
    state.window = { width: Math.max(1.01, state.window.width + dx * step), center: state.window.center - dy * step };
    scheduleDisplay();
  },
  onMeasure(line) {
    state.counters.measure += 1;
    const m = { id: `D${state.counters.measure}`, ...line };
    state.measurements.push(m);
    select(m.id);
    renderAll();
  },
  onBox(box) {
    state.pendingBox = box;
    const tooth = assignTooth(box, state.teeth);
    $("add-tooth").value = tooth || "";
    showTab("findings");
    renderAddForm();
    $("add-type").focus();
  },
  onSelect(finding) {
    select(finding ? finding.id : null);
  },
});
viewer.callbacks.measureText = (a, b) => lengthText({ x1: a.x, y1: a.y, x2: b.x, y2: b.y });
$("viewer").addEventListener("contextmenu", (e) => e.preventDefault());

let displayFrame = 0;
function scheduleDisplay() {
  if (displayFrame) return;
  displayFrame = requestAnimationFrame(() => {
    displayFrame = 0;
    updateDisplay();
  });
}

function updateDisplay() {
  const image = state.image;
  if (!image) return;
  const def = image.defaultWindow;
  const isDefault = state.window.center === def.center && state.window.width === def.width;
  let pixels = isDefault ? image.display : renderWindow(image, state.window.center, state.window.width);
  if (state.invert) pixels = invert(pixels);
  if (state.enhance) pixels = clahe(pixels, image.width, image.height);
  viewer.setDisplay(pixels);
  renderWindowControls();
  renderStatus();
}

function select(id) {
  state.selected = id;
  renderAll();
}

// ---------------------------------------------------------------- opening files

async function open(file) {
  if (!file) return;
  if (hasWork() && !confirm("Open a new X-ray? Findings and measurements for the current one will be cleared.")) return;
  busy("Opening…");
  await nextFrame();
  let image;
  try {
    image = await openFile(file);
  } catch (err) {
    busy(null);
    if (err instanceof ImageError) toast(err.message, "error");
    else {
      console.error(err);
      toast("Could not open this file.", "error");
    }
    return;
  }
  Object.assign(state, {
    image,
    modality: (() => {
      const m = guessModality(image.width, image.height, image.hint);
      return m === "unknown" ? "periapical" : m;
    })(),
    toothDetections: [],
    otherDetections: [],
    teeth: [],
    findings: [],
    measurements: [],
    selected: null,
    selectedTooth: null,
    window: { ...image.defaultWindow },
    invert: false,
    enhance: false,
    spacing: image.pixelSpacing,
    spacingSource: image.pixelSpacing ? "DICOM" : null,
    pendingBox: null,
    reportEdited: false,
    counters: { manual: 0, measure: 0 },
  });
  state.ai.lastRun = null;
  let lo = Infinity;
  let hi = -Infinity;
  for (const v of image.values) {
    if (v < lo) lo = v;
    if (v > hi) hi = v;
  }
  state.valueRange = [lo, hi];
  $("impression").value = "";
  $("empty").hidden = true;
  for (const id of ["export-png", "export-json", "print-report"]) $(id).disabled = false;
  viewer.setImage(image, image.display);
  updateDisplay();
  renderAll();
  busy(null);
  if (state.ai.manifest) runAi();
}

function hasWork() {
  return state.findings.some((f) => f.source === "manual" || f.status !== "pending") || state.measurements.length > 0;
}

// ---------------------------------------------------------------- AI

async function initAi() {
  let manifest = null;
  try {
    manifest = await loadManifest(new URL("../model/", import.meta.url));
  } catch (err) {
    console.error(err);
  }
  state.ai.manifest = manifest;
  renderAi();
}

async function runAi() {
  const { manifest } = state.ai;
  if (!manifest || !state.image) return;
  busy("Running AI on this device…");
  try {
    state.ai.detector ||= new Detector(manifest, new URL("./ai-worker.js", import.meta.url));
    const { detections, milliseconds } = await state.ai.detector.detect(state.image);
    state.toothDetections = detections.filter((d) => d.type === "tooth");
    state.otherDetections = detections.filter((d) => d.type !== "tooth");
    applyDetections();
    state.ai.lastRun = { milliseconds, count: state.findings.filter((f) => f.source === "ai").length };
    const n = state.ai.lastRun.count;
    const teeth = state.teeth.length;
    toast(`AI suggested ${n} finding${n === 1 ? "" : "s"} and found ${teeth} ${teeth === 1 ? "tooth" : "teeth"} in ${(milliseconds / 1000).toFixed(1)} s. Review each suggestion.`);
  } catch (err) {
    console.error(err);
    toast(`AI analysis failed: ${err.message}`, "error");
  } finally {
    busy(null);
    renderAll();
  }
}

/** (Re)build AI teeth and findings, keeping review decisions made so far. */
function applyDetections() {
  const previous = new Map(state.findings.filter((f) => f.source === "ai").map((f) => [f.id, f]));
  const { teeth, findings } = buildFindings([...state.toothDetections, ...state.otherDetections], {
    width: state.image.width,
    height: state.image.height,
    modality: state.modality,
    minConfidence: state.ai.manifest?.confidence ?? 0.25,
  });
  state.teeth = teeth;
  const ai = findings.map((f) => {
    const before = previous.get(f.id);
    return { ...f, source: "ai", status: before?.status || "pending", tooth: before?.edited ? before.tooth : f.tooth, type: before?.edited ? before.type : f.type, edited: before?.edited, note: before?.note || "" };
  });
  state.findings = [...ai, ...state.findings.filter((f) => f.source === "manual")];
}

// ---------------------------------------------------------------- findings actions

function updateFinding(id, changes) {
  const f = state.findings.find((x) => x.id === id);
  if (!f) return;
  Object.assign(f, changes);
  renderAll();
}

function removeSelected() {
  const id = state.selected;
  if (!id) return;
  if (id.startsWith("D")) {
    state.measurements = state.measurements.filter((m) => m.id !== id);
  } else {
    const f = state.findings.find((x) => x.id === id);
    if (!f) return;
    if (f.source === "manual") state.findings = state.findings.filter((x) => x.id !== id);
    else f.status = "rejected";
  }
  state.selected = null;
  renderAll();
}

function addFinding(event) {
  event.preventDefault();
  if (!state.image) {
    toast("Open an X-ray first.");
    return;
  }
  state.counters.manual += 1;
  const f = {
    id: `M${state.counters.manual}`,
    type: $("add-type").value,
    tooth: $("add-tooth").value || null,
    note: $("add-note").value.trim(),
    box: state.pendingBox,
    source: "manual",
    status: "confirmed",
    confidence: null,
  };
  state.findings.push(f);
  state.pendingBox = null;
  $("add-note").value = "";
  select(f.id);
  toast(`Added ${f.id}.`);
}

// ---------------------------------------------------------------- rendering

function toothOptions(selected) {
  const all = [...PERMANENT_TEETH.upper, ...PERMANENT_TEETH.lower].sort();
  return [el("option", { value: "" }, "Not assigned"), ...all.map((t) => el("option", { value: t, selected: t === selected }, t))];
}

function typeOptions(selected) {
  return FINDING_TYPES.filter((t) => t !== "tooth").map((t) => el("option", { value: t, selected: t === selected }, LABELS[t]));
}

function renderAi() {
  const { manifest, lastRun } = state.ai;
  const chip = $("ai-chip");
  if (!manifest) {
    chip.textContent = "No model";
    chip.className = "chip off";
    $("ai-text").textContent =
      "No trained dental model is installed on this site yet, so findings are recorded manually. When a model is published here, it runs automatically on this device.";
    $("about-ai").textContent = "No model is installed yet. Once a trained model is published with the site, detection runs on your device.";
    $("ai-run").disabled = true;
    return;
  }
  chip.textContent = "Ready";
  chip.className = "chip ready";
  const metrics = manifest.metrics && Object.keys(manifest.metrics).length ? ` Validation: ${Object.entries(manifest.metrics).map(([k, v]) => `${k} ${v}`).join(", ")}.` : "";
  $("ai-text").textContent = `${manifest.name || "Detector"} (${manifest.classes.length} classes, exported ${manifest.exported || "unknown"}).${metrics} Suggestions stay unreviewed until you confirm them.${lastRun ? ` Last run: ${(lastRun.milliseconds / 1000).toFixed(1)} s.` : ""}`;
  $("about-ai").textContent = `${manifest.name || "A detector"} runs in your browser with ONNX Runtime. Its suggestions are never added to the report until you confirm them.`;
  $("ai-run").disabled = !state.image;
}

function findingSub(f) {
  const bits = [f.tooth ? `Tooth ${f.tooth}` : "No tooth"];
  if (f.source === "ai") bits.push(`AI ${Math.round(f.confidence * 100)}%${f.needsReview ? ", low confidence" : ""}`);
  if (!f.box) bits.push("not marked on image");
  if (f.note) bits.push(f.note);
  return bits.join(" · ");
}

function renderFindings() {
  const list = $("findings-list");
  list.replaceChildren();
  const items = state.findings;
  for (const f of items) {
    const color = PATHOLOGY.has(f.type) ? "var(--pathology)" : "var(--work)";
    const statusLabel = f.source === "manual" ? "manual" : f.status;
    const actions = el("span", { class: "actions" });
    const stop = (fn) => (e) => {
      e.stopPropagation();
      fn();
    };
    if (f.source === "ai") {
      if (f.status !== "confirmed") actions.append(el("button", { type: "button", class: "btn small", title: "Confirm", "aria-label": `Confirm ${f.id}`, onclick: stop(() => updateFinding(f.id, { status: "confirmed" })) }, "✓"));
      if (f.status !== "rejected") actions.append(el("button", { type: "button", class: "btn small", title: "Reject", "aria-label": `Reject ${f.id}`, onclick: stop(() => updateFinding(f.id, { status: "rejected" })) }, "✕"));
      if (f.status !== "pending") actions.append(el("button", { type: "button", class: "btn small ghost", title: "Undo review", "aria-label": `Undo review of ${f.id}`, onclick: stop(() => updateFinding(f.id, { status: "pending" })) }, "↺"));
    } else {
      actions.append(el("button", { type: "button", class: "btn small", title: "Delete", "aria-label": `Delete ${f.id}`, onclick: stop(() => {
        state.findings = state.findings.filter((x) => x.id !== f.id);
        if (state.selected === f.id) state.selected = null;
        renderAll();
      }) }, "Delete"));
    }
    const li = el(
      "li",
      {
        class: `finding ${f.status}${state.selected === f.id ? " selected" : ""}`,
        tabindex: "0",
        "aria-label": `${f.id} ${LABELS[f.type]}, ${findingSub(f)}, ${statusLabel}`,
        onclick: () => select(f.id),
        onkeydown: (e) => {
          if (e.key === "Enter" || e.key === " ") {
            e.preventDefault();
            select(f.id);
          }
        },
      },
      el("span", { class: "swatch", "aria-hidden": "true" }),
      el("span", { class: "title" }, `${f.id} · ${LABELS[f.type]} `, el("span", { class: `status ${statusLabel}` }, statusLabel)),
      actions,
      el("span", { class: "sub" }, findingSub(f)),
    );
    li.querySelector(".swatch").style.background = color;
    if (state.selected === f.id && f.status !== "rejected") {
      const toothSelect = el("select", { "aria-label": "Tooth", onchange: (e) => updateFinding(f.id, { tooth: e.target.value || null, edited: true, status: f.source === "ai" ? "confirmed" : f.status }), onclick: (e) => e.stopPropagation() }, toothOptions(f.tooth));
      const typeSelect = el("select", { "aria-label": "Finding type", onchange: (e) => updateFinding(f.id, { type: e.target.value, edited: true, status: f.source === "ai" ? "confirmed" : f.status }), onclick: (e) => e.stopPropagation() }, typeOptions(f.type));
      li.append(el("div", { class: "edit-row" }, typeSelect, toothSelect));
    }
    list.append(li);
  }
  $("findings-empty").hidden = items.length > 0;
  const pending = items.filter((f) => f.source === "ai" && f.status === "pending").length;
  $("findings-count").textContent = items.length ? `${items.length} total${pending ? ` · ${pending} to review` : ""}` : "";
}

function renderAddForm() {
  const box = state.pendingBox;
  $("add-title").textContent = box ? "New finding (area marked)" : "Add finding";
  $("add-box-note").textContent = box
    ? `Area marked: ${Math.round(box.x2 - box.x1)} × ${Math.round(box.y2 - box.y1)} px.`
    : "No area marked; use the Mark tool to outline it on the image.";
  $("add-cancel").hidden = !box;
}

function renderChart() {
  const chart = $("chart");
  chart.replaceChildren();
  const byTooth = new Map();
  for (const f of state.findings) {
    if (!f.tooth || f.status === "rejected") continue;
    if (!byTooth.has(f.tooth)) byTooth.set(f.tooth, []);
    byTooth.get(f.tooth).push(f);
  }
  const detected = new Set(state.teeth.map((t) => t.fdi));
  const button = (fdi) => {
    const list = byTooth.get(fdi) || [];
    const accepted = list.filter((f) => f.source === "manual" || f.status === "confirmed");
    const classes = ["tooth"];
    if (detected.has(fdi)) classes.push("detected");
    if (accepted.some((f) => !PATHOLOGY.has(f.type))) classes.push("work");
    if (accepted.some((f) => PATHOLOGY.has(f.type))) classes.push("pathology");
    if (list.some((f) => f.source === "ai" && f.status === "pending")) classes.push("pending");
    return el(
      "button",
      {
        type: "button",
        class: classes.join(" "),
        "aria-pressed": state.selectedTooth === fdi ? "true" : "false",
        "aria-label": `Tooth ${fdi}${list.length ? `, ${list.length} finding(s)` : ""}`,
        onclick: () => {
          state.selectedTooth = state.selectedTooth === fdi ? null : fdi;
          $("add-tooth").value = state.selectedTooth || "";
          renderChart();
        },
      },
      fdi,
    );
  };
  chart.append(...PERMANENT_TEETH.upper.map(button), el("div", { class: "midline", "aria-hidden": "true" }), ...PERMANENT_TEETH.lower.map(button));

  const tooth = state.selectedTooth;
  $("tooth-title").textContent = tooth ? `Tooth ${tooth}` : "Select a tooth";
  const items = tooth ? byTooth.get(tooth) || [] : [];
  $("tooth-findings").replaceChildren(
    ...(tooth && !items.length ? [el("li", { class: "muted small" }, "No findings recorded for this tooth.")] : []),
    ...items.map((f) =>
      el("li", {}, el("button", { type: "button", class: "btn block", onclick: () => { select(f.id); showTab("findings"); } }, `${f.id} · ${LABELS[f.type]} (${f.source === "manual" ? "manual" : f.status})`)),
    ),
  );
}

function currentReport() {
  const image = state.image;
  if (!image) return "";
  return buildReport(
    { modality: state.modality, width: image.width, height: image.height, format: image.format, compression: image.compression, pixelSpacing: state.spacingSource === "DICOM" ? state.spacing : null },
    state.findings,
    { measurements: state.measurements.map((m) => ({ label: m.id, value: lengthText(m) + (state.spacingSource === "calibrated" ? " (calibrated)" : "") })), impression: $("impression").value },
  );
}

function renderReport() {
  if (!state.reportEdited) $("report").value = currentReport();
  $("report-state").textContent = state.reportEdited ? "Edited by you. Regenerate discards your edits." : "Updates automatically as findings change.";
}

function renderMeasurements() {
  const list = $("measurements");
  list.replaceChildren(
    ...state.measurements.map((m) =>
      el(
        "li",
        { class: `measurement${state.selected === m.id ? " selected" : ""}`, tabindex: "0", onclick: () => select(m.id), onkeydown: (e) => e.key === "Enter" && select(m.id) },
        el("span", {}, `${m.id}: ${lengthText(m)}`),
        el("button", { type: "button", class: "btn small", "aria-label": `Delete ${m.id}`, onclick: (e) => {
          e.stopPropagation();
          state.measurements = state.measurements.filter((x) => x.id !== m.id);
          if (state.selected === m.id) state.selected = null;
          renderAll();
        } }, "Delete"),
      ),
    ),
  );
  $("measure-empty").hidden = state.measurements.length > 0;
  $("calibrate-form").hidden = !(state.selected && state.selected.startsWith("D"));
  $("spacing-text").textContent = state.spacing
    ? `${state.spacingSource === "DICOM" ? "DICOM" : "Calibrated"}: ${state.spacing[1].toFixed(4)} mm/px`
    : "Not calibrated (px)";
}

function renderWindowControls() {
  if (!state.image) return;
  const [lo, hi] = state.valueRange;
  const span = Math.max(1, hi - lo);
  const wc = $("wc");
  const ww = $("ww");
  wc.min = String(lo - span / 2);
  wc.max = String(hi + span / 2);
  ww.min = "1";
  ww.max = String(span * 2 + 1);
  wc.value = String(state.window.center);
  ww.value = String(state.window.width);
  $("wc-out").textContent = Math.round(state.window.center);
  $("ww-out").textContent = Math.round(state.window.width);
}

function renderMeta() {
  const image = state.image;
  const rows = image
    ? [
        ["Format", image.format === "dicom" ? "DICOM" : "Image"],
        ["Compression", image.compression || "—"],
        ["Size", `${image.width} × ${image.height} px`],
        ["Bits stored", image.bitsStored || 8],
        ["Pixel spacing", state.spacing ? `${state.spacing[0]} × ${state.spacing[1]} mm (${state.spacingSource})` : "Unknown"],
        ["Photometric", image.inverted ? "MONOCHROME1 (inverted for display)" : "Standard"],
      ]
    : [];
  $("meta").replaceChildren(...rows.flatMap(([k, v]) => [el("dt", {}, k), el("dd", {}, String(v))]));
  $("modality").value = state.modality;
}

function renderStatus() {
  const image = state.image;
  if (!image) return;
  $("status-image").textContent = `${state.modality} · ${image.width}×${image.height} · ${image.format === "dicom" ? `DICOM ${image.compression}` : "image"}`;
  $("status-window").textContent = `W ${Math.round(state.window.width)} · L ${Math.round(state.window.center)}`;
  $("status-zoom").textContent = `Zoom ${Math.round(viewer.scale * (1 / (window.devicePixelRatio || 1)) * 100)}%`;
}

function renderAll() {
  viewer.setOverlays({
    teeth: state.teeth,
    findings: state.findings,
    measurements: state.measurements.map((m) => ({ ...m, text: lengthText(m) })),
    selectedId: state.selected,
    showTeeth: state.showTeeth,
  });
  renderFindings();
  renderAddForm();
  renderChart();
  renderReport();
  renderMeasurements();
  renderMeta();
  renderAi();
  renderStatus();
  $("invert").setAttribute("aria-pressed", String(state.invert));
  $("enhance").setAttribute("aria-pressed", String(state.enhance));
  $("show-teeth").setAttribute("aria-pressed", String(state.showTeeth));
}

// ---------------------------------------------------------------- tabs, tools, controls

const TABS = ["findings", "chart", "report", "image"];
function showTab(name) {
  for (const t of TABS) {
    $(`tab-${t}`).setAttribute("aria-selected", String(t === name));
    $(`tab-${t}`).tabIndex = t === name ? 0 : -1;
    $(`panel-${t}`).hidden = t !== name;
  }
}
TABS.forEach((t, i) => {
  $(`tab-${t}`).addEventListener("click", () => showTab(t));
  $(`tab-${t}`).addEventListener("keydown", (e) => {
    const step = e.key === "ArrowRight" ? 1 : e.key === "ArrowLeft" ? -1 : 0;
    if (!step) return;
    const next = TABS[(i + step + TABS.length) % TABS.length];
    showTab(next);
    $(`tab-${next}`).focus();
  });
});

function setTool(tool) {
  viewer.setTool(tool);
  for (const b of document.querySelectorAll(".tool")) b.setAttribute("aria-checked", String(b.dataset.tool === tool));
}
for (const b of document.querySelectorAll(".tool")) b.addEventListener("click", () => setTool(b.dataset.tool));

for (const id of ["file-input", "file-input-empty"]) {
  $(id).addEventListener("change", (e) => {
    open(e.target.files[0]);
    e.target.value = "";
  });
}

$("zoom-in").addEventListener("click", () => viewer.zoomBy(1.25));
$("zoom-out").addEventListener("click", () => viewer.zoomBy(0.8));
$("zoom-fit").addEventListener("click", () => viewer.fit());
$("invert").addEventListener("click", () => {
  state.invert = !state.invert;
  updateDisplay();
  renderAll();
});
$("enhance").addEventListener("click", () => {
  state.enhance = !state.enhance;
  updateDisplay();
  renderAll();
});
$("show-teeth").addEventListener("click", () => {
  state.showTeeth = !state.showTeeth;
  renderAll();
});
$("wc").addEventListener("input", (e) => {
  state.window.center = Number(e.target.value);
  scheduleDisplay();
});
$("ww").addEventListener("input", (e) => {
  state.window.width = Math.max(1.01, Number(e.target.value));
  scheduleDisplay();
});
$("window-reset").addEventListener("click", () => {
  if (!state.image) return;
  state.window = { ...state.image.defaultWindow };
  updateDisplay();
});
$("modality").addEventListener("change", (e) => {
  state.modality = e.target.value;
  if (state.toothDetections.length || state.otherDetections.length) applyDetections();
  renderAll();
});
$("ai-run").addEventListener("click", runAi);
$("add-form").addEventListener("submit", addFinding);
$("add-cancel").addEventListener("click", () => {
  state.pendingBox = null;
  renderAddForm();
});
$("impression").addEventListener("input", renderReport);
$("report").addEventListener("input", () => {
  state.reportEdited = true;
  renderReport();
});
$("report-regenerate").addEventListener("click", () => {
  state.reportEdited = false;
  renderReport();
});
$("report-copy").addEventListener("click", async () => {
  try {
    await navigator.clipboard.writeText($("report").value);
    toast("Report copied.");
  } catch {
    $("report").select();
    toast("Press Ctrl+C (⌘C) to copy the selected report.");
  }
});
$("report-download").addEventListener("click", () => download(new Blob([$("report").value], { type: "text/plain" }), exportName("txt")));
$("calibrate-form").addEventListener("submit", (e) => {
  e.preventDefault();
  const m = state.measurements.find((x) => x.id === state.selected);
  const mm = Number($("calibrate-mm").value);
  if (!m || !(mm > 0)) return;
  const px = Math.hypot(m.x2 - m.x1, m.y2 - m.y1);
  state.spacing = [mm / px, mm / px];
  state.spacingSource = "calibrated";
  toast(`Calibrated: ${(mm / px).toFixed(4)} mm per pixel.`);
  renderAll();
});

$("export-png").addEventListener("click", async () => {
  if (!state.image) return;
  busy("Preparing image…");
  await nextFrame();
  try {
    download(await viewer.exportPng(), exportName("png"));
  } finally {
    busy(null);
  }
});
$("export-json").addEventListener("click", () => {
  if (!state.image) return;
  const image = state.image;
  const data = {
    tool: "Dental X-ray Reader (browser)",
    exported: new Date().toISOString(),
    image: { width: image.width, height: image.height, radiograph_type: state.modality, format: image.format, compression: image.compression, pixel_spacing_mm: state.spacing, pixel_spacing_source: state.spacingSource },
    ai_model: state.ai.manifest ? { name: state.ai.manifest.name, exported: state.ai.manifest.exported, classes: state.ai.manifest.classes } : null,
    teeth: state.teeth,
    findings: state.findings.map(({ edited, ...f }) => f),
    measurements: state.measurements.map((m) => ({ ...m, length: lengthText(m) })),
    report: $("report").value,
  };
  download(new Blob([JSON.stringify(data, null, 2)], { type: "application/json" }), exportName("json"));
});
$("print-report").addEventListener("click", async () => {
  if (!state.image) return;
  busy("Preparing report…");
  await nextFrame();
  const url = URL.createObjectURL(await viewer.exportPng());
  const img = $("print-image");
  img.onload = () => {
    busy(null);
    window.print();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  };
  $("print-text").textContent = $("report").value;
  img.src = url;
});

$("about-open").addEventListener("click", () => $("about").showModal());

// drag and drop
let dragDepth = 0;
window.addEventListener("dragenter", (e) => {
  if (![...(e.dataTransfer?.types || [])].includes("Files")) return;
  dragDepth += 1;
  $("drop-hint").hidden = false;
});
window.addEventListener("dragleave", () => {
  dragDepth = Math.max(0, dragDepth - 1);
  if (!dragDepth) $("drop-hint").hidden = true;
});
window.addEventListener("dragover", (e) => e.preventDefault());
window.addEventListener("drop", (e) => {
  e.preventDefault();
  dragDepth = 0;
  $("drop-hint").hidden = true;
  open(e.dataTransfer?.files?.[0]);
});

// keyboard shortcuts
window.addEventListener("keydown", (e) => {
  if (e.target.closest("input, textarea, select, dialog") || e.ctrlKey || e.metaKey || e.altKey) return;
  const key = e.key.toLowerCase();
  const actions = {
    v: () => setTool("pan"),
    w: () => setTool("window"),
    m: () => setTool("measure"),
    f: () => setTool("box"),
    0: () => viewer.fit(),
    "+": () => viewer.zoomBy(1.25),
    "=": () => viewer.zoomBy(1.25),
    "-": () => viewer.zoomBy(0.8),
    i: () => $("invert").click(),
    e: () => $("enhance").click(),
    t: () => $("show-teeth").click(),
    o: () => $("file-input").click(),
    delete: removeSelected,
    backspace: removeSelected,
    escape: () => {
      setTool("pan");
      state.pendingBox = null;
      select(null);
    },
  };
  if (actions[key]) {
    e.preventDefault();
    actions[key]();
  }
});

window.addEventListener("beforeunload", (e) => {
  if (hasWork()) e.preventDefault();
});

// ---------------------------------------------------------------- start

$("add-type").replaceChildren(...typeOptions("caries"));
$("add-tooth").replaceChildren(...toothOptions(null));
showTab("findings");
renderAll();
initAi();
