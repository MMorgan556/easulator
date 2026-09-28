// Canvas viewer: zoom/pan (mouse, trackpad, touch), window/level, measuring and box drawing.
// Coordinates exposed to the app are always image pixels.

import { PATHOLOGY } from "./findings.js";

const TOOL_CURSORS = { pan: "grab", window: "ns-resize", measure: "crosshair", box: "crosshair" };

export class Viewer {
  constructor(canvas, { onWindow, onMeasure, onBox, onSelect } = {}) {
    this.canvas = canvas;
    this.ctx = canvas.getContext("2d");
    this.callbacks = { onWindow, onMeasure, onBox, onSelect };
    this.bitmap = null; // offscreen canvas with the current display image
    this.image = null;
    this.scale = 1;
    this.offsetX = 0;
    this.offsetY = 0;
    this.tool = "pan";
    this.overlays = { teeth: [], findings: [], measurements: [], selectedId: null, showTeeth: true, spacing: null };
    this.drag = null;
    this.pointers = new Map();
    this.frame = 0;

    const observer = new ResizeObserver(() => this.resize());
    observer.observe(canvas.parentElement);
    canvas.addEventListener("pointerdown", (e) => this.pointerDown(e));
    canvas.addEventListener("pointermove", (e) => this.pointerMove(e));
    canvas.addEventListener("pointerup", (e) => this.pointerUp(e));
    canvas.addEventListener("pointercancel", (e) => this.pointerUp(e));
    canvas.addEventListener("wheel", (e) => this.wheel(e), { passive: false });
    canvas.addEventListener("dblclick", () => this.fit());
    this.setTool("pan");
  }

  setImage(image, display) {
    this.image = image;
    this.setDisplay(display);
    this.fit();
  }

  setDisplay(display) {
    const { width, height } = this.image;
    // One RGBA buffer and canvas per image size, rewritten in place on every update.
    if (!this.bitmap || this.bitmap.width !== width || this.bitmap.height !== height) {
      this.bitmap = new OffscreenCanvas(width, height);
      this.bitmapCtx = this.bitmap.getContext("2d");
      this.imageData = new ImageData(width, height);
      this.imageData.data.fill(255); // alpha stays opaque
    }
    const rgba = this.imageData.data;
    for (let i = 0, j = 0; i < display.length; i++, j += 4) rgba[j] = rgba[j + 1] = rgba[j + 2] = display[i];
    this.bitmapCtx.putImageData(this.imageData, 0, 0);
    this.requestDraw();
  }

  setOverlays(overlays) {
    Object.assign(this.overlays, overlays);
    this.requestDraw();
  }

  setTool(tool) {
    this.tool = tool;
    this.canvas.style.cursor = TOOL_CURSORS[tool] || "default";
  }

  resize() {
    const rect = this.canvas.parentElement.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    this.canvas.width = Math.max(1, Math.round(rect.width * dpr));
    this.canvas.height = Math.max(1, Math.round(rect.height * dpr));
    this.canvas.style.width = `${rect.width}px`;
    this.canvas.style.height = `${rect.height}px`;
    if (this.image && !this.userMoved) this.fit();
    else this.requestDraw();
  }

  fit() {
    if (!this.image) return;
    const pad = 16 * (window.devicePixelRatio || 1);
    const w = this.canvas.width - pad * 2;
    const h = this.canvas.height - pad * 2;
    this.scale = Math.max(0.01, Math.min(w / this.image.width, h / this.image.height));
    this.offsetX = (this.canvas.width - this.image.width * this.scale) / 2;
    this.offsetY = (this.canvas.height - this.image.height * this.scale) / 2;
    this.userMoved = false;
    this.requestDraw();
  }

  zoomBy(factor, cx = this.canvas.width / 2, cy = this.canvas.height / 2) {
    if (!this.image) return;
    const next = Math.min(40, Math.max(0.02, this.scale * factor));
    const ix = (cx - this.offsetX) / this.scale;
    const iy = (cy - this.offsetY) / this.scale;
    this.scale = next;
    this.offsetX = cx - ix * next;
    this.offsetY = cy - iy * next;
    this.userMoved = true;
    this.requestDraw();
  }

  toImage(e) {
    const rect = this.canvas.getBoundingClientRect();
    const dpr = this.canvas.width / rect.width;
    const cx = (e.clientX - rect.left) * dpr;
    const cy = (e.clientY - rect.top) * dpr;
    return { cx, cy, x: (cx - this.offsetX) / this.scale, y: (cy - this.offsetY) / this.scale };
  }

  clampToImage(p) {
    return { x: Math.min(Math.max(p.x, 0), this.image.width), y: Math.min(Math.max(p.y, 0), this.image.height) };
  }

  hitFinding(p) {
    const hits = this.overlays.findings.filter((f) => f.box && f.status !== "rejected" && p.x >= f.box.x1 && p.x <= f.box.x2 && p.y >= f.box.y1 && p.y <= f.box.y2);
    hits.sort((a, b) => (a.box.x2 - a.box.x1) * (a.box.y2 - a.box.y1) - (b.box.x2 - b.box.x1) * (b.box.y2 - b.box.y1));
    return hits[0] || null;
  }

  pointerDown(e) {
    if (!this.image) return;
    this.canvas.setPointerCapture(e.pointerId);
    this.pointers.set(e.pointerId, this.toImage(e));
    if (this.pointers.size === 2) {
      const [a, b] = [...this.pointers.values()];
      this.drag = { mode: "pinch", distance: Math.hypot(a.cx - b.cx, a.cy - b.cy) };
      return;
    }
    const p = this.toImage(e);
    const panning = this.tool === "pan" || e.button === 1 || (e.button === 0 && e.altKey);
    if (e.button === 2 || (this.tool === "window" && e.button === 0)) {
      this.drag = { mode: "window", startX: e.clientX, startY: e.clientY };
    } else if (panning) {
      this.drag = { mode: "pan", cx: p.cx, cy: p.cy, moved: false, start: p };
      this.canvas.style.cursor = "grabbing";
    } else if (this.tool === "measure" || this.tool === "box") {
      const start = this.clampToImage(p);
      this.drag = { mode: this.tool, start, end: start };
    }
  }

  pointerMove(e) {
    if (!this.drag) return;
    const p = this.toImage(e);
    if (this.pointers.has(e.pointerId)) this.pointers.set(e.pointerId, p);
    if (this.drag.mode === "pinch" && this.pointers.size === 2) {
      const [a, b] = [...this.pointers.values()];
      const distance = Math.hypot(a.cx - b.cx, a.cy - b.cy);
      this.zoomBy(distance / this.drag.distance, (a.cx + b.cx) / 2, (a.cy + b.cy) / 2);
      this.drag.distance = distance;
    } else if (this.drag.mode === "pan") {
      this.offsetX += p.cx - this.drag.cx;
      this.offsetY += p.cy - this.drag.cy;
      if (Math.hypot(p.cx - this.drag.start.cx, p.cy - this.drag.start.cy) > 3) this.drag.moved = true;
      this.drag.cx = p.cx;
      this.drag.cy = p.cy;
      this.userMoved = true;
      this.requestDraw();
    } else if (this.drag.mode === "window") {
      this.callbacks.onWindow?.(e.clientX - this.drag.startX, e.clientY - this.drag.startY);
      this.drag.startX = e.clientX;
      this.drag.startY = e.clientY;
    } else if (this.drag.mode === "measure" || this.drag.mode === "box") {
      this.drag.end = this.clampToImage(p);
      this.requestDraw();
    }
  }

  pointerUp(e) {
    this.pointers.delete(e.pointerId);
    const drag = this.drag;
    if (this.pointers.size === 0) this.drag = null;
    if (!drag) return;
    this.setTool(this.tool);
    if (drag.mode === "pan" && !drag.moved) {
      this.callbacks.onSelect?.(this.hitFinding(drag.start));
    } else if (drag.mode === "measure") {
      const { start, end } = drag;
      if (Math.hypot(end.x - start.x, end.y - start.y) * this.scale > 4) this.callbacks.onMeasure?.({ x1: start.x, y1: start.y, x2: end.x, y2: end.y });
    } else if (drag.mode === "box") {
      const box = {
        x1: Math.min(drag.start.x, drag.end.x),
        y1: Math.min(drag.start.y, drag.end.y),
        x2: Math.max(drag.start.x, drag.end.x),
        y2: Math.max(drag.start.y, drag.end.y),
      };
      if ((box.x2 - box.x1) * this.scale > 4 && (box.y2 - box.y1) * this.scale > 4) this.callbacks.onBox?.(box);
    }
    this.requestDraw();
  }

  wheel(e) {
    if (!this.image) return;
    e.preventDefault();
    const p = this.toImage(e);
    const factor = Math.exp(-e.deltaY * (e.ctrlKey ? 0.01 : 0.0015));
    this.zoomBy(factor, p.cx, p.cy);
  }

  requestDraw() {
    if (this.frame) return;
    this.frame = requestAnimationFrame(() => {
      this.frame = 0;
      this.draw();
    });
  }

  /** Draws into this.ctx (screen) or into a supplied context at image scale (export). */
  draw(target = null) {
    const ctx = target?.ctx || this.ctx;
    const scale = target ? target.scale : this.scale;
    const ox = target ? 0 : this.offsetX;
    const oy = target ? 0 : this.offsetY;
    const dpr = target ? target.scale * Math.max(1, this.image.width / 1600) : window.devicePixelRatio || 1;
    if (!target) {
      ctx.setTransform(1, 0, 0, 1, 0, 0);
      ctx.fillStyle = "#0b0d10";
      ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);
    }
    if (!this.image || !this.bitmap) return;
    ctx.imageSmoothingEnabled = scale < 2;
    ctx.imageSmoothingQuality = "high";
    ctx.drawImage(this.bitmap, ox, oy, this.image.width * scale, this.image.height * scale);

    const X = (x) => ox + x * scale;
    const Y = (y) => oy + y * scale;
    const line = 1.5 * dpr;
    const font = 12 * dpr;
    ctx.font = `600 ${font}px system-ui, sans-serif`;
    ctx.textBaseline = "bottom";

    const label = (text, x, y, color) => {
      const pad = 3 * dpr;
      const w = ctx.measureText(text).width + pad * 2;
      const h = font + pad * 2;
      const top = y - h < 0 ? y + h : y;
      ctx.setLineDash([]);
      ctx.fillStyle = color;
      ctx.fillRect(x, top - h, w, h);
      ctx.fillStyle = "#0b0d10";
      ctx.fillText(text, x + pad, top - pad);
    };
    const rect = (b, color, dashed, width = line) => {
      ctx.setLineDash(dashed ? [5 * dpr, 4 * dpr] : []);
      ctx.strokeStyle = color;
      ctx.lineWidth = width;
      ctx.strokeRect(X(b.x1), Y(b.y1), (b.x2 - b.x1) * scale, (b.y2 - b.y1) * scale);
    };

    const o = this.overlays;
    if (o.showTeeth) {
      for (const t of o.teeth) {
        rect(t.box, "rgba(160,172,186,0.9)", false, line * 0.8);
        if (t.fdi !== "?") label(t.fdi, X(t.box.x1), Y(t.box.y1), "rgba(160,172,186,0.95)");
      }
    }
    for (const f of o.findings) {
      if (!f.box || f.status === "rejected") continue;
      const color = PATHOLOGY.has(f.type) ? "#ff5c61" : "#3ea6ff";
      const selected = f.id === o.selectedId;
      rect(f.box, color, f.status === "pending", selected ? line * 2.2 : line);
      if (selected) {
        ctx.setLineDash([]);
        ctx.strokeStyle = "#ffd166";
        ctx.lineWidth = line;
        ctx.strokeRect(X(f.box.x1) - 3 * dpr, Y(f.box.y1) - 3 * dpr, (f.box.x2 - f.box.x1) * scale + 6 * dpr, (f.box.y2 - f.box.y1) * scale + 6 * dpr);
      }
      label(f.id, X(f.box.x1), Y(f.box.y1), color);
    }
    const measure = (m, text) => {
      ctx.setLineDash([]);
      ctx.strokeStyle = "#ffd166";
      ctx.lineWidth = line;
      ctx.beginPath();
      ctx.moveTo(X(m.x1), Y(m.y1));
      ctx.lineTo(X(m.x2), Y(m.y2));
      ctx.stroke();
      for (const [x, y] of [[m.x1, m.y1], [m.x2, m.y2]]) {
        ctx.beginPath();
        ctx.arc(X(x), Y(y), 2.5 * dpr, 0, Math.PI * 2);
        ctx.fillStyle = "#ffd166";
        ctx.fill();
      }
      if (text) label(text, X((m.x1 + m.x2) / 2) + 6 * dpr, Y((m.y1 + m.y2) / 2), "#ffd166");
    };
    for (const m of o.measurements) measure(m, `${m.id} ${m.text}`);

    if (!target && this.drag && (this.drag.mode === "measure" || this.drag.mode === "box")) {
      const { start, end } = this.drag;
      if (this.drag.mode === "measure") measure({ x1: start.x, y1: start.y, x2: end.x, y2: end.y }, this.callbacks.measureText?.(start, end));
      else rect({ x1: Math.min(start.x, end.x), y1: Math.min(start.y, end.y), x2: Math.max(start.x, end.x), y2: Math.max(start.y, end.y) }, "#ffd166", true);
    }
  }

  /** Full-resolution PNG of the image with overlays. */
  async exportPng() {
    const canvas = new OffscreenCanvas(this.image.width, this.image.height);
    this.draw({ ctx: canvas.getContext("2d"), scale: 1 });
    return canvas.convertToBlob({ type: "image/png" });
  }
}
