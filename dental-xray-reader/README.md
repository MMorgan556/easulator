# Dental X-ray Reader

An AI-assisted dental radiograph analysis service. You upload a DICOM or image X-ray and get back numbered teeth, findings linked to teeth, and a written draft report.

> **Clinical decision support only.** This is not a certified medical device. Every finding must be reviewed by a licensed dentist, and clinical use requires regulatory clearance (FDA 510(k), CE MDR, or your local equivalent).

## How it works

```
upload (.dcm / .png / .jpg / .tif)
  │
  ├─ ingest.py      decode DICOM (modality/VOI LUT, MONOCHROME1), read pixel spacing,
  │                 guess modality (panoramic / bitewing / periapical); no patient tags leave this step
  ├─ preprocess.py  contrast stretch + clipped histogram equalization, letterbox to model size
  ├─ detection.py   YOLO detector (teeth + 9 finding classes), or demo placeholder output
  ├─ postprocess.py per-class NMS, FDI tooth numbering, link findings to teeth,
  │                 flag low-confidence findings for review
  └─ report.py      Claude writes a draft report from the structured findings JSON
                    (never the image), with a template fallback
```

The vision model decides what is on the image. The language model only writes up the structured findings, so it cannot add findings of its own.

### Finding classes

`tooth`, `caries`, `periapical_lesion`, `restoration`, `crown`, `root_canal_treatment`, `implant`, `impacted_tooth`, `bone_loss`, `calculus`

## Quick start

```bash
cd dental-xray-reader
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
pytest
uvicorn app.main:app --reload
```

Open http://localhost:8000/docs, or run:

```bash
curl -F "file=@panoramic.dcm" http://localhost:8000/analyze
```

Out of the box it runs with `DETECTOR_BACKEND=demo`, which returns **synthetic placeholder findings** so you can exercise the whole pipeline and API before you have trained weights. Every demo response carries a `DEMO MODE` warning.

### Claude reports

Set `ANTHROPIC_API_KEY` and reports are written by Claude (`claude-opus-5` by default). Without a key, the service uses the built-in template report. If a Claude request fails in `auto` mode, the service falls back to the template and says so in `warnings`.

## Configuration

| Variable | Default | Meaning |
|---|---|---|
| `DETECTOR_BACKEND` | `demo` | `demo` (placeholder output) or `yolo` |
| `YOLO_WEIGHTS` | `weights/dental-yolo.pt` | Ultralytics checkpoint for `yolo` |
| `REPORT_BACKEND` | `auto` | `auto` (Claude if credentials set), `claude` (required; errors return 502), `template` |
| `CLAUDE_MODEL` | `claude-opus-5` | Model used for report writing |
| `MIN_CONFIDENCE` | `0.25` | Detections below this are dropped |
| `REVIEW_CONFIDENCE` | `0.6` | Findings below this are flagged `needs_review` |
| `MODEL_INPUT_SIZE` | `1024` | Detector input resolution |
| `MAX_UPLOAD_BYTES` | `67108864` | Upload size limit |

## Training a real detector

1. Collect and label radiographs in YOLO format, for example with CVAT or Label Studio. Public starting points include DENTEX (panoramic, MICCAI 2023), the Tufts Dental Database, and dental datasets on Roboflow Universe. Have several dentists label the same images so you can measure how much they agree.
2. Lay the data out as described in `configs/dental.yaml`. Class names must match the list above.
3. Train:
   ```bash
   pip install -r requirements-ml.txt
   python scripts/train_yolo.py --data configs/dental.yaml --model yolo11m.pt --epochs 150
   ```
   Horizontal flips are disabled because they swap the patient's left and right, which breaks FDI numbering.
4. Serve it:
   ```bash
   DETECTOR_BACKEND=yolo YOLO_WEIGHTS=weights/dental-yolo.pt uvicorn app.main:app
   ```

## API

| Method | Path | Description |
|---|---|---|
| `POST` | `/analyze?report=true` | Multipart `file` upload; returns `AnalysisResult` JSON |
| `GET` | `/health` | Active detector and report generator |
| `GET` | `/classes` | Supported finding types |

Response shape (abridged):

```json
{
  "study_id": "3f1c…",
  "image": {"width": 2800, "height": 1400, "modality": "panoramic", "source_format": "dicom", "pixel_spacing_mm": [0.1, 0.1]},
  "detector": "yolo",
  "teeth": [{"fdi": "16", "confidence": 0.97, "box": {"x1": 410, "y1": 300, "x2": 560, "y2": 690}}],
  "findings": [{"id": "F1", "type": "caries", "confidence": 0.82, "tooth": "16", "needs_review": false, "box": {…}}],
  "report": {"text": "…", "generator": "claude:claude-opus-5", "disclaimer": "…"},
  "warnings": []
}
```

`study_id` is a hash of the upload's contents. The service stores nothing and never returns DICOM patient tags.

## Known limitations and next steps

- **FDI numbering** uses a geometric rule on panoramic images. It assumes the full dentition is present, so missing teeth shift the numbers. On bitewing and periapical images, teeth are returned with `fdi: "?"`. The next step is a dedicated numbering model, for example a detector trained with one class per FDI number, as in the DENTEX challenge.
- **Segmentation**: add nnU-Net or MONAI for pixel-accurate caries depth and bone-level measurements; `pixel_spacing_mm` is already passed through for mm measurements.
- **Explainability**: Grad-CAM overlays and uncertainty estimates (ensembles or MC dropout).
- **Integration**: Orthanc/PACS ingestion, an OHIF viewer with overlays, and a queue for asynchronous processing.
- **Compliance**: audit logging, authentication, encryption at rest, and an IEC 62304 / ISO 13485 / ISO 14971 process before any clinical use.
