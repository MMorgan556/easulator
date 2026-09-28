# Dental X-ray Reader

An AI-assisted dental radiograph analysis service. You upload a DICOM or image X-ray and get back numbered teeth, findings linked to teeth, and a written draft report.

> **Clinical decision support only.** This is not a certified medical device. Every finding must be reviewed by a licensed dentist, and clinical use requires regulatory clearance (FDA 510(k), CE MDR, or your local equivalent).

## How it works

```
upload (.dcm / .png / .jpg / .tif)
  │
  ├─ ingest.py      decode DICOM (compressed or not; rescale, DICOM window, MONOCHROME1),
  │                 read pixel spacing, honour EXIF rotation, guess modality
  │                 (panoramic / bitewing / periapical); no patient tags leave this step
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

Open http://localhost:8000 for the web viewer (upload an X-ray, see boxes, findings and the report), http://localhost:8000/docs for the interactive API, or run:

```bash
curl -F "file=@panoramic.dcm" http://localhost:8000/analyze
```

Out of the box it runs with `DETECTOR_BACKEND=demo`, which returns **synthetic placeholder findings** so you can exercise the whole pipeline and API before you have trained weights. Every demo response carries a `DEMO MODE` warning.

### Claude reports

Set `ANTHROPIC_API_KEY` and reports are written by Claude (`claude-opus-5` by default). Without a key, the service uses the built-in template report. If a Claude request fails in `auto` mode, the service falls back to the template and says so in `warnings`.

## Deploy to Render

The repository root has a `render.yaml` Blueprint that runs this service as a free Docker web service with a password.

1. Open https://dashboard.render.com/blueprints, click **New Blueprint Instance**, and connect this GitHub repository. Alternatively, open `https://render.com/deploy?repo=<your repository URL>`.
2. When Render asks for **ACCESS_CODE**, enter a password of at least 12 characters. Share it only with the people who should upload X-rays.
3. Click **Apply**. The first build takes a few minutes, and Render then shows the site's `https://…onrender.com` URL.
4. Optional: to have Claude write the reports, add an `ANTHROPIC_API_KEY` environment variable in the service's **Environment** tab.

Every merge to `main` that changes `dental-xray-reader/` redeploys automatically.

Uploads are refused (HTTP 503) until `ACCESS_CODE` is set to 12 or more characters, so the site cannot accidentally go live without a password. `/health` reports any such configuration problem. On Render's free plan, the service sleeps after 15 minutes without traffic, and the first request after that takes about a minute.

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
| `MAX_UPLOAD_BYTES` | `33554432` | Upload size limit (the Render config sets 25 MB) |
| `MAX_CONCURRENT_IMAGES` | `1` | Uploads read and decoded at the same time; others wait without using memory. Each slot needs about 250 MB for a 20 MP image, so raise it only on larger machines |
| `MAX_CONCURRENT_REPORTS` | `8` | Reports written at the same time |
| `ACCESS_CODE` | unset | When set, `/analyze` requires it in the `X-Access-Code` header |
| `REQUIRE_ACCESS_CODE` | `false` | Refuse uploads unless `ACCESS_CODE` is 12+ characters (`true` on Render) |

## Training a real detector

1. Collect radiographs. Public starting points include DENTEX (panoramic, MICCAI 2023), the Tufts Dental Database, and dental datasets on Roboflow Universe.
2. Export them exactly as the service sees them, then label the exported PNGs (not the originals) in YOLO format, for example with CVAT or Label Studio:
   ```bash
   python scripts/prepare_dataset.py raw_xrays/train datasets/dental/images/train
   ```
   This applies the same DICOM windowing, inversion, contrast stretch and EXIF rotation as uploads, so the model trains on what it will see in production. Have several dentists label the same images so you can measure how much they agree.
3. Put the labels in `datasets/dental/labels/{train,val,test}` as described in `configs/dental.yaml`. Class names must match the list above.
4. Train:
   ```bash
   pip install -r requirements-ml.txt
   python scripts/train_yolo.py --data configs/dental.yaml --model yolo11m.pt --epochs 150
   ```
   Horizontal flips are disabled because they swap the patient's left and right, which breaks FDI numbering.
5. Serve it:
   ```bash
   DETECTOR_BACKEND=yolo YOLO_WEIGHTS=weights/dental-yolo.pt uvicorn app.main:app
   ```

## API

| Method | Path | Description |
|---|---|---|
| `GET` | `/` | Web viewer |
| `POST` | `/analyze?report=true&preview=false` | Multipart `file` upload plus an `X-Access-Code` header when enabled; returns `AnalysisResult` JSON. `preview=true` adds a downscaled PNG for display |
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

`study_id` is a hash of the upload's contents. The service stores nothing and never returns DICOM patient tags. Compressed DICOM (JPEG baseline/lossless, JPEG 2000, RLE) is supported. Images larger than 20 megapixels are rejected, which keeps memory use within Render's 512 MB free plan.

## Known limitations and next steps

- **FDI numbering** uses a geometric rule on panoramic images. It assumes the full dentition is present, so missing teeth shift the numbers. On bitewing and periapical images, teeth are returned with `fdi: "?"`. The next step is a dedicated numbering model, for example a detector trained with one class per FDI number, as in the DENTEX challenge.
- **Segmentation**: add nnU-Net or MONAI for pixel-accurate caries depth and bone-level measurements; `pixel_spacing_mm` is already passed through for mm measurements.
- **Explainability**: Grad-CAM overlays and uncertainty estimates (ensembles or MC dropout).
- **Integration**: Orthanc/PACS ingestion, an OHIF viewer with overlays, and a queue for asynchronous processing.
- **Compliance**: per-user accounts instead of a shared access code, audit logging, a signed business associate agreement (BAA) with the host before handling real patient data, and an IEC 62304 / ISO 13485 / ISO 14971 process before any clinical use.
