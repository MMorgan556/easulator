"""HTTP API and web viewer. Run with: uvicorn app.main:app --reload"""

from __future__ import annotations

import hmac
from functools import lru_cache
from pathlib import Path

from fastapi import Depends, FastAPI, File, Header, HTTPException, Query, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse

from . import __version__
from .config import get_settings
from .detection import build_detector
from .ingest import IngestError
from .pipeline import Pipeline
from .report import ReportGenerationError, build_report_generator
from .schemas import AnalysisResult, FindingType

STATIC_DIR = Path(__file__).parent / "static"

app = FastAPI(
    title="Dental X-ray Reader",
    version=__version__,
    description="AI-assisted dental radiograph analysis. Clinical decision support only.",
)


@lru_cache
def get_pipeline() -> Pipeline:
    settings = get_settings()
    return Pipeline(settings, build_detector(settings), build_report_generator(settings))


def check_access(
    pipeline: Pipeline = Depends(get_pipeline),
    x_access_code: str | None = Header(default=None),
) -> None:
    settings = pipeline.settings
    problem = settings.access_code_problem()
    if problem:
        # Fail closed rather than serve patient images without protection.
        raise HTTPException(status_code=503, detail=f"Server is not configured: {problem}")
    if not settings.access_code:
        return
    supplied = (x_access_code or "").strip().encode()
    if not hmac.compare_digest(supplied, settings.access_code.encode()):
        raise HTTPException(status_code=401, detail="Missing or incorrect access code")


@app.get("/", include_in_schema=False)
def viewer() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health")
def health(pipeline: Pipeline = Depends(get_pipeline)) -> dict:
    settings = pipeline.settings
    return {
        "status": "ok",
        "detector": pipeline.detector.name,
        "report_generator": pipeline.reporter.name,
        "access_code_required": bool(settings.access_code or settings.require_access_code),
        "configuration_problem": settings.access_code_problem(),
    }


@app.get("/classes")
def classes() -> list[str]:
    return [t.value for t in FindingType]


@app.post("/analyze", response_model=AnalysisResult, dependencies=[Depends(check_access)])
async def analyze(
    file: UploadFile = File(..., description="DICOM (.dcm), PNG, JPEG or TIFF dental radiograph"),
    report: bool = Query(True, description="Generate a written draft report"),
    preview: bool = Query(False, description="Include a downscaled PNG of the image for display"),
    pipeline: Pipeline = Depends(get_pipeline),
) -> AnalysisResult:
    limit = pipeline.settings.max_upload_bytes
    data = await file.read(limit + 1)
    if len(data) > limit:
        raise HTTPException(status_code=413, detail=f"Upload exceeds {limit // (1024 * 1024)} MB")
    try:
        # Decoding, inference and the report call are blocking; keep them off the event loop.
        return await run_in_threadpool(pipeline.analyze, data, with_report=report, with_preview=preview)
    except IngestError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except ReportGenerationError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
