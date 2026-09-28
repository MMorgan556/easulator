"""HTTP API. Run with: uvicorn app.main:app --reload"""

from __future__ import annotations

from functools import lru_cache

from fastapi import Depends, FastAPI, File, HTTPException, Query, UploadFile

from . import __version__
from .config import get_settings
from .detection import build_detector
from .ingest import IngestError
from .pipeline import Pipeline
from .report import ReportGenerationError, build_report_generator
from .schemas import AnalysisResult, FindingType

app = FastAPI(
    title="Dental X-ray Reader",
    version=__version__,
    description="AI-assisted dental radiograph analysis. Clinical decision support only.",
)


@lru_cache
def get_pipeline() -> Pipeline:
    settings = get_settings()
    return Pipeline(settings, build_detector(settings), build_report_generator(settings))


@app.get("/health")
def health(pipeline: Pipeline = Depends(get_pipeline)) -> dict:
    return {"status": "ok", "detector": pipeline.detector.name, "report_generator": pipeline.reporter.name}


@app.get("/classes")
def classes() -> list[str]:
    return [t.value for t in FindingType]


@app.post("/analyze", response_model=AnalysisResult)
async def analyze(
    file: UploadFile = File(..., description="DICOM (.dcm), PNG, JPEG or TIFF dental radiograph"),
    report: bool = Query(True, description="Generate a written draft report"),
    pipeline: Pipeline = Depends(get_pipeline),
) -> AnalysisResult:
    limit = pipeline.settings.max_upload_bytes
    data = await file.read(limit + 1)
    if len(data) > limit:
        raise HTTPException(status_code=413, detail=f"Upload exceeds {limit} bytes")
    try:
        return pipeline.analyze(data, with_report=report)
    except IngestError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except ReportGenerationError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
