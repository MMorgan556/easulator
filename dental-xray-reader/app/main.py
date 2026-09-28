"""HTTP API and web viewer. Run with: uvicorn app.main:app --reload"""

from __future__ import annotations

import hmac
from functools import lru_cache
from pathlib import Path

from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse
from starlette.datastructures import UploadFile

from . import __version__
from .config import get_settings
from .detection import build_detector
from .pipeline import Pipeline
from .report import ReportGenerationError, build_report_generator
from .schemas import AnalysisResult, FindingType

STATIC_DIR = Path(__file__).parent / "static"
# Room for multipart boundaries and part headers on top of the file itself.
MULTIPART_OVERHEAD_BYTES = 64 * 1024

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


async def read_upload(request: Request, limit: int) -> bytes:
    """Read the multipart ``file`` field, refusing oversized bodies as they arrive.

    The body is read here rather than through a ``File(...)`` parameter because FastAPI
    parses body parameters before running dependencies, which would let anyone make the
    server spool an arbitrarily large upload before the access check rejects it.
    """
    too_large = HTTPException(status_code=413, detail=f"Upload exceeds {limit // (1024 * 1024)} MB")
    body_limit = limit + MULTIPART_OVERHEAD_BYTES
    declared = request.headers.get("content-length")
    if declared is not None:
        if not declared.isdigit():
            raise HTTPException(status_code=400, detail="Invalid Content-Length header")
        if int(declared) > body_limit:
            raise too_large

    received = 0

    async def counted_receive() -> dict:
        # Enforces the limit even when no Content-Length is sent (chunked uploads).
        nonlocal received
        message = await request.receive()
        if message["type"] == "http.request":
            received += len(message.get("body", b""))
            if received > body_limit:
                raise too_large
        return message

    # Starlette's parser spools file parts to disk past 1 MB, so the body is never all in memory.
    form = await Request(request.scope, counted_receive).form(max_files=1, max_fields=10)
    try:
        upload = form.get("file")
        if not isinstance(upload, UploadFile):
            raise HTTPException(status_code=422, detail="Send the X-ray as a multipart form field named 'file'")
        data = await upload.read()
    finally:
        await form.close()
    if len(data) > limit:
        raise too_large
    return data


UPLOAD_SCHEMA = {
    "requestBody": {
        "required": True,
        "content": {
            "multipart/form-data": {
                "schema": {
                    "type": "object",
                    "required": ["file"],
                    "properties": {
                        "file": {
                            "type": "string",
                            "format": "binary",
                            "description": "DICOM (.dcm), PNG, JPEG or TIFF dental radiograph",
                        }
                    },
                }
            }
        },
    }
}


@app.post(
    "/analyze",
    response_model=AnalysisResult,
    dependencies=[Depends(check_access)],
    openapi_extra=UPLOAD_SCHEMA,
)
async def analyze(
    request: Request,
    report: bool = Query(True, description="Generate a written draft report"),
    preview: bool = Query(False, description="Include a downscaled PNG of the image for display"),
    pipeline: Pipeline = Depends(get_pipeline),
) -> AnalysisResult:
    # Image stage: bounded, so concurrent uploads can't exhaust memory. Requests waiting here
    # have not read their body yet and hold no worker thread.
    async with pipeline.image_slots:
        data = await read_upload(request, pipeline.settings.max_upload_bytes)
        # Decoding and inference are blocking; keep them off the event loop.
        exam = await run_in_threadpool(pipeline.examine_or_reason, data, with_preview=preview)
        del data
    if isinstance(exam, str):
        raise HTTPException(status_code=422, detail=exam)

    # Report stage: holds only the small structured findings, bounded separately so slow
    # Claude calls can't use up the worker threads that other endpoints need.
    written = None
    if report:
        async with pipeline.report_slots:
            try:
                written = await run_in_threadpool(pipeline.write_report, exam)
            except ReportGenerationError as exc:
                raise HTTPException(status_code=502, detail=str(exc)) from exc
    return pipeline.result(exam, written)
