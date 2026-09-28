import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.detection import DemoDetector
from app.main import app, get_pipeline
from app.pipeline import Pipeline
from app.report import ReportGenerationError, TemplateReportGenerator

from .conftest import png_bytes


class FailingReporter:
    name = "claude"

    def generate(self, *args):
        raise ReportGenerationError("simulated outage")


def client_with(settings: Settings, reporter=None) -> TestClient:
    """Test client around a demo pipeline; defaults leave the access code off."""
    pipeline = Pipeline(settings, DemoDetector(), reporter or TemplateReportGenerator())
    app.dependency_overrides[get_pipeline] = lambda: pipeline
    return TestClient(app)


@pytest.fixture(autouse=True)
def clear_overrides():
    yield
    app.dependency_overrides.clear()


def test_health():
    body = client_with(Settings(access_code="", require_access_code=False)).get("/health").json()
    assert body == {
        "status": "ok",
        "detector": "demo",
        "report_generator": "template",
        "access_code_required": False,
        "configuration_problem": None,
    }


def test_viewer_page_is_served():
    resp = client_with(Settings()).get("/")
    assert resp.status_code == 200
    assert "Dental X-ray Reader" in resp.text


def test_preview_is_optional_and_downscaled(panoramic_png):
    import base64
    import io

    from PIL import Image

    client = client_with(Settings())
    assert client.post("/analyze", files={"file": ("p.png", panoramic_png)}).json()["preview_png"] is None

    big = png_bytes(3200, 1600)
    body = client.post("/analyze?preview=true", files={"file": ("p.png", big)}).json()
    preview = Image.open(io.BytesIO(base64.b64decode(body["preview_png"])))
    assert preview.size == (1600, 800)
    assert body["image"]["width"] == 3200  # boxes stay in full-resolution coordinates


STRONG_CODE = "correct-horse-battery"


@pytest.mark.parametrize(
    ("header", "status"),
    [(None, 401), ("wrong-code-entirely", 401), (STRONG_CODE, 200), (f"  {STRONG_CODE} ", 200)],
)
def test_access_code_is_enforced(panoramic_png, header, status):
    client = client_with(Settings(access_code=STRONG_CODE, require_access_code=True))
    headers = {"X-Access-Code": header} if header is not None else {}
    resp = client.post("/analyze", files={"file": ("p.png", panoramic_png)}, headers=headers)
    assert resp.status_code == status


@pytest.mark.parametrize(("code", "problem"), [("", "not set"), ("short", "at least 12")])
def test_required_access_code_fails_closed(panoramic_png, code, problem):
    client = client_with(Settings(access_code=code, require_access_code=True))
    resp = client.post("/analyze", files={"file": ("p.png", panoramic_png)}, headers={"X-Access-Code": code})
    assert resp.status_code == 503
    assert problem in resp.json()["detail"]
    health = client.get("/health").json()
    assert health["access_code_required"] is True
    assert problem in health["configuration_problem"]


def test_health_and_viewer_stay_open_when_protected():
    client = client_with(Settings(access_code=STRONG_CODE, require_access_code=True))
    assert client.get("/health").status_code == 200
    assert client.get("/").status_code == 200


def test_classes_lists_finding_types():
    assert "caries" in client_with(Settings()).get("/classes").json()


def test_analyze_png(panoramic_png):
    resp = client_with(Settings()).post("/analyze", files={"file": ("pan.png", panoramic_png, "image/png")})
    assert resp.status_code == 200
    body = resp.json()
    assert body["detector"] == "demo"
    assert body["image"]["modality"] == "panoramic"
    assert len(body["teeth"]) == 16
    assert {f["type"] for f in body["findings"]} == {"caries", "periapical_lesion"}
    assert any("DEMO MODE" in w for w in body["warnings"])
    assert body["report"]["generator"] == "template"
    assert len(body["study_id"]) == 16


def test_analyze_dicom(panoramic_dicom):
    resp = client_with(Settings()).post("/analyze", files={"file": ("pan.dcm", panoramic_dicom)})
    assert resp.status_code == 200
    assert resp.json()["image"]["source_format"] == "dicom"


def test_analyze_without_report(panoramic_png):
    resp = client_with(Settings()).post("/analyze?report=false", files={"file": ("p.png", panoramic_png)})
    assert resp.json()["report"] is None


def test_bitewing_warns_about_numbering():
    resp = client_with(Settings()).post("/analyze", files={"file": ("bw.png", png_bytes(1300, 1000))})
    assert any("FDI numbering" in w for w in resp.json()["warnings"])


def test_invalid_upload_returns_422():
    resp = client_with(Settings()).post("/analyze", files={"file": ("x.png", b"garbage")})
    assert resp.status_code == 422


def test_oversized_upload_returns_413(panoramic_png):
    resp = client_with(Settings(max_upload_bytes=100)).post("/analyze", files={"file": ("p.png", panoramic_png)})
    assert resp.status_code == 413


def test_auto_mode_falls_back_to_template_when_claude_fails(panoramic_png):
    client = client_with(Settings(report_backend="auto"), FailingReporter())
    body = client.post("/analyze", files={"file": ("p.png", panoramic_png)}).json()
    assert body["report"]["generator"] == "template"
    assert any("simulated outage" in w for w in body["warnings"])


def test_claude_mode_surfaces_failure(panoramic_png):
    client = client_with(Settings(report_backend="claude"), FailingReporter())
    resp = client.post("/analyze", files={"file": ("p.png", panoramic_png)})
    assert resp.status_code == 502


def _raw_post(headers: list[tuple[bytes, bytes]], chunks: list[bytes]) -> tuple[int, int]:
    """Call the ASGI app directly; returns (status, number of body chunks the app pulled)."""
    import asyncio

    pulled = []

    async def receive():
        index = len(pulled)
        pulled.append(index)
        if index < len(chunks):
            return {"type": "http.request", "body": chunks[index], "more_body": index < len(chunks) - 1}
        return {"type": "http.disconnect"}

    sent = []

    async def send(message):
        sent.append(message)

    scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": "/analyze",
        "raw_path": b"/analyze",
        "query_string": b"report=false",
        "root_path": "",
        "headers": [(b"content-type", b"multipart/form-data; boundary=xyz"), *headers],
        "client": ("127.0.0.1", 1),
        "server": ("testserver", 80),
    }
    asyncio.run(app(scope, receive, send))
    return sent[0]["status"], len(pulled)


def test_wrong_access_code_rejected_before_body_is_read():
    client_with(Settings(access_code=STRONG_CODE, require_access_code=True))
    status, pulled = _raw_post([(b"content-length", b"999999999")], [b"x" * 1024] * 1000)
    assert (status, pulled) == (401, 0)


def test_declared_oversized_body_rejected_before_it_is_read():
    client_with(Settings(access_code=STRONG_CODE, require_access_code=True, max_upload_bytes=1024))
    headers = [(b"x-access-code", STRONG_CODE.encode()), (b"content-length", b"999999999")]
    status, pulled = _raw_post(headers, [b"x" * 1024] * 1000)
    assert (status, pulled) == (413, 0)


def test_undeclared_oversized_body_stops_at_the_limit():
    client_with(Settings(max_upload_bytes=1024))
    part_header = (
        b"--xyz\r\nContent-Disposition: form-data; name=\"file\"; filename=\"a.dcm\"\r\n"
        b"Content-Type: application/octet-stream\r\n\r\n"
    )
    status, pulled = _raw_post([], [part_header] + [b"x" * 16 * 1024] * 1000)
    assert status == 413
    assert pulled < 10  # stopped after ~64 KiB instead of consuming all 16 MB


def test_malformed_multipart_is_rejected_early():
    client_with(Settings())
    status, pulled = _raw_post([], [b"x" * 16 * 1024] * 1000)
    assert status == 400
    assert pulled < 10


def test_missing_file_field_returns_422():
    resp = client_with(Settings()).post("/analyze", data={"other": "value"}, files={"not_file": ("a.png", b"x")})
    assert resp.status_code == 422
    assert "named 'file'" in resp.json()["detail"]


def test_openapi_documents_file_upload():
    schema = client_with(Settings()).get("/openapi.json").json()
    body = schema["paths"]["/analyze"]["post"]["requestBody"]["content"]["multipart/form-data"]["schema"]
    assert body["properties"]["file"]["format"] == "binary"


def test_concurrency_limits_per_stage():
    """Image work is capped at MAX_CONCURRENT_IMAGES; report writing has its own, larger cap."""
    import asyncio
    import threading
    import time

    import httpx

    lock = threading.Lock()
    peak = {"image": [0, 0], "report": [0, 0]}  # [current, max]

    def track(stage, delta):
        with lock:
            peak[stage][0] += delta
            peak[stage][1] = max(peak[stage][1], peak[stage][0])

    class SlowDetector(DemoDetector):
        def detect(self, pixels):
            track("image", 1)
            time.sleep(0.05)
            track("image", -1)
            return super().detect(pixels)

    class SlowReporter(TemplateReportGenerator):
        def generate(self, *args):
            track("report", 1)
            time.sleep(0.2)
            track("report", -1)
            return super().generate(*args)

    pipeline = Pipeline(Settings(max_concurrent_images=2, max_concurrent_reports=3), SlowDetector(), SlowReporter())
    app.dependency_overrides[get_pipeline] = lambda: pipeline
    data = png_bytes(400, 200)

    async def run():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            responses = await asyncio.gather(
                *(client.post("/analyze", files={"file": ("p.png", data)}) for _ in range(8))
            )
        return [r.status_code for r in responses]

    assert asyncio.run(run()) == [200] * 8
    assert peak["image"][1] == 2
    assert peak["report"][1] == 3


def test_examine_or_reason_returns_rejection_reason():
    pipeline = Pipeline(Settings(), DemoDetector(), TemplateReportGenerator())
    assert pipeline.examine_or_reason(b"garbage") == "Upload is neither DICOM nor a readable image file"
    assert pipeline.examine_or_reason(png_bytes()).study_id
