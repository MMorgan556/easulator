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
    pipeline = Pipeline(settings, DemoDetector(), reporter or TemplateReportGenerator())
    app.dependency_overrides[get_pipeline] = lambda: pipeline
    return TestClient(app)


@pytest.fixture(autouse=True)
def clear_overrides():
    yield
    app.dependency_overrides.clear()


def test_health():
    body = client_with(Settings()).get("/health").json()
    assert body == {"status": "ok", "detector": "demo", "report_generator": "template"}


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
