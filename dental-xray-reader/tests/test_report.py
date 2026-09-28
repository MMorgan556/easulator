import json
from types import SimpleNamespace

import pytest

from app.config import Settings
from app.report import (
    ClaudeReportGenerator,
    ReportGenerationError,
    TemplateReportGenerator,
    build_report_generator,
)
from app.schemas import BoundingBox, Finding, FindingType, ImageInfo, ImageModality, Tooth

INFO = ImageInfo(width=1000, height=500, modality=ImageModality.PANORAMIC, source_format="png")
BOX = BoundingBox(x1=0, y1=0, x2=10, y2=10)
TEETH = [Tooth(fdi="16", confidence=0.9, box=BOX)]
FINDINGS = [
    Finding(id="F1", type=FindingType.CARIES, confidence=0.82, box=BOX, tooth="16"),
    Finding(id="F2", type=FindingType.PERIAPICAL_LESION, confidence=0.41, box=BOX, tooth=None, needs_review=True),
    Finding(id="F3", type=FindingType.CROWN, confidence=0.9, box=BOX, tooth="16"),
]


class FakeMessages:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return self.response


def fake_client(stop_reason="end_turn", text="Draft report"):
    content = [SimpleNamespace(type="thinking", thinking=""), SimpleNamespace(type="text", text=text)]
    messages = FakeMessages(SimpleNamespace(stop_reason=stop_reason, content=content))
    return SimpleNamespace(beta=SimpleNamespace(messages=messages)), messages


def test_template_report_groups_findings():
    report = TemplateReportGenerator().generate(INFO, TEETH, FINDINGS, ["DEMO MODE"])
    assert report.generator == "template"
    assert "Tooth 16: caries (82%)" in report.text
    assert "[REVIEW] No tooth assigned: periapical radiolucency (41%)" in report.text
    assert "crown on tooth 16" in report.text
    assert "Note: DEMO MODE" in report.text
    assert "licensed dentist" in report.disclaimer


def test_template_report_with_no_findings():
    report = TemplateReportGenerator().generate(INFO, TEETH, [], [])
    assert "No pathology detected" in report.text


def test_claude_report_sends_structured_findings_only():
    client, messages = fake_client(text="  Findings: caries on 16.  ")
    report = ClaudeReportGenerator("claude-opus-5", client=client).generate(INFO, TEETH, FINDINGS, [])

    assert report.text == "Findings: caries on 16."
    assert report.generator == "claude:claude-opus-5"
    call = messages.calls[0]
    assert call["model"] == "claude-opus-5"
    assert call["fallbacks"] == "default"
    payload = json.loads(call["messages"][0]["content"])
    assert payload["findings"][0] == {
        "id": "F1",
        "type": "caries",
        "tooth_fdi": "16",
        "confidence": 0.82,
        "needs_review": False,
    }
    assert "box" not in payload["findings"][0]


def test_claude_refusal_raises():
    client, _ = fake_client(stop_reason="refusal", text="")
    with pytest.raises(ReportGenerationError, match="declined"):
        ClaudeReportGenerator("claude-opus-5", client=client).generate(INFO, TEETH, FINDINGS, [])


def test_auto_backend_uses_template_without_credentials(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    assert build_report_generator(Settings(report_backend="auto")).name == "template"


def test_unknown_backend_rejected():
    with pytest.raises(ValueError):
        build_report_generator(Settings(report_backend="nope"))
