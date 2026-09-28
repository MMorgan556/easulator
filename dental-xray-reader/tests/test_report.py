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


def fake_client(stop_reason="end_turn", text="Draft report", model="claude-opus-5"):
    content = [SimpleNamespace(type="thinking", thinking=""), SimpleNamespace(type="text", text=text)]
    messages = FakeMessages(SimpleNamespace(stop_reason=stop_reason, content=content, model=model))
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


def test_claude_truncated_report_raises():
    client, _ = fake_client(stop_reason="max_tokens", text="Findings: cari")
    with pytest.raises(ReportGenerationError, match="cut off"):
        ClaudeReportGenerator("claude-opus-5", client=client).generate(INFO, TEETH, FINDINGS, [])


def test_claude_request_through_real_sdk():
    """Runs the real Anthropic SDK against a fake transport to check the wire request and response parsing."""
    import anthropic
    import httpx2

    seen = {}

    def handler(request: httpx2.Request) -> httpx2.Response:
        seen["path"] = request.url.path
        seen["beta"] = request.headers.get("anthropic-beta")
        seen["body"] = json.loads(request.content)
        return httpx2.Response(
            200,
            json={
                "id": "msg_test",
                "type": "message",
                "role": "assistant",
                "model": "claude-opus-5",
                "content": [
                    {"type": "thinking", "thinking": "", "signature": "sig"},
                    {"type": "text", "text": "Tooth 16: caries (82%)."},
                ],
                "stop_reason": "end_turn",
                "stop_sequence": None,
                "usage": {"input_tokens": 100, "output_tokens": 20},
            },
        )

    client = anthropic.Anthropic(
        api_key="test-key",
        max_retries=0,
        http_client=anthropic.DefaultHttpxClient(transport=httpx2.MockTransport(handler)),
    )
    report = ClaudeReportGenerator("claude-opus-5", client=client).generate(INFO, TEETH, FINDINGS, [])

    assert report.text == "Tooth 16: caries (82%)."
    assert seen["path"] == "/v1/messages"
    assert "server-side-fallback-2026-07-01" in seen["beta"]
    body = seen["body"]
    assert body["model"] == "claude-opus-5"
    assert body["fallbacks"] == "default"
    assert body["thinking"] == {"type": "adaptive"}
    assert body["system"].startswith("You write draft dental radiograph reports")
    assert json.loads(body["messages"][0]["content"])["findings"][0]["type"] == "caries"


def test_claude_api_error_becomes_report_error():
    import anthropic
    import httpx2

    def handler(request):
        return httpx2.Response(529, json={"type": "error", "error": {"type": "overloaded_error", "message": "busy"}})

    client = anthropic.Anthropic(
        api_key="test-key",
        max_retries=0,
        http_client=anthropic.DefaultHttpxClient(transport=httpx2.MockTransport(handler)),
    )
    with pytest.raises(ReportGenerationError, match="529"):
        ClaudeReportGenerator("claude-opus-5", client=client).generate(INFO, TEETH, FINDINGS, [])


def test_report_names_the_model_that_actually_answered():
    client, _ = fake_client(model="claude-opus-4-8")  # e.g. a server-side fallback served the request
    report = ClaudeReportGenerator("claude-opus-5", client=client).generate(INFO, TEETH, FINDINGS, [])
    assert report.generator == "claude:claude-opus-4-8"


def test_missing_credentials_become_report_error(monkeypatch, tmp_path):
    import anthropic

    for name in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_PROFILE"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))  # no `ant auth login` profile either
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    generator = ClaudeReportGenerator("claude-opus-5", client=anthropic.Anthropic(max_retries=0))
    with pytest.raises(ReportGenerationError, match="credentials"):
        generator.generate(INFO, TEETH, FINDINGS, [])


def test_unexpected_sdk_error_becomes_report_error():
    import anthropic

    class Broken:
        def create(self, **kwargs):
            raise anthropic.AnthropicError("unexpected response shape")

    client = SimpleNamespace(beta=SimpleNamespace(messages=Broken()))
    with pytest.raises(ReportGenerationError, match="unexpected response shape"):
        ClaudeReportGenerator("claude-opus-5", client=client).generate(INFO, TEETH, FINDINGS, [])


def test_programming_type_errors_are_not_disguised_as_credentials():
    class Broken:
        def create(self, **kwargs):
            raise TypeError("create() got an unexpected keyword argument 'fallbacks'")

    client = SimpleNamespace(beta=SimpleNamespace(messages=Broken()))
    with pytest.raises(TypeError, match="unexpected keyword"):
        ClaudeReportGenerator("claude-opus-5", client=client).generate(INFO, TEETH, FINDINGS, [])
