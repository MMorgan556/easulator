"""Turn structured findings into a written draft report.

The vision models decide *what* is on the image. The language model only writes
up those structured findings; it never sees the image, so it cannot add findings.
"""

from __future__ import annotations

import json
import os
from collections import defaultdict
from typing import Any, Protocol

from .config import Settings
from .schemas import PATHOLOGY_TYPES, Finding, ImageInfo, Report, Tooth

LABELS = {
    "caries": "caries",
    "periapical_lesion": "periapical radiolucency",
    "restoration": "restoration",
    "crown": "crown",
    "root_canal_treatment": "root canal treatment",
    "implant": "implant",
    "impacted_tooth": "impacted tooth",
    "bone_loss": "alveolar bone loss",
    "calculus": "calculus",
}

SYSTEM_PROMPT = """You write draft dental radiograph reports for a licensed dentist to review.

You receive JSON produced by computer-vision models that analyzed one dental X-ray. \
You do not see the image. Report only what the JSON contains: never add, infer, or \
upgrade findings, and never state a diagnosis more certain than the data supports.

Write plain text with these sections:
1. Image: modality and any quality notes from the warnings.
2. Findings: group by FDI tooth number (then findings with no tooth). For each finding give \
the type and confidence as a percentage. Prefix findings with "needs_review": true with \
"[REVIEW]" and say the model was uncertain.
3. Existing dental work: restorations, crowns, root canal treatments, implants.
4. Suggested follow-up: short, cautious suggestions phrased as considerations for the \
dentist (for example "consider clinical examination of 16"), not treatment orders.

If there are no pathology findings, say so plainly. Keep it concise and clinical."""


def _findings_payload(info: ImageInfo, teeth: list[Tooth], findings: list[Finding], warnings: list[str]) -> dict:
    return {
        "image": {"modality": info.modality.value, "width": info.width, "height": info.height},
        "teeth_detected": [t.fdi for t in teeth],
        "findings": [
            {
                "id": f.id,
                "type": f.type.value,
                "tooth_fdi": f.tooth,
                "confidence": f.confidence,
                "needs_review": f.needs_review,
            }
            for f in findings
        ],
        "warnings": warnings,
    }


class ReportGenerator(Protocol):
    name: str

    def generate(
        self, info: ImageInfo, teeth: list[Tooth], findings: list[Finding], warnings: list[str]
    ) -> Report: ...


class ReportGenerationError(RuntimeError):
    pass


class TemplateReportGenerator:
    """Deterministic report with no external calls."""

    name = "template"

    def generate(
        self, info: ImageInfo, teeth: list[Tooth], findings: list[Finding], warnings: list[str]
    ) -> Report:
        lines = [f"Image: {info.modality.value} radiograph, {info.width}x{info.height} px."]
        lines += [f"Note: {w}" for w in warnings]
        lines.append(f"Teeth detected: {len(teeth)}.")

        pathology = [f for f in findings if f.type in PATHOLOGY_TYPES]
        work = [f for f in findings if f.type not in PATHOLOGY_TYPES]

        lines.append("")
        lines.append("Findings:")
        if not pathology:
            lines.append("  No pathology detected by the model.")
        by_tooth: dict[str, list[Finding]] = defaultdict(list)
        for f in pathology:
            by_tooth[f.tooth or "unassigned"].append(f)
        for tooth in sorted(by_tooth, key=lambda t: (t == "unassigned", t)):
            label = f"Tooth {tooth}" if tooth != "unassigned" else "No tooth assigned"
            for f in by_tooth[tooth]:
                flag = "[REVIEW] " if f.needs_review else ""
                lines.append(f"  {flag}{label}: {LABELS[f.type.value]} ({f.confidence:.0%})")

        lines.append("")
        lines.append("Existing dental work:")
        if not work:
            lines.append("  None detected.")
        for f in work:
            where = f"tooth {f.tooth}" if f.tooth else "tooth not assigned"
            lines.append(f"  {LABELS[f.type.value]} on {where} ({f.confidence:.0%})")

        review = [f for f in findings if f.needs_review]
        if review:
            lines.append("")
            lines.append(f"{len(review)} low-confidence finding(s) flagged [REVIEW] need dentist confirmation.")
        return Report(text="\n".join(lines), generator=self.name)


class ClaudeReportGenerator:
    """Writes the report with Claude from the structured findings JSON."""

    name = "claude"

    def __init__(self, model: str, client: Any | None = None):
        if client is None:
            import anthropic

            client = anthropic.Anthropic()
        self.client = client
        self.model = model

    def generate(
        self, info: ImageInfo, teeth: list[Tooth], findings: list[Finding], warnings: list[str]
    ) -> Report:
        import anthropic

        payload = _findings_payload(info, teeth, findings, warnings)
        try:
            response = self.client.beta.messages.create(
                model=self.model,
                max_tokens=16000,
                system=SYSTEM_PROMPT,
                thinking={"type": "adaptive"},
                # If the request is declined, retry it server-side on Anthropic's recommended fallback model.
                fallbacks="default",
                betas=["server-side-fallback-2026-07-01"],
                messages=[{"role": "user", "content": json.dumps(payload, indent=2)}],
            )
        except anthropic.APIStatusError as exc:
            raise ReportGenerationError(f"Claude API error {exc.status_code}: {exc.message}") from exc
        except anthropic.APIConnectionError as exc:
            raise ReportGenerationError("Could not reach the Claude API") from exc

        if response.stop_reason == "refusal":
            raise ReportGenerationError("Claude declined to write this report")
        if response.stop_reason == "max_tokens":
            raise ReportGenerationError("Claude's report was cut off before it finished")
        text = "".join(block.text for block in response.content if block.type == "text").strip()
        if not text:
            raise ReportGenerationError(f"Claude returned no report text (stop_reason={response.stop_reason})")
        return Report(text=text, generator=f"claude:{self.model}")


def claude_credentials_configured() -> bool:
    return bool(os.getenv("ANTHROPIC_API_KEY") or os.getenv("ANTHROPIC_AUTH_TOKEN"))


def build_report_generator(settings: Settings) -> ReportGenerator:
    backend = settings.report_backend
    if backend == "template" or (backend == "auto" and not claude_credentials_configured()):
        return TemplateReportGenerator()
    if backend in ("auto", "claude"):
        return ClaudeReportGenerator(settings.claude_model)
    raise ValueError(f"Unknown REPORT_BACKEND {backend!r} (expected 'auto', 'claude' or 'template')")
