"""Turn raw detections into numbered teeth and tooth-linked findings."""

from __future__ import annotations

from .schemas import BoundingBox, Detection, Finding, FindingType, ImageModality, Tooth


def non_max_suppression(detections: list[Detection], iou_threshold: float = 0.5) -> list[Detection]:
    """Per-class NMS: keep the most confident box among heavily overlapping duplicates."""
    kept: list[Detection] = []
    for det in sorted(detections, key=lambda d: d.confidence, reverse=True):
        if all(k.type != det.type or k.box.iou(det.box) < iou_threshold for k in kept):
            kept.append(det)
    return kept


def _split_arches(teeth: list[Detection], image_height: int) -> tuple[list[Detection], list[Detection]]:
    """Split teeth into (upper, lower) arch at the largest vertical gap between centers."""
    if not teeth:
        return [], []
    ordered = sorted(teeth, key=lambda t: t.box.center[1])
    if len(ordered) >= 2:
        ys = [t.box.center[1] for t in ordered]
        gap, idx = max((ys[i + 1] - ys[i], i) for i in range(len(ys) - 1))
        typical_height = sorted(t.box.y2 - t.box.y1 for t in ordered)[len(ordered) // 2]
        if gap >= typical_height * 0.5:
            return ordered[: idx + 1], ordered[idx + 1 :]
    # Only one arch is visible: place it by where it sits in the image.
    mean_y = sum(t.box.center[1] for t in ordered) / len(ordered)
    return (ordered, []) if mean_y < image_height / 2 else ([], ordered)


def number_teeth(
    tooth_detections: list[Detection], image_width: int, image_height: int, modality: ImageModality
) -> list[Tooth]:
    """Assign FDI numbers on a panoramic image.

    Uses the radiographic display convention: the patient's right side appears on
    the viewer's left. Quadrants: 1 upper-right, 2 upper-left, 3 lower-left,
    4 lower-right; teeth are numbered 1-8 outward from the midline.

    This geometric rule assumes a full dentition with no gaps; missing teeth shift
    the numbering. Production systems should use a dedicated numbering model.
    """
    if modality != ImageModality.PANORAMIC:
        return [Tooth(fdi="?", confidence=d.confidence, box=d.box) for d in tooth_detections]

    midline = image_width / 2
    upper, lower = _split_arches(tooth_detections, image_height)
    teeth: list[Tooth] = []
    for arch, (right_q, left_q) in ((upper, (1, 2)), (lower, (4, 3))):
        viewer_left = sorted((t for t in arch if t.box.center[0] < midline), key=lambda t: -t.box.center[0])
        viewer_right = sorted((t for t in arch if t.box.center[0] >= midline), key=lambda t: t.box.center[0])
        for quadrant, side in ((right_q, viewer_left), (left_q, viewer_right)):
            for position, det in enumerate(side, start=1):
                fdi = f"{quadrant}{position}" if position <= 8 else "?"
                teeth.append(Tooth(fdi=fdi, confidence=det.confidence, box=det.box))
    return teeth


def _assign_tooth(box: BoundingBox, teeth: list[Tooth], min_overlap: float = 0.3) -> str | None:
    """The tooth covering the largest share of the finding, if it covers enough of it."""
    if box.area <= 0:
        return None
    best, best_share = None, 0.0
    for tooth in teeth:
        share = box.intersection(tooth.box) / box.area
        if share > best_share:
            best, best_share = tooth, share
    if best is None or best_share < min_overlap or best.fdi == "?":
        return None
    return best.fdi


def build_findings(
    detections: list[Detection],
    image_width: int,
    image_height: int,
    modality: ImageModality,
    min_confidence: float,
    review_confidence: float,
) -> tuple[list[Tooth], list[Finding]]:
    confident = [d for d in detections if d.confidence >= min_confidence]
    kept = non_max_suppression(confident)
    teeth = number_teeth([d for d in kept if d.type == FindingType.TOOTH], image_width, image_height, modality)

    others = sorted((d for d in kept if d.type != FindingType.TOOTH), key=lambda d: -d.confidence)
    findings = [
        Finding(
            id=f"F{i}",
            type=d.type,
            confidence=round(d.confidence, 3),
            box=d.box,
            tooth=_assign_tooth(d.box, teeth),
            needs_review=d.confidence < review_confidence,
        )
        for i, d in enumerate(others, start=1)
    ]
    teeth.sort(key=lambda t: (t.fdi == "?", t.fdi))
    return teeth, findings
