from app.postprocess import build_findings, non_max_suppression, number_teeth
from app.schemas import BoundingBox, Detection, FindingType, ImageModality


def det(kind, x1, y1, x2, y2, conf=0.9):
    return Detection(type=kind, confidence=conf, box=BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2))


def full_mouth(width=1600, height=800, per_side=8):
    """16 upper and 16 lower teeth, evenly spaced either side of the midline."""
    teeth = []
    tw = (width / 2) / (per_side + 1)
    for y1, y2 in ((100, 380), (420, 700)):
        for i in range(per_side):
            # viewer-left side, moving away from the midline
            x2 = width / 2 - 5 - i * tw
            teeth.append(det(FindingType.TOOTH, x2 - tw + 10, y1, x2, y2))
            # viewer-right side
            x1 = width / 2 + 5 + i * tw
            teeth.append(det(FindingType.TOOTH, x1, y1, x1 + tw - 10, y2))
    return teeth


def test_fdi_numbering_on_panoramic():
    teeth = number_teeth(full_mouth(), 1600, 800, ImageModality.PANORAMIC)
    fdis = {t.fdi for t in teeth}
    assert fdis == {f"{q}{n}" for q in (1, 2, 3, 4) for n in range(1, 9)}

    by_fdi = {t.fdi: t for t in teeth}
    # Patient's right is on the viewer's left: 11 and 41 sit just left of the midline.
    assert by_fdi["11"].box.center[0] < 800 < by_fdi["21"].box.center[0]
    assert by_fdi["41"].box.center[0] < 800 < by_fdi["31"].box.center[0]
    # Upper quadrants above lower ones; 18 further from the midline than 11.
    assert by_fdi["11"].box.center[1] < by_fdi["41"].box.center[1]
    assert by_fdi["18"].box.center[0] < by_fdi["11"].box.center[0]


def test_non_panoramic_teeth_are_not_numbered():
    teeth = number_teeth(full_mouth()[:4], 1600, 800, ImageModality.BITEWING)
    assert all(t.fdi == "?" for t in teeth)


def test_single_upper_arch_is_not_split():
    upper_only = [t for t in full_mouth() if t.box.y1 == 100]
    teeth = number_teeth(upper_only, 1600, 800, ImageModality.PANORAMIC)
    assert {t.fdi[0] for t in teeth} == {"1", "2"}


def test_single_lower_arch_gets_lower_quadrants():
    lower_only = [t for t in full_mouth() if t.box.y1 == 420]
    teeth = number_teeth(lower_only, 1600, 800, ImageModality.PANORAMIC)
    assert {t.fdi[0] for t in teeth} == {"3", "4"}


def test_single_tooth_is_numbered():
    teeth = number_teeth([det(FindingType.TOOTH, 700, 450, 790, 700)], 1600, 800, ImageModality.PANORAMIC)
    assert [t.fdi for t in teeth] == ["41"]


def test_nms_removes_duplicates_but_keeps_other_classes():
    a = det(FindingType.CARIES, 0, 0, 100, 100, 0.9)
    dup = det(FindingType.CARIES, 5, 5, 105, 105, 0.7)
    other_class = det(FindingType.RESTORATION, 5, 5, 105, 105, 0.8)
    kept = non_max_suppression([dup, a, other_class])
    assert a in kept and other_class in kept and dup not in kept


def test_findings_linked_to_tooth_and_flagged_by_confidence():
    teeth = full_mouth()
    t11 = number_teeth(teeth, 1600, 800, ImageModality.PANORAMIC)
    box_11 = next(t.box for t in t11 if t.fdi == "11")
    caries = det(FindingType.CARIES, box_11.x1 + 5, box_11.y1 + 5, box_11.x1 + 30, box_11.y1 + 30, 0.85)
    lesion = det(FindingType.PERIAPICAL_LESION, box_11.x1, box_11.y2 - 20, box_11.x2, box_11.y2, 0.4)
    noise = det(FindingType.CALCULUS, 0, 0, 10, 10, 0.1)

    numbered, findings = build_findings(
        teeth + [caries, lesion, noise], 1600, 800, ImageModality.PANORAMIC, min_confidence=0.25, review_confidence=0.6
    )
    assert len(numbered) == 32
    assert [f.type for f in findings] == [FindingType.CARIES, FindingType.PERIAPICAL_LESION]
    assert all(f.tooth == "11" for f in findings)
    assert [f.needs_review for f in findings] == [False, True]
    assert [f.id for f in findings] == ["F1", "F2"]


def test_finding_outside_any_tooth_is_unassigned():
    _, findings = build_findings(
        full_mouth() + [det(FindingType.BONE_LOSS, 0, 0, 20, 20)], 1600, 800, ImageModality.PANORAMIC, 0.25, 0.6
    )
    assert findings[0].tooth is None
