import importlib.util
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from .conftest import dicom_bytes, png_bytes

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_dataset_path_is_resolved_relative_to_config(tmp_path):
    train = load_script("train_yolo")
    (tmp_path / "configs").mkdir()
    (tmp_path / "datasets" / "dental").mkdir(parents=True)
    config = tmp_path / "configs" / "dental.yaml"
    config.write_text("path: ../datasets/dental\ntrain: images/train\nval: images/val\nnames:\n  0: tooth\n  1: caries\n")

    resolved = train.resolve_dataset_config(config, tmp_path / "runs")

    import yaml

    data = yaml.safe_load(resolved.read_text())
    assert data["path"] == str((tmp_path / "datasets" / "dental").resolve())
    assert data["names"] == {0: "tooth", 1: "caries"}


def test_missing_dataset_folder_is_reported(tmp_path):
    train = load_script("train_yolo")
    config = tmp_path / "dental.yaml"
    config.write_text("path: nowhere\nnames: [tooth]\n")
    with pytest.raises(SystemExit, match="does not exist"):
        train.resolve_dataset_config(config, tmp_path / "runs")


def test_unknown_class_is_rejected(tmp_path):
    train = load_script("train_yolo")
    config = tmp_path / "dental.yaml"
    config.write_text("path: .\nnames: [tooth, gold_crown]\n")
    with pytest.raises(SystemExit, match="gold_crown"):
        train.resolve_dataset_config(config, tmp_path / "runs")


def test_repo_dataset_config_names_match_finding_types():
    import yaml

    from app.schemas import FindingType

    config = yaml.safe_load((SCRIPTS.parent / "configs" / "dental.yaml").read_text())
    assert list(config["names"].values()) == [t.value for t in FindingType]


def test_prepare_dataset_exports_normalized_pngs(tmp_path):
    prepare = load_script("prepare_dataset")
    src = tmp_path / "raw" / "pans"
    src.mkdir(parents=True)
    (src / "a.dcm").write_bytes(dicom_bytes(photometric="MONOCHROME1"))
    (src / "b.png").write_bytes(png_bytes())
    (src / "notes.txt").write_text("not an image")

    written, failures = prepare.export(tmp_path / "raw", tmp_path / "out")

    assert written == 2
    assert len(failures) == 1 and "notes.txt" in failures[0]
    exported = np.asarray(Image.open(tmp_path / "out" / "pans" / "a.png"))
    from app.ingest import load_image

    assert np.array_equal(exported, load_image(dicom_bytes(photometric="MONOCHROME1")).pixels)


def test_prepare_dataset_reports_name_collisions(tmp_path):
    prepare = load_script("prepare_dataset")
    (tmp_path / "raw").mkdir()
    (tmp_path / "raw" / "scan.dcm").write_bytes(dicom_bytes())
    (tmp_path / "raw" / "scan.png").write_bytes(png_bytes())
    written, failures = prepare.export(tmp_path / "raw", tmp_path / "out")
    assert written == 1
    assert len(failures) == 1 and "already exports to scan.png" in failures[0]
