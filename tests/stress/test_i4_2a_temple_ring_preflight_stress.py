from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from core.reconstruction.errors import ReconstructionValidationError
from core.reconstruction.temple_ring_preflight import (
    inspect_temple_ring_dataset,
    run_i4_2a_preflight,
    sha256_file,
    validate_i4_1_9a_closure,
)

UNIT = Path(__file__).resolve().parents[1] / "unit/test_i4_2a_temple_ring_preflight.py"
_spec = importlib.util.spec_from_file_location("i42a_helpers", UNIT)
assert _spec is not None and _spec.loader is not None
_helpers = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_helpers)
_dataset = _helpers._dataset
_closure = _helpers._closure
_png = _helpers._png
_calibration_line = _helpers._calibration_line


def test_i42a_rejects_46_calibration_rows(tmp_path: Path) -> None:
    root = _dataset(tmp_path)
    lines = (root / "templeR_par.txt").read_text(encoding="utf-8").splitlines()[:-1]
    (root / "templeR_par.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    with pytest.raises(ReconstructionValidationError, match="exactly 47"):
        inspect_temple_ring_dataset(root)


def test_i42a_rejects_duplicate_calibration_name(tmp_path: Path) -> None:
    root = _dataset(tmp_path)
    lines = (root / "templeR_par.txt").read_text(encoding="utf-8").splitlines()
    lines[-1] = _calibration_line("templeR0046.png", 47)
    (root / "templeR_par.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    with pytest.raises(ReconstructionValidationError, match="duplicate"):
        inspect_temple_ring_dataset(root)


def test_i42a_rejects_wrong_image_resolution(tmp_path: Path) -> None:
    root = _dataset(tmp_path)
    _png(root / "templeR0020.png", width=320, height=240)
    with pytest.raises(ReconstructionValidationError, match="dimensions changed"):
        inspect_temple_ring_dataset(root)


def test_i42a_rejects_nonfinite_calibration(tmp_path: Path) -> None:
    root = _dataset(tmp_path)
    lines = (root / "templeR_par.txt").read_text(encoding="utf-8").splitlines()
    parts = lines[0].split()
    parts[1] = "nan"
    lines[0] = " ".join(parts)
    (root / "templeR_par.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    with pytest.raises(ReconstructionValidationError, match="non-finite"):
        inspect_temple_ring_dataset(root)


def test_i42a_rejects_upstream_dense_provider_promotion(tmp_path: Path) -> None:
    closure = _closure(tmp_path)
    payload = json.loads(closure.read_text(encoding="utf-8"))
    payload["authority"]["dense_provider_promoted"] = True
    closure.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ReconstructionValidationError, match="dense_provider_promoted"):
        validate_i4_1_9a_closure(closure)


def test_i42a_detects_image_mutation_between_runs_via_manifest_hash(tmp_path: Path) -> None:
    root = _dataset(tmp_path)
    closure = _closure(tmp_path)
    out1 = tmp_path / "out1"
    run_i4_2a_preflight(dataset_dir=root, i4_1_9a_closure=closure, output_dir=out1)
    manifest1 = json.loads((out1 / "i4_2a_reconstruction_input_manifest.json").read_text(encoding="utf-8"))
    image = root / "templeR0010.png"
    image.write_bytes(image.read_bytes() + b"mutated")
    out2 = tmp_path / "out2"
    run_i4_2a_preflight(dataset_dir=root, i4_1_9a_closure=closure, output_dir=out2)
    manifest2 = json.loads((out2 / "i4_2a_reconstruction_input_manifest.json").read_text(encoding="utf-8"))
    old = next(item["sha256"] for item in manifest1["images"] if item["name"] == "templeR0010.png")
    new = next(item["sha256"] for item in manifest2["images"] if item["name"] == "templeR0010.png")
    assert old != new


def test_i42a_is_deterministic_for_identical_inputs_except_paths(tmp_path: Path) -> None:
    root = _dataset(tmp_path)
    closure = _closure(tmp_path)
    out1 = tmp_path / "out1"
    out2 = tmp_path / "out2"
    run_i4_2a_preflight(dataset_dir=root, i4_1_9a_closure=closure, output_dir=out1)
    run_i4_2a_preflight(dataset_dir=root, i4_1_9a_closure=closure, output_dir=out2)
    m1 = json.loads((out1 / "i4_2a_reconstruction_input_manifest.json").read_text(encoding="utf-8"))
    m2 = json.loads((out2 / "i4_2a_reconstruction_input_manifest.json").read_text(encoding="utf-8"))
    assert m1 == m2


def test_i42a_gt_manifest_and_reconstruction_manifest_are_separate_hashes(tmp_path: Path) -> None:
    out = tmp_path / "out"
    run_i4_2a_preflight(dataset_dir=_dataset(tmp_path), i4_1_9a_closure=_closure(tmp_path), output_dir=out)
    assert sha256_file(out / "i4_2a_reconstruction_input_manifest.json") != sha256_file(out / "i4_2a_ground_truth_validation_manifest.json")
