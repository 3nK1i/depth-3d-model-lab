from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from core.reconstruction.errors import ReconstructionValidationError
from core.reconstruction.temple_ring_closure import (
    build_i4_2f_closure,
    require_empty_or_absent_output,
    validate_i4_2f_evidence,
)

UNIT_PATH = Path(__file__).resolve().parents[1] / "unit/test_i4_2f_temple_ring_closure.py"
_spec = importlib.util.spec_from_file_location("i42f_unit_helpers", UNIT_PATH)
assert _spec is not None and _spec.loader is not None
_helpers = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_helpers)
_fixture = _helpers._fixture
_write_json = _helpers._write_json


def test_i42f_detects_native_visibility_mutation_after_review(tmp_path: Path) -> None:
    e_dir, p_dir = _fixture(tmp_path)
    e = json.loads((e_dir / "i4_2e_summary.json").read_text())
    path = Path(e["source"]["native_visibility_path"])
    path.write_bytes(path.read_bytes() + b"mutated")
    with pytest.raises(ReconstructionValidationError, match="fused.ply.vis hash mismatch"):
        validate_i4_2f_evidence(i4_2e_dir=e_dir, preview_dir=p_dir)


def test_i42f_detects_i4_2d_authority_regression(tmp_path: Path) -> None:
    e_dir, p_dir = _fixture(tmp_path)
    e = json.loads((e_dir / "i4_2e_summary.json").read_text())
    d_path = Path(e["source"]["i4_2d_dir"]) / "i4_2d_summary.json"
    d = json.loads(d_path.read_text())
    d["authority"]["dense_provider_promoted"] = True
    _write_json(d_path, d)
    with pytest.raises(ReconstructionValidationError, match="dense_provider_promoted"):
        validate_i4_2f_evidence(i4_2e_dir=e_dir, preview_dir=p_dir)


def test_i42f_detects_frozen_b_summary_mutation(tmp_path: Path) -> None:
    e_dir, p_dir = _fixture(tmp_path)
    e = json.loads((e_dir / "i4_2e_summary.json").read_text())
    d = json.loads((Path(e["source"]["i4_2d_dir"]) / "i4_2d_summary.json").read_text())
    b_path = Path(d["artifacts"]["frozen_i4_2b_summary"])
    b_path.write_bytes(b_path.read_bytes() + b"mutated")
    with pytest.raises(ReconstructionValidationError, match="frozen I4.2B summary hash mismatch"):
        validate_i4_2f_evidence(i4_2e_dir=e_dir, preview_dir=p_dir)


def test_i42f_detects_trajectory_quality_regression_even_with_c_pass_flag(tmp_path: Path) -> None:
    e_dir, p_dir = _fixture(tmp_path)
    e = json.loads((e_dir / "i4_2e_summary.json").read_text())
    d_path = Path(e["source"]["i4_2d_dir"]) / "i4_2d_summary.json"
    d = json.loads(d_path.read_text())
    c_path = Path(d["artifacts"]["frozen_i4_2c_summary"])
    c = json.loads(c_path.read_text())
    c["trajectory_quality"]["accepted"] = False
    _write_json(c_path, c)
    d["input_authority"]["i4_2c_summary_sha256"] = __import__("hashlib").sha256(c_path.read_bytes()).hexdigest()
    _write_json(d_path, d)
    with pytest.raises(ReconstructionValidationError, match="accepted held-out"):
        validate_i4_2f_evidence(i4_2e_dir=e_dir, preview_dir=p_dir)


def test_i42f_rejects_changed_manual_review_dimensions(tmp_path: Path) -> None:
    e_dir, p_dir = _fixture(tmp_path)
    e_path = e_dir / "i4_2e_summary.json"
    e = json.loads(e_path.read_text())
    e["manual_review_contract"]["review_dimensions"] = e["manual_review_contract"]["review_dimensions"][:-1]
    _write_json(e_path, e)
    with pytest.raises(ReconstructionValidationError, match="dimension contract"):
        validate_i4_2f_evidence(i4_2e_dir=e_dir, preview_dir=p_dir)


def test_i42f_closure_is_deterministic_for_same_evidence_and_verdict(tmp_path: Path) -> None:
    e_dir, p_dir = _fixture(tmp_path)
    evidence = validate_i4_2f_evidence(i4_2e_dir=e_dir, preview_dir=p_dir)
    a = build_i4_2f_closure(evidence_validation=evidence, verdict="d12_safe_on_temple_ring", confirmed=True)
    b = build_i4_2f_closure(evidence_validation=evidence, verdict="d12_safe_on_temple_ring", confirmed=True)
    assert a == b


def test_i42f_immutable_output_refuses_existing_evidence(tmp_path: Path) -> None:
    output = tmp_path / "closure"
    output.mkdir()
    (output / "i4_2f_closure.json").write_text("{}\n", encoding="utf-8")
    with pytest.raises(ReconstructionValidationError, match="refusing to overwrite"):
        require_empty_or_absent_output(output)


def test_i42f_safe_closure_preserves_no_metric_measurement_or_mesh_authority(tmp_path: Path) -> None:
    e_dir, p_dir = _fixture(tmp_path)
    evidence = validate_i4_2f_evidence(i4_2e_dir=e_dir, preview_dir=p_dir)
    closure = build_i4_2f_closure(evidence_validation=evidence, verdict="d12_safe_on_temple_ring", confirmed=True)
    assert closure["authority"]["metric_scale_claimed"] is False
    assert closure["authority"]["measurement_authoritative"] is False
    assert closure["authority"]["certified_metrology"] is False
    assert closure["authority"]["surface_mesh_generated"] is False
