from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from core.reconstruction.errors import ReconstructionValidationError
from core.reconstruction.temple_ring_mvs import TempleRingMvsConfig, _require_clean_output, load_i4_2d_inputs

UNIT = Path(__file__).resolve().parents[1] / "unit/test_i4_2d_temple_ring_mvs.py"
_spec = importlib.util.spec_from_file_location("i42d_helpers", UNIT)
assert _spec is not None and _spec.loader is not None
_helpers = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_helpers)
_evidence = _helpers._evidence
_write_json = _helpers._write_json


def test_i42d_never_overwrites_nonempty_attempt(tmp_path: Path) -> None:
    output = tmp_path / "i4_2d_attempt01"
    output.mkdir()
    marker = output / "preserve.txt"
    marker.write_text("keep", encoding="utf-8")
    with pytest.raises(ReconstructionValidationError, match="refusing to overwrite raw MVS evidence"):
        _require_clean_output(output)
    assert marker.read_text(encoding="utf-8") == "keep"


def test_i42d_rejects_nonpassing_i42b(tmp_path: Path) -> None:
    b, c = _evidence(tmp_path)
    path = b / "i4_2b_summary.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["status"] = "degraded"
    payload["pass"] = False
    _write_json(path, payload)
    with pytest.raises(ReconstructionValidationError, match="requires I4.2B PASS"):
        load_i4_2d_inputs(i4_2b_dir=b, i4_2c_dir=c)


def test_i42d_rejects_nonpassing_i42c(tmp_path: Path) -> None:
    b, c = _evidence(tmp_path)
    path = c / "i4_2c_summary.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["status"] = "degraded"
    payload["pass"] = False
    _write_json(path, payload)
    with pytest.raises(ReconstructionValidationError, match="requires I4.2C PASS"):
        load_i4_2d_inputs(i4_2b_dir=b, i4_2c_dir=c)


def test_i42d_rejects_unaccepted_trajectory(tmp_path: Path) -> None:
    b, c = _evidence(tmp_path)
    path = c / "i4_2c_summary.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["trajectory_quality"]["accepted"] = False
    _write_json(path, payload)
    with pytest.raises(ReconstructionValidationError, match="accepted held-out trajectory"):
        load_i4_2d_inputs(i4_2b_dir=b, i4_2c_dir=c)


def test_i42d_rejects_sparse_mutation_after_i42c(tmp_path: Path) -> None:
    b, c = _evidence(tmp_path)
    path = b / "workspace/authoritative_sparse/images.bin"
    path.write_bytes(path.read_bytes() + b"changed")
    with pytest.raises(ReconstructionValidationError, match="changed before I4.2D"):
        load_i4_2d_inputs(i4_2b_dir=b, i4_2c_dir=c)


def test_i42d_rejects_staged_image_mutation(tmp_path: Path) -> None:
    b, c = _evidence(tmp_path)
    path = b / "workspace/images/templeR0024.png"
    path.write_bytes(path.read_bytes() + b"changed")
    with pytest.raises(ReconstructionValidationError, match="staged image changed"):
        load_i4_2d_inputs(i4_2b_dir=b, i4_2c_dir=c)


def test_i42d_rejects_i42c_bound_to_different_i42b_summary(tmp_path: Path) -> None:
    b, c = _evidence(tmp_path)
    path = c / "i4_2c_summary.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["evidence"]["source_sha256"]["i4_2b_summary"] = "0" * 64
    _write_json(path, payload)
    with pytest.raises(ReconstructionValidationError, match="checksum-bind"):
        load_i4_2d_inputs(i4_2b_dir=b, i4_2c_dir=c)


def test_i42d_rejects_upstream_provider_architecture_promotion(tmp_path: Path) -> None:
    b, c = _evidence(tmp_path)
    path = b / "i4_2b_summary.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["authority"]["i4_3_provider_architecture_authorized"] = True
    _write_json(path, payload)
    with pytest.raises(ReconstructionValidationError, match="I4.2B state i4_3_provider_architecture_authorized=True"):
        load_i4_2d_inputs(i4_2b_dir=b, i4_2c_dir=c)


def test_i42d_rejects_invalid_gpu_and_cache_before_heavy_mvs() -> None:
    with pytest.raises(ReconstructionValidationError, match="gpu_index"):
        TempleRingMvsConfig(gpu_index="gpu0")
    with pytest.raises(ReconstructionValidationError):
        TempleRingMvsConfig(patch_match_cache_gib=0.0)
    with pytest.raises(ReconstructionValidationError):
        TempleRingMvsConfig(stereo_fusion_cache_gib=-1.0)
