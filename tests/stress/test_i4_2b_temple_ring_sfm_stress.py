from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from core.reconstruction.errors import ReconstructionValidationError
from core.reconstruction.temple_ring_sfm import (
    TempleRingSfmConfig,
    _validate_worker_result,
    build_temple_ring_sfm_worker_request,
    load_i4_2a_reconstruction_inputs,
    stage_temple_ring_images,
)

UNIT = Path(__file__).resolve().parents[1] / "unit/test_i4_2b_temple_ring_sfm.py"
_spec = importlib.util.spec_from_file_location("i42b_helpers", UNIT)
assert _spec is not None and _spec.loader is not None
_helpers = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_helpers)
_preflight = _helpers._preflight
_fake_worker_result = _helpers._fake_worker_result
_write_sparse = _helpers._write_sparse


def test_i42b_rejects_nonempty_attempt_directory(tmp_path: Path) -> None:
    from core.reconstruction.temple_ring_sfm import _require_clean_output
    output = tmp_path / "attempt01"
    output.mkdir()
    (output / "evidence.txt").write_text("keep", encoding="utf-8")
    with pytest.raises(ReconstructionValidationError, match="new immutable attempt"):
        _require_clean_output(output)


def test_i42b_rejects_gt_pose_injected_into_reconstruction_manifest(tmp_path: Path) -> None:
    root = _preflight(tmp_path)
    manifest = root / "i4_2a_reconstruction_input_manifest.json"
    summary = root / "i4_2a_summary.json"
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["R_camera_from_world"] = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    manifest.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
    sp = json.loads(summary.read_text(encoding="utf-8"))
    import hashlib
    sp["artifacts"]["reconstruction_input_manifest_sha256"] = hashlib.sha256(manifest.read_bytes()).hexdigest()
    summary.write_text(json.dumps(sp, sort_keys=True) + "\n", encoding="utf-8")
    with pytest.raises(ReconstructionValidationError, match="forbidden held-out evidence"):
        load_i4_2a_reconstruction_inputs(root)


def test_i42b_rejects_reordered_canonical_views(tmp_path: Path) -> None:
    root = _preflight(tmp_path)
    manifest = root / "i4_2a_reconstruction_input_manifest.json"
    summary = root / "i4_2a_summary.json"
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["images"][0], payload["images"][1] = payload["images"][1], payload["images"][0]
    manifest.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
    import hashlib
    sp = json.loads(summary.read_text(encoding="utf-8"))
    sp["artifacts"]["reconstruction_input_manifest_sha256"] = hashlib.sha256(manifest.read_bytes()).hexdigest()
    summary.write_text(json.dumps(sp, sort_keys=True) + "\n", encoding="utf-8")
    with pytest.raises(ReconstructionValidationError, match="canonical image order changed"):
        load_i4_2a_reconstruction_inputs(root)


def test_i42b_request_rejects_gt_token_via_name_injection(tmp_path: Path) -> None:
    inputs = load_i4_2a_reconstruction_inputs(_preflight(tmp_path))
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    staged = list(stage_temple_ring_images(inputs, workspace / "images"))
    staged[0] = dict(staged[0])
    staged[0]["name"] = "R_camera_from_world"
    with pytest.raises(ReconstructionValidationError, match="leaked forbidden token"):
        build_temple_ring_sfm_worker_request(
            inputs=inputs, workspace=workspace, staged_images=staged, config=TempleRingSfmConfig()
        )


def test_i42b_result_rejects_missing_sparse_binary(tmp_path: Path) -> None:
    inputs = load_i4_2a_reconstruction_inputs(_preflight(tmp_path))
    workspace = tmp_path / "workspace"
    _write_sparse(workspace)
    (workspace / "authoritative_sparse/points3D.bin").unlink()
    with pytest.raises(ReconstructionValidationError, match="sparse model is incomplete"):
        _validate_worker_result(_fake_worker_result(), inputs=inputs, workspace=workspace)


def test_i42b_result_rejects_gt_consumption_claim(tmp_path: Path) -> None:
    inputs = load_i4_2a_reconstruction_inputs(_preflight(tmp_path))
    workspace = tmp_path / "workspace"
    _write_sparse(workspace)
    result = _fake_worker_result()
    result["authority"]["ground_truth_extrinsics_consumed"] = True
    with pytest.raises(ReconstructionValidationError, match="forbidden upstream state"):
        _validate_worker_result(result, inputs=inputs, workspace=workspace)


def test_i42b_result_low_sparse_quality_is_degraded(tmp_path: Path) -> None:
    inputs = load_i4_2a_reconstruction_inputs(_preflight(tmp_path))
    workspace = tmp_path / "workspace"
    _write_sparse(workspace)
    result = _fake_worker_result()
    result["statistics"]["landmarks"] = 25
    result["statistics"]["observations"] = 50
    result["statistics"]["mean_reprojection_error_px"] = 4.0
    validation, status = _validate_worker_result(result, inputs=inputs, workspace=workspace)
    assert status == "degraded"
    assert validation["generic_sfm_quality"]["status"] == "degraded"


def test_i42b_result_is_deterministic_for_same_evidence(tmp_path: Path) -> None:
    inputs = load_i4_2a_reconstruction_inputs(_preflight(tmp_path))
    workspace = tmp_path / "workspace"
    _write_sparse(workspace)
    first = _validate_worker_result(_fake_worker_result(), inputs=inputs, workspace=workspace)
    second = _validate_worker_result(_fake_worker_result(), inputs=inputs, workspace=workspace)
    assert first == second
