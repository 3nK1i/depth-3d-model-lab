from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest

from core.reconstruction.errors import ReconstructionValidationError
from core.reconstruction.temple_ring_trajectory_validation import (
    _require_clean_output,
    _validate_pose_export_result,
    evaluate_trajectory,
    fit_similarity_umeyama,
    load_i4_2c_inputs,
    run_temple_ring_heldout_trajectory_validation,
)

UNIT = Path(__file__).resolve().parents[1] / "unit/test_i4_2c_temple_ring_trajectory_validation.py"
_spec = importlib.util.spec_from_file_location("i42c_helpers", UNIT)
assert _spec is not None and _spec.loader is not None
_helpers = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_helpers)
_evidence = _helpers._evidence
_gt_records = _helpers._gt_records
_pose_worker_result = _helpers._pose_worker_result
_ready_probe = _helpers._ready_probe
_sfm_records = _helpers._sfm_records
_write_json = _helpers._write_json


def test_i42c_rejects_nonempty_output_attempt(tmp_path: Path) -> None:
    output = tmp_path / "i4_2c_attempt01"
    output.mkdir()
    (output / "keep.json").write_text("{}\n", encoding="utf-8")
    with pytest.raises(ReconstructionValidationError, match="new immutable validation attempt"):
        _require_clean_output(output)


def test_i42c_rejects_degenerate_center_trajectory() -> None:
    source = np.array([[float(i), 0.0, 0.0] for i in range(47)])
    target = source * 2.0
    with pytest.raises(ReconstructionValidationError, match="degenerate"):
        fit_similarity_umeyama(source, target)


def test_i42c_forbids_reflection_in_similarity_fit() -> None:
    gt = np.stack([np.asarray(item["camera_center_world"]) for item in _gt_records()])
    reflected = gt.copy()
    reflected[:, 0] *= -1.0
    fit = fit_similarity_umeyama(gt, reflected)
    assert np.linalg.det(fit.rotation) == pytest.approx(1.0, abs=1e-10)
    residual = np.linalg.norm(fit.apply(gt) - reflected, axis=1)
    assert float(np.max(residual)) > 1e-4


def test_i42c_rejects_mutated_authoritative_sparse_binary(tmp_path: Path) -> None:
    a, b = _evidence(tmp_path)
    path = b / "workspace/authoritative_sparse/images.bin"
    path.write_bytes(path.read_bytes() + b"changed")
    with pytest.raises(ReconstructionValidationError, match="changed since I4.2B"):
        load_i4_2c_inputs(i4_2a_dir=a, i4_2b_dir=b)


def test_i42c_rejects_duplicate_pose_export_name() -> None:
    result = _pose_worker_result()
    result["registered_images"][1]["name"] = result["registered_images"][0]["name"]
    with pytest.raises(ReconstructionValidationError, match="duplicate"):
        _validate_pose_export_result(result)


def test_i42c_rejects_nonfinite_pose_export_matrix() -> None:
    result = _pose_worker_result()
    result["registered_images"][0]["camera_from_reconstruction"][0][0] = float("nan")
    with pytest.raises(ReconstructionValidationError, match="finite"):
        _validate_pose_export_result(result)


def test_i42c_rejects_ground_truth_center_that_disagrees_with_rt(tmp_path: Path) -> None:
    a, b = _evidence(tmp_path)
    gt_path = a / "i4_2a_ground_truth_validation_manifest.json"
    summary_path = a / "i4_2a_summary.json"
    payload = json.loads(gt_path.read_text(encoding="utf-8"))
    payload["camera_records"][10]["camera_center_world"][0] += 0.1
    _write_json(gt_path, payload)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["artifacts"]["ground_truth_validation_manifest_sha256"] = hashlib.sha256(gt_path.read_bytes()).hexdigest()
    _write_json(summary_path, summary)
    # Keep I4.2B's frozen A summary synchronized so this test reaches the R/t consistency check.
    _write_json(b / "evidence/i4_2a_summary.json", summary)
    with pytest.raises(ReconstructionValidationError, match="centre/R/t inconsistency"):
        load_i4_2c_inputs(i4_2a_dir=a, i4_2b_dir=b)


def test_i42c_detects_sparse_mutation_during_pose_export(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    a, b = _evidence(tmp_path)
    worker = tmp_path / "fake.py"
    worker.write_text("# fake\n", encoding="utf-8")
    monkeypatch.setattr(
        "core.reconstruction.temple_ring_trajectory_validation.probe_pycolmap_environment",
        lambda _python: _ready_probe(),
    )

    class Completed:
        returncode = 0

    def fake_run(command, **_kwargs):
        (b / "workspace/authoritative_sparse/images.bin").write_bytes(b"mutated during read")
        result_path = Path(command[command.index("--result") + 1])
        _write_json(result_path, _pose_worker_result())
        return Completed()

    monkeypatch.setattr("core.reconstruction.temple_ring_trajectory_validation.subprocess.run", fake_run)
    with pytest.raises(ReconstructionValidationError, match="detected mutation"):
        run_temple_ring_heldout_trajectory_validation(
            i4_2a_dir=a,
            i4_2b_dir=b,
            output_dir=tmp_path / "out",
            pycolmap_python=sys.executable,
            pose_export_worker_script=worker,
        )


def test_i42c_evaluation_is_deterministic_for_same_evidence() -> None:
    first = evaluate_trajectory(sfm_records=_sfm_records(), ground_truth_records=_gt_records())
    second = evaluate_trajectory(sfm_records=_sfm_records(), ground_truth_records=_gt_records())
    assert first[0] == second[0]
    assert first[1] == second[1]
    assert first[2] == second[2]
    assert first[3] == second[3]
