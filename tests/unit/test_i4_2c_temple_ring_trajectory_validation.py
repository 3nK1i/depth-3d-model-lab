from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import sys

import numpy as np
import pytest

from core.reconstruction.errors import ReconstructionValidationError
from core.reconstruction.sfm_environment import PYCOLMAP_REQUIRED_APIS, PycolmapEnvironmentProbe
from core.reconstruction.temple_ring_trajectory_validation import (
    TEMPLE_RING_EXPECTED_NAMES,
    TempleRingTrajectoryThresholds,
    _validate_pose_export_result,
    camera_center_from_camera_from_world,
    evaluate_trajectory,
    fit_similarity_umeyama,
    load_i4_2c_inputs,
    rotation_error_degrees,
    run_temple_ring_heldout_trajectory_validation,
)

K = [[1520.4, 0.0, 302.32], [0.0, 1525.9, 246.87], [0.0, 0.0, 1.0]]


def _rot_z(deg: float) -> np.ndarray:
    angle = np.deg2rad(deg)
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)


def _rot_y(deg: float) -> np.ndarray:
    angle = np.deg2rad(deg)
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=np.float64)


def _gt_records() -> list[dict]:
    records = []
    for index, name in enumerate(TEMPLE_RING_EXPECTED_NAMES):
        angle = 2.0 * np.pi * index / 47.0
        center = np.array([0.55 * np.cos(angle), 0.48 * np.sin(angle), 0.07 * np.sin(2.0 * angle)], dtype=np.float64)
        R = np.eye(3, dtype=np.float64)
        t = -(R @ center)
        records.append({
            "image_name": name,
            "R_camera_from_world": R.tolist(),
            "t_camera_from_world": t.tolist(),
            "camera_center_world": center.tolist(),
        })
    return records


def _known_similarity() -> tuple[float, np.ndarray, np.ndarray]:
    return 2.3, _rot_z(31.0) @ _rot_y(-13.0), np.array([0.2, -0.17, 0.08], dtype=np.float64)


def _sfm_records(*, center_noise: float = 0.0, rotation_noise_deg: float = 0.0) -> list[dict]:
    scale, A, translation = _known_similarity()
    records = []
    for index, gt in enumerate(_gt_records()):
        C_gt = np.asarray(gt["camera_center_world"], dtype=np.float64)
        C_sfm = A.T @ ((C_gt - translation) / scale)
        if center_noise:
            C_sfm = C_sfm + center_noise * np.array([
                np.sin(index * 0.7), np.cos(index * 0.5), np.sin(index * 0.31)
            ])
        R_gt = np.asarray(gt["R_camera_from_world"], dtype=np.float64)
        noise = _rot_z(rotation_noise_deg * np.sin(index * 0.4))
        R_sfm = noise @ R_gt @ A
        t_sfm = -(R_sfm @ C_sfm)
        records.append({
            "name": gt["image_name"],
            "image_id": index + 1,
            "camera_id": 1,
            "R_camera_from_sfm_world": R_sfm,
            "t_camera_from_sfm_world": t_sfm,
            "camera_center_sfm_world": C_sfm,
        })
    return records


def _pose_worker_result(*, records: list[dict] | None = None) -> dict:
    source = records or _sfm_records()
    serialized = []
    for item in source:
        R = np.asarray(item["R_camera_from_sfm_world"], dtype=np.float64)
        t = np.asarray(item["t_camera_from_sfm_world"], dtype=np.float64)
        serialized.append({
            "name": item["name"],
            "image_id": item["image_id"],
            "camera_id": item["camera_id"],
            "camera_from_reconstruction": np.column_stack([R, t]).tolist(),
        })
    return {
        "schema_version": 1,
        "phase": "I4.2C-pose-export",
        "status": "ok",
        "read_only": True,
        "authoritative_sparse_modified": False,
        "registered_images": serialized,
    }


def _write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _evidence(tmp_path: Path) -> tuple[Path, Path]:
    a = tmp_path / "i4_2a"
    b = tmp_path / "i4_2b"
    a.mkdir()
    (b / "evidence").mkdir(parents=True)
    sparse = b / "workspace/authoritative_sparse"
    sparse.mkdir(parents=True)
    for filename in ("cameras.bin", "images.bin", "points3D.bin"):
        (sparse / filename).write_bytes((filename + "\n").encode())
    sparse_hashes = {name: hashlib.sha256((sparse / name).read_bytes()).hexdigest() for name in ("cameras.bin", "images.bin", "points3D.bin")}

    gt = {
        "schema_version": 1,
        "phase": "I4.2A",
        "dataset": "Middlebury TempleRing",
        "purpose": "held-out validation evidence only; forbidden as SfM pose input",
        "shared_intrinsics_K": K,
        "camera_records": _gt_records(),
        "sfm_pose_input_authorized": False,
        "bundle_adjustment_pose_prior_authorized": False,
        "metric_scale_claimed": False,
        "measurement_authoritative": False,
        "certified_metrology": False,
    }
    gt_path = _write_json(a / "i4_2a_ground_truth_validation_manifest.json", gt)
    recon_path = _write_json(a / "i4_2a_reconstruction_input_manifest.json", {"phase": "I4.2A", "image_count": 47, "shared_intrinsics_K": K})
    a_summary = {
        "schema_version": 1,
        "phase": "I4.2A",
        "status": "pass",
        "pass": True,
        "artifacts": {
            "ground_truth_validation_manifest_sha256": hashlib.sha256(gt_path.read_bytes()).hexdigest(),
            "reconstruction_input_manifest_sha256": hashlib.sha256(recon_path.read_bytes()).hexdigest(),
        },
    }
    a_summary_path = _write_json(a / "i4_2a_summary.json", a_summary)
    shutil.copy2(a_summary_path, b / "evidence/i4_2a_summary.json")
    shutil.copy2(recon_path, b / "evidence/i4_2a_reconstruction_input_manifest.json")

    b_summary = {
        "schema_version": 1,
        "phase": "I4.2B",
        "status": "pass",
        "pass": True,
        "dataset": {"shared_K": K},
        "validation": {
            "all_47_registered": True,
            "registered_images": 47,
            "fixed_intrinsics_preserved": True,
            "single_shared_camera": True,
            "generic_sfm_quality": {"status": "accepted", "usable": True},
            "authoritative_sparse_sha256": sparse_hashes,
        },
        "authority": {
            "single_authoritative_sfm_frame": True,
            "intrinsics_fixed": True,
            "ground_truth_extrinsics_consumed": False,
            "ground_truth_camera_centers_consumed": False,
            "ground_truth_angles_consumed": False,
            "mvs_started": False,
            "da3_started": False,
            "production_filter_policy_changed": False,
            "dense_provider_promoted": False,
            "i4_3_provider_architecture_authorized": False,
            "metric_scale_claimed": False,
            "measurement_authoritative": False,
            "certified_metrology": False,
            "surface_mesh_generated": False,
        },
        "artifacts": {"authoritative_sparse": str(sparse)},
    }
    _write_json(b / "i4_2b_summary.json", b_summary)
    return a, b


def _ready_probe() -> PycolmapEnvironmentProbe:
    return PycolmapEnvironmentProbe(
        python_executable=sys.executable,
        available=True,
        pycolmap_version="4.1.1",
        distribution="pycolmap-cuda12",
        distribution_version="4.1.1",
        cuda_devices=1,
        apis=tuple((name, True) for name in PYCOLMAP_REQUIRED_APIS),
    )


def test_camera_center_from_camera_from_world_round_trip() -> None:
    R = _rot_z(20.0) @ _rot_y(8.0)
    center = np.array([0.3, -0.7, 1.2])
    t = -(R @ center)
    recovered, recovered_R, recovered_t = camera_center_from_camera_from_world(np.column_stack([R, t]))
    assert np.allclose(recovered, center, atol=1e-12)
    assert np.allclose(recovered_R, R, atol=1e-12)
    assert np.allclose(recovered_t, t, atol=1e-12)


def test_umeyama_recovers_known_proper_similarity() -> None:
    gt = np.stack([np.asarray(item["camera_center_world"]) for item in _gt_records()])
    scale, A, translation = _known_similarity()
    sfm = ((gt - translation) @ A) / scale
    fitted = fit_similarity_umeyama(sfm, gt)
    assert fitted.scale == pytest.approx(scale, rel=0, abs=1e-10)
    assert np.allclose(fitted.rotation, A, atol=1e-10)
    assert np.allclose(fitted.translation, translation, atol=1e-10)
    assert np.linalg.det(fitted.rotation) == pytest.approx(1.0, abs=1e-10)


def test_rotation_error_is_zero_after_same_center_only_alignment() -> None:
    alignment, quality, _views, status = evaluate_trajectory(
        sfm_records=_sfm_records(), ground_truth_records=_gt_records()
    )
    assert status == "pass"
    assert quality["metrics"]["rotation_max_deg"] < 1e-5
    assert alignment["ground_truth_rotations_used_for_alignment"] is False


def test_exact_synthetic_trajectory_passes_strict_default_gates() -> None:
    alignment, quality, views, status = evaluate_trajectory(
        sfm_records=_sfm_records(), ground_truth_records=_gt_records()
    )
    assert status == "pass"
    assert quality["accepted"] is True
    assert quality["metrics"]["center_rmse_fraction_of_median_ring_radius"] < 1e-10
    assert len(views) == 47
    assert alignment["reflection_allowed"] is False


def test_center_error_above_predeclared_gate_is_degraded() -> None:
    _alignment, quality, _views, status = evaluate_trajectory(
        sfm_records=_sfm_records(center_noise=0.02), ground_truth_records=_gt_records()
    )
    assert status == "degraded"
    assert quality["accepted"] is False
    assert not all(quality["checks"].values())


def test_rotation_error_above_predeclared_gate_is_degraded() -> None:
    _alignment, quality, _views, status = evaluate_trajectory(
        sfm_records=_sfm_records(rotation_noise_deg=8.0), ground_truth_records=_gt_records()
    )
    assert status == "degraded"
    assert quality["metrics"]["rotation_max_deg"] > 5.0


def test_load_i42c_requires_exact_passing_a_and_b_evidence(tmp_path: Path) -> None:
    a, b = _evidence(tmp_path)
    inputs = load_i4_2c_inputs(i4_2a_dir=a, i4_2b_dir=b)
    assert len(inputs.ground_truth_records) == 47
    assert inputs.shared_K[0, 0] == pytest.approx(1520.4)
    assert inputs.expected_sparse_sha256["images.bin"]


def test_load_i42c_rejects_mutated_heldout_manifest(tmp_path: Path) -> None:
    a, b = _evidence(tmp_path)
    path = a / "i4_2a_ground_truth_validation_manifest.json"
    path.write_text(path.read_text(encoding="utf-8") + " ", encoding="utf-8")
    with pytest.raises(ReconstructionValidationError, match="ground-truth manifest checksum changed"):
        load_i4_2c_inputs(i4_2a_dir=a, i4_2b_dir=b)


def test_load_i42c_rejects_nonpassing_sfm(tmp_path: Path) -> None:
    a, b = _evidence(tmp_path)
    path = b / "i4_2b_summary.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["status"] = "degraded"
    payload["pass"] = False
    _write_json(path, payload)
    with pytest.raises(ReconstructionValidationError, match="passing I4.2B"):
        load_i4_2c_inputs(i4_2a_dir=a, i4_2b_dir=b)


def test_pose_export_result_rejects_missing_view() -> None:
    result = _pose_worker_result()
    result["registered_images"] = result["registered_images"][:-1]
    with pytest.raises(ReconstructionValidationError, match="exactly 47"):
        _validate_pose_export_result(result)


def test_full_runner_mock_worker_persists_validation_only_pass(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    a, b = _evidence(tmp_path)
    worker = tmp_path / "fake_pose_worker.py"
    worker.write_text("# fake\n", encoding="utf-8")
    monkeypatch.setattr(
        "core.reconstruction.temple_ring_trajectory_validation.probe_pycolmap_environment",
        lambda _python: _ready_probe(),
    )

    class Completed:
        returncode = 0

    def fake_run(command, **_kwargs):
        result_path = Path(command[command.index("--result") + 1])
        _write_json(result_path, _pose_worker_result())
        return Completed()

    monkeypatch.setattr("core.reconstruction.temple_ring_trajectory_validation.subprocess.run", fake_run)
    output = tmp_path / "i4_2c"
    result = run_temple_ring_heldout_trajectory_validation(
        i4_2a_dir=a,
        i4_2b_dir=b,
        output_dir=output,
        pycolmap_python=sys.executable,
        pose_export_worker_script=worker,
    )
    assert result["status"] == "pass"
    assert result["authority"]["ground_truth_extrinsics_consumed_for_validation"] is True
    assert result["authority"]["ground_truth_pose_input_to_sfm"] is False
    assert result["authority"]["similarity_applied_to_authoritative_sparse_model"] is False
    assert result["authority"]["mvs_started"] is False
    assert (output / "i4_2c_similarity_alignment.json").is_file()
    assert (output / "i4_2c_per_view_residuals.json").is_file()
