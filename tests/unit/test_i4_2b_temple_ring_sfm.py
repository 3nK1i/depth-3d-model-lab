from __future__ import annotations

import json
from pathlib import Path
import struct
import sys

import pytest

from core.reconstruction.errors import ReconstructionValidationError
from core.reconstruction.sfm_environment import PYCOLMAP_REQUIRED_APIS, PycolmapEnvironmentProbe
from core.reconstruction.temple_ring_preflight import run_i4_2a_preflight
from core.reconstruction.temple_ring_sfm import (
    TEMPLE_RING_EXPECTED_NAMES,
    TempleRingSfmConfig,
    _validate_worker_result,
    build_temple_ring_sfm_worker_request,
    load_i4_2a_reconstruction_inputs,
    run_temple_ring_authoritative_sfm,
    stage_temple_ring_images,
)


def _png(path: Path, *, width: int = 640, height: int = 480) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = b"\x89PNG\r\n\x1a\n" + struct.pack(">I", 13) + b"IHDR" + struct.pack(">II", width, height) + b"\x08\x02\x00\x00\x00"
    path.write_bytes(payload)
    return path


def _calibration_line(name: str, index: int) -> str:
    K = [1520.4, 0, 302.32, 0, 1525.9, 246.87, 0, 0, 1]
    R = [1, 0, 0, 0, 1, 0, 0, 0, 1]
    t = [float(index) * 0.01, float(index) * -0.02, 3.0]
    return " ".join([name] + [f"{value:.12g}" for value in K + R + t])


def _dataset(tmp_path: Path) -> Path:
    root = tmp_path / "templeRing"
    root.mkdir(parents=True)
    lines = []
    for index, name in enumerate(TEMPLE_RING_EXPECTED_NAMES, start=1):
        _png(root / name)
        lines.append(_calibration_line(name, index))
    (root / "templeR_par.txt").write_text("47\n" + "\n".join(lines) + "\n", encoding="utf-8")
    (root / "templeR_ang.txt").write_text("\n".join(f"0 0 {name}" for name in TEMPLE_RING_EXPECTED_NAMES) + "\n", encoding="utf-8")
    return root


def _closure(tmp_path: Path) -> Path:
    path = tmp_path / "closure.json"
    path.write_text(json.dumps({
        "phase": "I4.1.9A-A3.1",
        "status": "pass",
        "pass": True,
        "manual_review": {"verdict": "mvs_better_but_imperfect", "classification": "MVS clearly better than DA3, but imperfect"},
        "decision": {
            "house_mvs_positive_control_closed": True,
            "i4_1_9a_closed": True,
            "i4_2_detail_preservation_authorized": True,
            "i4_3_provider_architecture_authorized": False,
            "mvs_static_object_orbit_provider_candidate_supported": True,
            "random_dense_threshold_tuning_authorized": False,
        },
        "authority": {
            "camera_geometry_modified": False,
            "sfm_rerun": False,
            "bundle_adjustment_rerun": False,
            "pose_refinement": False,
            "dense_provider_promoted": False,
            "metric_scale_claimed": False,
            "measurement_authoritative": False,
            "certified_metrology": False,
            "surface_mesh_generated": False,
            "single_authoritative_sfm_frame": True,
        },
    }, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _preflight(tmp_path: Path) -> Path:
    output = tmp_path / "preflight"
    run_i4_2a_preflight(dataset_dir=_dataset(tmp_path), i4_1_9a_closure=_closure(tmp_path), output_dir=output)
    return output


def _fake_worker_result(*, registered: int = 47, params: list[float] | None = None) -> dict:
    names = list(TEMPLE_RING_EXPECTED_NAMES[:registered])
    return {
        "schema_version": 1,
        "phase": "I4.2B",
        "status": "ok",
        "worker_elapsed_s": 1.0,
        "camera": {
            "camera_count": 1,
            "model": "PINHOLE",
            "params": params or [1520.4, 1525.9, 302.32, 246.87],
            "intrinsics_fixed": True,
        },
        "registered_image_names": names,
        "statistics": {
            "registered_images": registered,
            "landmarks": 1000,
            "observations": 5000,
            "mean_reprojection_error_px": 0.5,
        },
        "authority": {
            "single_authoritative_sfm_frame": True,
            "ground_truth_extrinsics_consumed": False,
            "ground_truth_camera_centers_consumed": False,
            "ground_truth_angles_consumed": False,
            "metric_scale_claimed": False,
            "measurement_authoritative": False,
            "certified_metrology": False,
            "surface_mesh_generated": False,
            "mvs_started": False,
            "da3_started": False,
            "dense_provider_promoted": False,
            "i4_3_provider_architecture_authorized": False,
        },
    }


def _write_sparse(workspace: Path) -> None:
    sparse = workspace / "authoritative_sparse"
    sparse.mkdir(parents=True, exist_ok=True)
    for name in ("cameras.bin", "images.bin", "points3D.bin"):
        (sparse / name).write_bytes((name + "\n").encode())


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


def test_config_locks_full_resolution_exhaustive_fixed_intrinsics() -> None:
    config = TempleRingSfmConfig()
    payload = config.as_dict()
    assert payload["maximum_image_size"] == 640
    assert payload["matching_strategy"] == "exhaustive"
    assert payload["camera_model"] == "PINHOLE"
    assert payload["intrinsics_fixed"] is True
    assert payload["multiple_models"] is False
    assert payload["permanent_global_view_cap"] is False


def test_config_rejects_hidden_resolution_change() -> None:
    with pytest.raises(ReconstructionValidationError, match="must remain 640"):
        TempleRingSfmConfig(maximum_image_size=320)


def test_load_preflight_accepts_exact_47_without_gt_pose_leak(tmp_path: Path) -> None:
    inputs = load_i4_2a_reconstruction_inputs(_preflight(tmp_path))
    assert len(inputs.image_records) == 47
    assert [item["name"] for item in inputs.image_records] == list(TEMPLE_RING_EXPECTED_NAMES)
    assert inputs.shared_K[0, 0] == pytest.approx(1520.4)


def test_load_preflight_rejects_mutated_manifest(tmp_path: Path) -> None:
    root = _preflight(tmp_path)
    path = root / "i4_2a_reconstruction_input_manifest.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["image_count"] = 46
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ReconstructionValidationError, match="checksum changed"):
        load_i4_2a_reconstruction_inputs(root)


def test_load_preflight_rejects_changed_source_image(tmp_path: Path) -> None:
    root = _preflight(tmp_path)
    image = tmp_path / "templeRing/templeR0024.png"
    image.write_bytes(image.read_bytes() + b"mutated")
    with pytest.raises(ReconstructionValidationError, match="checksum changed"):
        load_i4_2a_reconstruction_inputs(root)


def test_stage_preserves_all_image_hashes(tmp_path: Path) -> None:
    inputs = load_i4_2a_reconstruction_inputs(_preflight(tmp_path))
    staged = stage_temple_ring_images(inputs, tmp_path / "staged")
    assert len(staged) == 47
    assert tuple(item["name"] for item in staged) == TEMPLE_RING_EXPECTED_NAMES
    assert all(Path(item["path"]).is_file() for item in staged)


def test_worker_request_contains_k_but_not_held_out_pose(tmp_path: Path) -> None:
    inputs = load_i4_2a_reconstruction_inputs(_preflight(tmp_path))
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    staged = stage_temple_ring_images(inputs, workspace / "images")
    request = build_temple_ring_sfm_worker_request(
        inputs=inputs, workspace=workspace, staged_images=staged, config=TempleRingSfmConfig()
    )
    text = json.dumps(request, sort_keys=True)
    assert request["shared_intrinsics"]["fx"] == pytest.approx(1520.4)
    assert request["config"]["matching_strategy"] == "exhaustive"
    assert "R_camera_from_world" not in text
    assert "t_camera_from_world" not in text
    assert "camera_center_world" not in text
    assert "templeR_ang.txt" not in text


def test_worker_result_pass_requires_all_47_and_fixed_k(tmp_path: Path) -> None:
    inputs = load_i4_2a_reconstruction_inputs(_preflight(tmp_path))
    workspace = tmp_path / "worker_workspace"
    _write_sparse(workspace)
    validation, status = _validate_worker_result(_fake_worker_result(), inputs=inputs, workspace=workspace)
    assert status == "pass"
    assert validation["all_47_registered"] is True
    assert validation["fixed_intrinsics_preserved"] is True
    assert validation["generic_sfm_quality"]["status"] == "accepted"


def test_worker_result_46_views_is_degraded_not_silently_accepted(tmp_path: Path) -> None:
    inputs = load_i4_2a_reconstruction_inputs(_preflight(tmp_path))
    workspace = tmp_path / "worker_workspace"
    _write_sparse(workspace)
    validation, status = _validate_worker_result(_fake_worker_result(registered=46), inputs=inputs, workspace=workspace)
    assert status == "degraded"
    assert validation["all_47_registered"] is False


def test_worker_result_rejects_intrinsics_drift(tmp_path: Path) -> None:
    inputs = load_i4_2a_reconstruction_inputs(_preflight(tmp_path))
    workspace = tmp_path / "worker_workspace"
    _write_sparse(workspace)
    with pytest.raises(ReconstructionValidationError, match="fixed benchmark intrinsics changed"):
        _validate_worker_result(
            _fake_worker_result(params=[1520.5, 1525.9, 302.32, 246.87]),
            inputs=inputs,
            workspace=workspace,
        )


def test_full_runner_with_mock_worker_persists_pass_and_never_loads_gt(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    preflight = _preflight(tmp_path)
    worker = tmp_path / "fake_worker.py"
    worker.write_text("# fake\n", encoding="utf-8")

    monkeypatch.setattr("core.reconstruction.temple_ring_sfm.probe_pycolmap_environment", lambda _python: _ready_probe())

    class Completed:
        returncode = 0

    def fake_run(command, **_kwargs):
        request_path = Path(command[command.index("--request") + 1])
        result_path = Path(command[command.index("--result") + 1])
        request = json.loads(request_path.read_text(encoding="utf-8"))
        workspace = Path(request["workspace"])
        _write_sparse(workspace)
        result_path.write_text(json.dumps(_fake_worker_result()) + "\n", encoding="utf-8")
        return Completed()

    monkeypatch.setattr("core.reconstruction.temple_ring_sfm.subprocess.run", fake_run)
    output = tmp_path / "out"
    result = run_temple_ring_authoritative_sfm(
        i4_2a_dir=preflight,
        output_dir=output,
        pycolmap_python=sys.executable,
        worker_script=worker,
    )
    assert result["status"] == "pass"
    assert result["authority"]["ground_truth_extrinsics_consumed"] is False
    assert result["authority"]["mvs_started"] is False
    assert (output / "i4_2b_summary.json").is_file()
