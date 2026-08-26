from __future__ import annotations

import json
from pathlib import Path
import struct

import numpy as np
import pytest

from core.reconstruction.errors import ReconstructionValidationError
from core.reconstruction.temple_ring_preflight import (
    TEMPLE_RING_EXPECTED_VIEWS,
    TEMPLE_RING_REVIEW_DIMENSIONS,
    build_i4_2a_evidence,
    inspect_temple_ring_dataset,
    load_temple_ring_calibration,
    parse_calibration_line,
    read_png_size,
    run_i4_2a_preflight,
    validate_i4_1_9a_closure,
)


def _png(path: Path, *, width: int = 640, height: int = 480) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = b"\x89PNG\r\n\x1a\n" + struct.pack(">I", 13) + b"IHDR" + struct.pack(">II", width, height) + b"\x08\x02\x00\x00\x00"
    path.write_bytes(payload)
    return path


def _calibration_line(name: str, index: int, *, fx: float = 1520.4) -> str:
    K = [fx, 0, 302.32, 0, 1525.9, 246.87, 0, 0, 1]
    R = [1, 0, 0, 0, 1, 0, 0, 0, 1]
    t = [float(index) * 0.01, float(index) * -0.02, 3.0]
    return " ".join([name] + [f"{value:.12g}" for value in K + R + t])


def _dataset(tmp_path: Path) -> Path:
    root = tmp_path / "templeRing"
    root.mkdir(parents=True)
    lines = []
    for index in range(1, TEMPLE_RING_EXPECTED_VIEWS + 1):
        name = f"templeR{index:04d}.png"
        _png(root / name)
        lines.append(_calibration_line(name, index))
    (root / "templeR_par.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (root / "templeR_ang.txt").write_text("angles\n", encoding="utf-8")
    return root


def _closure(tmp_path: Path) -> Path:
    path = tmp_path / "i4_1_9a_a3_1_closure.json"
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


def test_parse_calibration_line_preserves_k_and_held_out_pose() -> None:
    record = parse_calibration_line(_calibration_line("templeR0001.png", 1), line_number=1)
    assert record.image_name == "templeR0001.png"
    assert record.K[0, 0] == pytest.approx(1520.4)
    assert np.allclose(record.R, np.eye(3))
    assert np.allclose(record.camera_center, -record.t)


def test_parse_calibration_rejects_non_rigid_rotation() -> None:
    tokens = _calibration_line("templeR0001.png", 1).split()
    tokens[10] = "2"  # r11
    with pytest.raises(ReconstructionValidationError, match="proper rigid rotation"):
        parse_calibration_line(" ".join(tokens), line_number=1)


def test_png_size_reads_expected_header(tmp_path: Path) -> None:
    path = _png(tmp_path / "templeR0001.png")
    assert read_png_size(path) == (640, 480)


def test_png_size_rejects_non_png(tmp_path: Path) -> None:
    path = tmp_path / "templeR0001.png"
    path.write_bytes(b"not-png")
    with pytest.raises(ReconstructionValidationError, match="must be PNG"):
        read_png_size(path)


def test_dataset_accepts_exact_47_view_ring(tmp_path: Path) -> None:
    dataset = inspect_temple_ring_dataset(_dataset(tmp_path))
    assert len(dataset["images"]) == 47
    assert len(dataset["calibration_records"]) == 47
    assert dataset["shared_K_max_abs_spread"] == pytest.approx(0.0)
    assert dataset["cross_dataset_contamination"] == []


def test_dataset_rejects_missing_view(tmp_path: Path) -> None:
    root = _dataset(tmp_path)
    (root / "templeR0047.png").unlink()
    with pytest.raises(ReconstructionValidationError, match="image/calibration set mismatch"):
        inspect_temple_ring_dataset(root)


def test_dataset_rejects_cross_temple_dataset_image(tmp_path: Path) -> None:
    root = _dataset(tmp_path)
    _png(root / "temple0001.png")
    with pytest.raises(ReconstructionValidationError, match="cross-dataset"):
        inspect_temple_ring_dataset(root)


def test_dataset_rejects_inconsistent_shared_k(tmp_path: Path) -> None:
    root = _dataset(tmp_path)
    lines = (root / "templeR_par.txt").read_text(encoding="utf-8").splitlines()
    lines[-1] = _calibration_line("templeR0047.png", 47, fx=1521.4)
    (root / "templeR_par.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    with pytest.raises(ReconstructionValidationError, match="single shared intrinsic"):
        load_temple_ring_calibration(root / "templeR_par.txt")


def test_upstream_closure_must_authorize_i42_but_not_i43(tmp_path: Path) -> None:
    closure = _closure(tmp_path)
    info = validate_i4_1_9a_closure(closure)
    assert info["verdict"] == "mvs_better_but_imperfect"
    payload = json.loads(closure.read_text(encoding="utf-8"))
    payload["decision"]["i4_3_provider_architecture_authorized"] = True
    closure.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ReconstructionValidationError, match="I4.3"):
        validate_i4_1_9a_closure(closure)


def test_reconstruction_manifest_contains_k_but_never_ground_truth_pose(tmp_path: Path) -> None:
    dataset = inspect_temple_ring_dataset(_dataset(tmp_path))
    upstream = validate_i4_1_9a_closure(_closure(tmp_path))
    reconstruction, ground_truth, summary = build_i4_2a_evidence(dataset=dataset, upstream_closure=upstream)
    text = json.dumps(reconstruction, sort_keys=True)
    assert reconstruction["image_count"] == 47
    assert reconstruction["ground_truth_extrinsics_included"] is False
    assert "R_camera_from_world" not in text
    assert "t_camera_from_world" not in text
    assert "camera_center_world" not in text
    assert len(ground_truth["camera_records"]) == 47
    assert summary["evaluation_contract"]["production_manifold_policy_under_test"] == "d12_sparse"
    assert tuple(summary["evaluation_contract"]["review_dimensions"]) == TEMPLE_RING_REVIEW_DIMENSIONS


def test_full_preflight_writes_separate_immutable_manifests(tmp_path: Path) -> None:
    output = tmp_path / "out"
    result = run_i4_2a_preflight(dataset_dir=_dataset(tmp_path), i4_1_9a_closure=_closure(tmp_path), output_dir=output)
    assert result["status"] == "pass"
    assert (output / "i4_2a_reconstruction_input_manifest.json").is_file()
    assert (output / "i4_2a_ground_truth_validation_manifest.json").is_file()
    assert (output / "i4_2a_summary.json").is_file()
    with pytest.raises(ReconstructionValidationError, match="refusing to overwrite"):
        run_i4_2a_preflight(dataset_dir=tmp_path / "templeRing", i4_1_9a_closure=tmp_path / "i4_1_9a_a3_1_closure.json", output_dir=output)
