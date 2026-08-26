from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import pytest

from core.reconstruction.errors import ReconstructionValidationError
from core.reconstruction.mvs_environment import MvsGpuResource, MvsHostResourceProbe
from core.reconstruction.temple_ring_mvs import (
    TEMPLE_RING_DEFAULT_PATCH_MATCH_CACHE_GIB,
    TEMPLE_RING_DEFAULT_SOURCE_NEIGHBORS,
    TEMPLE_RING_DEFAULT_STEREO_FUSION_CACHE_GIB,
    TEMPLE_RING_EXPECTED_NAMES,
    TEMPLE_RING_EXPECTED_VIEWS,
    TempleRingMvsConfig,
    _validate_worker_result,
    build_temple_ring_mvs_request,
    load_i4_2d_inputs,
)


def _write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _evidence(tmp_path: Path) -> tuple[Path, Path]:
    b = tmp_path / "i4_2b"
    c = tmp_path / "i4_2c"
    images = b / "workspace/images"
    sparse = b / "workspace/authoritative_sparse"
    worker = b / "worker"
    images.mkdir(parents=True)
    sparse.mkdir(parents=True)
    worker.mkdir(parents=True)
    c.mkdir(parents=True)

    image_hashes = {}
    for index, name in enumerate(TEMPLE_RING_EXPECTED_NAMES):
        path = images / name
        path.write_bytes(f"image-{index}\n".encode())
        image_hashes[name] = _sha(path)
    for filename in ("cameras.bin", "images.bin", "points3D.bin", "frames.bin", "rigs.bin"):
        (sparse / filename).write_bytes((filename + "\n").encode())
    primary_hashes = {name: _sha(sparse / name) for name in ("cameras.bin", "images.bin", "points3D.bin")}

    b_request = {
        "schema_version": 1,
        "phase": "I4.2B",
        "action": "authoritative_temple_ring_sfm",
        "image_names": list(TEMPLE_RING_EXPECTED_NAMES),
        "image_sha256": image_hashes,
        "shared_intrinsics": {
            "model": "PINHOLE", "width": 640, "height": 480,
            "fx": 1520.4, "fy": 1525.9, "cx": 302.32, "cy": 246.87,
        },
        "authority": {
            "ground_truth_extrinsics_consumed": False,
            "ground_truth_camera_centers_consumed": False,
            "ground_truth_angles_consumed": False,
        },
    }
    _write_json(worker / "request.json", b_request)
    b_summary = {
        "schema_version": 1,
        "phase": "I4.2B",
        "status": "pass",
        "pass": True,
        "validation": {
            "all_47_registered": True,
            "registered_images": 47,
            "fixed_intrinsics_preserved": True,
            "single_shared_camera": True,
            "authoritative_sparse_sha256": primary_hashes,
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
        "artifacts": {
            "images": str(images),
            "authoritative_sparse": str(sparse),
            "worker_request": str(worker / "request.json"),
        },
    }
    b_summary_path = _write_json(b / "i4_2b_summary.json", b_summary)
    b_summary_hash = _sha(b_summary_path)
    c_summary = {
        "schema_version": 1,
        "phase": "I4.2C",
        "status": "pass",
        "pass": True,
        "trajectory_quality": {"accepted": True, "status": "pass"},
        "authority": {
            "authoritative_sparse_modified": False,
            "single_authoritative_sfm_frame_preserved": True,
            "bundle_adjustment_rerun": False,
            "da3_started": False,
            "dense_provider_promoted": False,
            "i4_3_provider_architecture_authorized": False,
            "measurement_authoritative": False,
            "metric_scale_claimed": False,
            "mvs_started": False,
            "pose_refinement": False,
            "production_filter_policy_changed": False,
            "sfm_rerun": False,
            "similarity_applied_to_authoritative_sparse_model": False,
            "surface_mesh_generated": False,
        },
        "evidence": {
            "authoritative_sparse_sha256_before": primary_hashes,
            "authoritative_sparse_sha256_after": primary_hashes,
            "source_sha256": {"i4_2b_summary": b_summary_hash},
        },
    }
    _write_json(c / "i4_2c_summary.json", c_summary)
    return b, c


def _resources(tmp_path: Path) -> MvsHostResourceProbe:
    return MvsHostResourceProbe(
        workspace_path=str(tmp_path),
        disk_free_gib=30.0,
        ram_available_gib=16.0,
        gpus=(MvsGpuResource(index=0, name="Test GPU", total_vram_gib=8.0, free_vram_gib=6.0),),
    )


def _load_worker_module():
    root = Path(__file__).resolve().parents[2]
    spec = importlib.util.spec_from_file_location(
        "i4_2d_worker_test_module",
        root / "scripts/reconstruction_pycolmap_temple_ring_mvs_worker.py",
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_i42d_defaults_keep_all_47_references_and_only_bound_local_neighbors() -> None:
    config = TempleRingMvsConfig()
    assert TEMPLE_RING_EXPECTED_VIEWS == 47
    assert config.source_neighbors == TEMPLE_RING_DEFAULT_SOURCE_NEIGHBORS == 10
    assert config.patch_match_cache_gib == TEMPLE_RING_DEFAULT_PATCH_MATCH_CACHE_GIB == 2.0
    assert config.stereo_fusion_cache_gib == TEMPLE_RING_DEFAULT_STEREO_FUSION_CACHE_GIB == 2.0
    payload = config.as_dict()
    assert payload["all_reference_views_available"] == 47
    assert payload["source_selection"] == "__auto__, 10"
    assert payload["permanent_global_view_cap"] is False
    assert payload["production_manifold_applied"] is False
    assert payload["quality_thresholds_relaxed"] is False


def test_i42d_rejects_disabling_geometric_consistency() -> None:
    with pytest.raises(ReconstructionValidationError, match="requires geometric consistency"):
        TempleRingMvsConfig(geometric_consistency=False)


def test_i42d_rejects_source_neighbor_globalization() -> None:
    with pytest.raises(ReconstructionValidationError):
        TempleRingMvsConfig(source_neighbors=47)
    with pytest.raises(ReconstructionValidationError):
        TempleRingMvsConfig(source_neighbors=0)


def test_i42d_loads_only_exact_passing_b_c_and_checksum_bound_inputs(tmp_path: Path) -> None:
    b, c = _evidence(tmp_path)
    inputs = load_i4_2d_inputs(i4_2b_dir=b, i4_2c_dir=c)
    assert inputs.image_names == TEMPLE_RING_EXPECTED_NAMES
    assert len(inputs.image_hash_map) == 47
    assert set(inputs.sparse_hash_map) == {"cameras.bin", "images.bin", "points3D.bin", "frames.bin", "rigs.bin"}


def test_i42d_request_contains_no_gt_and_no_production_filter(tmp_path: Path) -> None:
    b, c = _evidence(tmp_path)
    inputs = load_i4_2d_inputs(i4_2b_dir=b, i4_2c_dir=c)
    request = build_temple_ring_mvs_request(
        inputs=inputs,
        output_dir=tmp_path / "out",
        config=TempleRingMvsConfig(),
        resources=_resources(tmp_path),
    )
    assert request["expected_registered_views"] == 47
    assert request["image_names"] == list(TEMPLE_RING_EXPECTED_NAMES)
    assert request["authority"]["held_out_ground_truth_consumed"] is False
    assert request["authority"]["production_filter_policy_changed"] is False
    assert request["authority"]["production_manifold_applied"] is False
    assert request["authority"]["i4_3_provider_architecture_authorized"] is False


def test_i42d_patch_match_cfg_requires_all_47_references_and_bounded_sources(tmp_path: Path) -> None:
    worker = _load_worker_module()
    cfg = tmp_path / "patch-match.cfg"
    cfg.write_text("".join(f"{name}\n__auto__, 10\n" for name in TEMPLE_RING_EXPECTED_NAMES), encoding="utf-8")
    result = worker.parse_patch_match_cfg(cfg, list(TEMPLE_RING_EXPECTED_NAMES), 10)
    assert result["pass"]
    assert result["reference_views"] == 47
    assert result["source_limit_max"] == 10


def test_i42d_patch_match_cfg_rejects_all_sources_default(tmp_path: Path) -> None:
    worker = _load_worker_module()
    cfg = tmp_path / "patch-match.cfg"
    cfg.write_text("".join(f"{name}\n__all__\n" for name in TEMPLE_RING_EXPECTED_NAMES), encoding="utf-8")
    result = worker.parse_patch_match_cfg(cfg, list(TEMPLE_RING_EXPECTED_NAMES), 10)
    assert result["pass"] is False
    assert result["checks"]["no_all_sources_default"] is False


def test_i42d_fused_ply_header_requires_positive_vertex_count(tmp_path: Path) -> None:
    worker = _load_worker_module()
    ply = tmp_path / "fused.ply"
    ply.write_bytes(b"ply\nformat binary_little_endian 1.0\nelement vertex 123\nproperty float x\nend_header\n")
    assert worker.ply_vertex_count(ply) == 123


def _passing_worker_result(inputs) -> dict:
    return {
        "phase": "I4.2D",
        "status": "pass",
        "sfm_rerun": False,
        "bundle_adjustment_rerun": False,
        "pose_refinement": False,
        "held_out_ground_truth_consumed": False,
        "metric_scale_claimed": False,
        "measurement_authoritative": False,
        "certified_metrology": False,
        "surface_mesh_generated": False,
        "production_filter_policy_changed": False,
        "production_manifold_applied": False,
        "dense_provider_promoted": False,
        "i4_3_provider_architecture_authorized": False,
        "heavy_mvs_started": True,
        "image_undistortion_started": True,
        "image_undistortion_completed": True,
        "patch_match_started": True,
        "patch_match_completed": True,
        "stereo_fusion_started": True,
        "stereo_fusion_completed": True,
        "dense_sparse_validation": {"pass": True},
        "patch_match_config_validation": {"pass": True},
        "patch_match_outputs": {"geometric_depth_maps": 47, "geometric_normal_maps": 47},
        "fusion": {
            "ply_vertex_count": 1234,
            "input_type": "geometric",
            "production_manifold_applied": False,
        },
        "detail_qualification_pending": True,
        "authoritative_sparse_sha256_after": inputs.sparse_hash_map,
    }


def test_i42d_worker_result_validation_requires_complete_raw_mvs_evidence(tmp_path: Path) -> None:
    b, c = _evidence(tmp_path)
    inputs = load_i4_2d_inputs(i4_2b_dir=b, i4_2c_dir=c)
    _validate_worker_result(_passing_worker_result(inputs), inputs=inputs)


def test_i42d_worker_result_validation_rejects_hidden_manifold_or_missing_map(tmp_path: Path) -> None:
    b, c = _evidence(tmp_path)
    inputs = load_i4_2d_inputs(i4_2b_dir=b, i4_2c_dir=c)
    result = _passing_worker_result(inputs)
    result["production_manifold_applied"] = True
    with pytest.raises(ReconstructionValidationError, match="production_manifold_applied=True"):
        _validate_worker_result(result, inputs=inputs)
    result = _passing_worker_result(inputs)
    result["patch_match_outputs"]["geometric_depth_maps"] = 46
    with pytest.raises(ReconstructionValidationError, match="47 geometric depth maps"):
        _validate_worker_result(result, inputs=inputs)
