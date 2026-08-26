from __future__ import annotations

import json
from pathlib import Path

import pytest

from core.reconstruction.errors import ReconstructionValidationError
from core.reconstruction.mvs_execution import (
    A1MvsAuthority,
    MVS_HOUSE_DEFAULT_PATCH_MATCH_CACHE_GIB,
    MVS_HOUSE_DEFAULT_SOURCE_NEIGHBORS,
    MVS_HOUSE_DEFAULT_STEREO_FUSION_CACHE_GIB,
    MVS_HOUSE_EXPECTED_REGISTERED_VIEWS,
    MvsExecutionConfig,
    build_house_mvs_execution_request,
)
from core.reconstruction.mvs_environment import MvsHostResourceProbe


def test_a2_defaults_keep_all_68_references_and_bound_only_local_neighbors(tmp_path: Path) -> None:
    config = MvsExecutionConfig()
    assert MVS_HOUSE_EXPECTED_REGISTERED_VIEWS == 68
    assert config.source_neighbors == MVS_HOUSE_DEFAULT_SOURCE_NEIGHBORS == 10
    assert config.patch_match_cache_gib == MVS_HOUSE_DEFAULT_PATCH_MATCH_CACHE_GIB == 2.0
    assert config.stereo_fusion_cache_gib == MVS_HOUSE_DEFAULT_STEREO_FUSION_CACHE_GIB == 2.0
    payload = config.as_dict()
    assert payload["all_reference_views_available"] == 68
    assert payload["source_selection"] == "__auto__, 10"
    assert payload["permanent_global_view_cap"] is False
    assert payload["quality_thresholds_relaxed"] is False
    assert payload["geometric_consistency"] is True


def test_first_house_control_rejects_disabling_geometric_consistency() -> None:
    with pytest.raises(ReconstructionValidationError, match="requires geometric consistency"):
        MvsExecutionConfig(geometric_consistency=False)


def test_source_neighbor_limit_cannot_turn_into_all_other_views() -> None:
    with pytest.raises(ReconstructionValidationError):
        MvsExecutionConfig(source_neighbors=68)
    with pytest.raises(ReconstructionValidationError):
        MvsExecutionConfig(source_neighbors=0)


def test_execution_request_carries_authority_fingerprints_and_forbidden_flags(tmp_path: Path) -> None:
    sparse = tmp_path / "sparse"; sparse.mkdir()
    for name, content in (("cameras.txt", "c"), ("images.txt", "i"), ("points3D.txt", "p")):
        (sparse / name).write_text(content, encoding="utf-8")
    images = tmp_path / "images"; images.mkdir()
    names = tuple(f"{idx:02d}.jpg" for idx in range(68))
    authority = A1MvsAuthority(
        workspace=tmp_path,
        images=images,
        sparse=sparse,
        summary=tmp_path / "summary.json",
        manifest=tmp_path / "manifest.json",
        image_names=names,
        camera_intrinsics_fingerprint="a" * 64,
        camera_pose_fingerprint_68="b" * 64,
    )
    resources = MvsHostResourceProbe(
        workspace_path=str(tmp_path), disk_free_gib=20.0, ram_available_gib=12.0, gpus=()
    )
    request = build_house_mvs_execution_request(
        a1=authority,
        output_dir=tmp_path / "out",
        config=MvsExecutionConfig(),
        resources=resources,
    )
    assert request["expected_registered_views"] == 68
    assert request["camera_intrinsics_fingerprint"] == "a" * 64
    assert request["camera_pose_fingerprint_68"] == "b" * 64
    assert request["authority"]["sfm_rerun"] is False
    assert request["authority"]["bundle_adjustment_rerun"] is False
    assert request["authority"]["surface_mesh_generated"] is False
    assert request["config"]["permanent_global_view_cap"] is False


def _load_worker_module():
    import importlib.util
    root = Path(__file__).resolve().parents[2]
    spec = importlib.util.spec_from_file_location(
        "i4_1_9a_a2_worker_test_module",
        root / "scripts/reconstruction_pycolmap_mvs_worker.py",
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_patch_match_cfg_requires_all_68_references_and_ten_or_fewer_sources(tmp_path: Path) -> None:
    worker = _load_worker_module()
    names = [f"{idx:02d}.jpg" for idx in range(68)]
    cfg = tmp_path / "patch-match.cfg"
    cfg.write_text("".join(f"{name}\n__auto__, 10\n" for name in names), encoding="utf-8")
    result = worker.parse_patch_match_cfg(cfg, names, 10)
    assert result["pass"]
    assert result["reference_views"] == 68
    assert result["source_limit_max"] == 10


def test_fused_ply_header_must_report_nonzero_vertex_count(tmp_path: Path) -> None:
    worker = _load_worker_module()
    ply = tmp_path / "fused.ply"
    ply.write_bytes(
        b"ply\nformat binary_little_endian 1.0\nelement vertex 321\nproperty float x\nend_header\n"
    )
    assert worker.ply_vertex_count(ply) == 321
