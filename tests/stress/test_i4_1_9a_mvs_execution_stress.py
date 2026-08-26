from __future__ import annotations

import json
from pathlib import Path

import pytest

from core.reconstruction.errors import ReconstructionValidationError
from core.reconstruction.mvs_execution import (
    MvsExecutionConfig,
    _require_clean_output,
    load_a1_mvs_authority,
)


def test_non_empty_a2_attempt_is_never_overwritten(tmp_path: Path) -> None:
    output = tmp_path / "attempt01"
    output.mkdir()
    marker = output / "preserve.txt"
    marker.write_text("keep", encoding="utf-8")
    with pytest.raises(ReconstructionValidationError, match="refusing to overwrite MVS evidence"):
        _require_clean_output(output)
    assert marker.read_text(encoding="utf-8") == "keep"


def test_invalid_gpu_index_fails_before_heavy_execution() -> None:
    with pytest.raises(ReconstructionValidationError, match="gpu_index"):
        MvsExecutionConfig(gpu_index="gpu0")


def test_cache_sizes_must_be_positive() -> None:
    with pytest.raises(ReconstructionValidationError):
        MvsExecutionConfig(patch_match_cache_gib=0.0)
    with pytest.raises(ReconstructionValidationError):
        MvsExecutionConfig(stereo_fusion_cache_gib=-1.0)


def test_a1_non_pass_summary_blocks_a2_before_worker(tmp_path: Path) -> None:
    root = tmp_path / "a1"; root.mkdir()
    (root / "i4_1_9a_a1_summary.json").write_text(
        json.dumps({"phase": "I4.1.9A-A1", "status": "fail"}), encoding="utf-8"
    )
    evidence = root / "evidence"; evidence.mkdir()
    (evidence / "a1_camera_authority_manifest.json").write_text(
        json.dumps({"phase": "I4.1.9A-A1"}), encoding="utf-8"
    )
    with pytest.raises(ReconstructionValidationError, match="requires an A1 PASS"):
        load_a1_mvs_authority(root)


def test_a1_authority_promotion_blocks_a2(tmp_path: Path) -> None:
    root = tmp_path / "a1"; root.mkdir()
    summary = {
        "phase": "I4.1.9A-A1",
        "status": "pass",
        "counts": {"total_registered_views": 68},
        "authority": {
            "single_authoritative_sfm_frame": True,
            "sfm_rerun": False,
            "bundle_adjustment_rerun": False,
            "pose_refinement": False,
            "camera_geometry_modified": False,
            "camera_intrinsics_modified": False,
            "house_k_txt_consumed": False,
            "metric_scale_claimed": True,
            "measurement_authoritative": False,
            "certified_metrology": False,
            "surface_mesh_generated": False,
            "heavy_mvs_started": False,
            "image_undistortion_started": False,
            "patch_match_started": False,
            "stereo_fusion_started": False,
        },
    }
    (root / "i4_1_9a_a1_summary.json").write_text(json.dumps(summary), encoding="utf-8")
    evidence = root / "evidence"; evidence.mkdir()
    (evidence / "a1_camera_authority_manifest.json").write_text(
        json.dumps({"phase": "I4.1.9A-A1", "views": [], "camera_intrinsics_fingerprint": "a"*64, "camera_pose_fingerprint_68": "b"*64}),
        encoding="utf-8",
    )
    with pytest.raises(ReconstructionValidationError, match="inconsistent with a pre-MVS bridge"):
        load_a1_mvs_authority(root)
