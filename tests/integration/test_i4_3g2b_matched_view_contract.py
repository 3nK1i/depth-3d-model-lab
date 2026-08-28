from __future__ import annotations

import hashlib
import json
from pathlib import Path

from core.reconstruction.mvs_da3_matched_view import load_matched_view_sources


def _write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return path


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _ply(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "ply\nformat ascii 1.0\nelement vertex 1\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        "end_header\n0 0 0 255 255 255\n"
    )
    return path


def test_load_sources_verifies_shared_sfm_and_frozen_artifacts(tmp_path: Path):
    sfm = _write_json(tmp_path / "sfm.json", {
        "landmarks": [
            {"position_reconstruction": [-1, 0, 0]},
            {"position_reconstruction": [1, 0, 0]},
            {"position_reconstruction": [0, 1, 1]},
            {"position_reconstruction": [0, -1, 0.5]},
        ]
    })
    sfm_sha = _sha(sfm)
    mvs_ply = _ply(tmp_path / "mvs.ply")
    da3_ply = _ply(tmp_path / "da3.ply")
    common = {
        "authoritative_sfm_identity": {
            "snapshot_path": str(sfm), "snapshot_sha256": sfm_sha, "frame_id": "g2b:test:sfm"
        },
    }
    mvs_ev = _write_json(tmp_path / "mvs_ev.json", {
        **common,
        "artifacts": [{"path": str(mvs_ply), "sha256": _sha(mvs_ply), "size_bytes": mvs_ply.stat().st_size}],
    })
    da3_ev = _write_json(tmp_path / "da3_ev.json", {
        **common,
        "artifacts": [{"path": str(da3_ply), "sha256": _sha(da3_ply), "size_bytes": da3_ply.stat().st_size}],
    })
    authority = {
        "camera_pose_authority_owner": "pycolmap_sfm",
        "same_authoritative_sfm_shared_by_both_providers": True,
        "authoritative_sfm_modified": False,
        "provider_selected": False,
        "provider_promoted": False,
        "auto_policy_implemented": False,
        "hybrid_authorized": False,
        "metric_scale_claimed": False,
        "measurement_authoritative": False,
        "certified_metrology": False,
        "surface_mesh_generated": False,
        "permanent_global_view_cap": None,
    }
    prod = _write_json(tmp_path / "prod.json", {
        "status": "production_native_complete", "pass": True,
        "scenario": {"scenario_id": "g2_temple_ring_mvs_da3"},
        "providers": {
            "colmap_mvs": {"evidence": {"path": str(mvs_ev), "sha256": _sha(mvs_ev)}},
            "da3": {"evidence": {"path": str(da3_ev), "sha256": _sha(da3_ev)}},
        },
        "authority": authority,
    })
    verdict = _write_json(tmp_path / "verdict.json", {
        "scenario_id": "g2_temple_ring_mvs_da3",
        "production_native_comparison_valid": True,
        "matched_view_diagnostic_authorized": True,
        "production_native_rerun_authorized": False,
    })
    sources = load_matched_view_sources(
        prod, verdict,
        expected_production_sha256=_sha(prod),
        expected_verdict_sha256=_sha(verdict),
    )
    assert sources.sfm_snapshot_sha256 == sfm_sha
    assert sources.mvs_ply == mvs_ply.resolve()
    assert sources.da3_ply == da3_ply.resolve()
    assert sources.frame_id == "g2b:test:sfm"
    assert sources.shared_frame.robust_diagonal > 0
