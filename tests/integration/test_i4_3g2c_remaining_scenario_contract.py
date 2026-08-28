from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from core.reconstruction.mvs_da3_matched_view import load_matched_view_sources


def _write_json(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_ply(path: Path) -> Path:
    path.write_text(
        "ply\nformat ascii 1.0\nelement vertex 1\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        "end_header\n0 0 0 255 255 255\n",
        encoding="ascii",
    )
    return path


def _verdict(scenario_id: str) -> dict:
    if scenario_id == "g2_complete_room_mvs_da3":
        return {
            "scenario_id": scenario_id,
            "scenario_classification": "operating_envelope_failure_authoritative_sfm_coverage",
            "full_room_authoritative_sfm_qualified": False,
            "full_room_provider_comparison_qualified": False,
            "localized_provider_comparison_valid": True,
            "execution_defect_established": False,
            "infrastructure_rerun_authorized": False,
        }
    if scenario_id == "g2_low_texture_mvs_da3":
        return {
            "scenario_id": scenario_id,
            "scenario_classification": "operating_envelope_failure_authoritative_sfm_coverage",
            "authoritative_sfm_qualified_for_full_scenario": False,
            "full_scenario_provider_comparison_qualified": False,
            "localized_provider_comparison_valid": True,
            "execution_defect_established": False,
            "infrastructure_rerun_authorized": False,
        }
    if scenario_id == "g2_weak_overlap_mvs_da3":
        return {
            "scenario_id": scenario_id,
            "authoritative_sfm_connectivity_qualified": True,
            "production_native_comparison_valid": True,
            "mvs_status": "pass",
            "da3_status": "pass",
            "same_authoritative_sfm_shared": True,
            "execution_defect_established": False,
            "infrastructure_rerun_authorized": False,
            "scenario_verdict_pending_g2c_manual_review": True,
        }
    raise AssertionError(scenario_id)


def _production_fixture(tmp_path: Path, scenario_id: str):
    sfm = _write_json(
        tmp_path / "sfm.json",
        {
            "landmarks": [
                {"position_reconstruction": [0.0, 0.0, 0.0]},
                {"position_reconstruction": [1.0, 0.0, 0.0]},
                {"position_reconstruction": [0.0, 1.0, 1.0]},
                {"position_reconstruction": [1.0, 1.0, 0.5]},
            ]
        },
    )
    sfm_sha = _sha(sfm)
    frame_id = f"frame:{scenario_id}"
    mvs_ply = _write_ply(tmp_path / "mvs.ply")
    da3_ply = _write_ply(tmp_path / "da3.ply")

    def evidence(provider: str, ply: Path):
        return {
            "provider_implementation": provider,
            "authoritative_sfm_identity": {
                "snapshot_path": str(sfm),
                "snapshot_sha256": sfm_sha,
                "frame_id": frame_id,
            },
            "artifacts": [{"path": str(ply), "sha256": _sha(ply)}],
        }

    mvs_ev = _write_json(tmp_path / "mvs.json", evidence("colmap_mvs", mvs_ply))
    da3_ev = _write_json(tmp_path / "da3.json", evidence("da3", da3_ply))
    prod = _write_json(
        tmp_path / "prod.json",
        {
            "status": "production_native_complete",
            "pass": True,
            "scenario": {"scenario_id": scenario_id},
            "authority": {
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
            },
            "providers": {
                "colmap_mvs": {"evidence": {"path": str(mvs_ev), "sha256": _sha(mvs_ev)}},
                "da3": {"evidence": {"path": str(da3_ev), "sha256": _sha(da3_ev)}},
            },
        },
    )
    verdict = _write_json(tmp_path / "verdict.json", _verdict(scenario_id))
    return prod, verdict, sfm_sha, frame_id


@pytest.mark.parametrize(
    "scenario_id",
    [
        "g2_complete_room_mvs_da3",
        "g2_low_texture_mvs_da3",
        "g2_weak_overlap_mvs_da3",
    ],
)
def test_generic_loader_accepts_remaining_frozen_g2_scenarios(tmp_path: Path, scenario_id: str):
    prod, verdict, sfm_sha, frame_id = _production_fixture(tmp_path, scenario_id)
    result = load_matched_view_sources(
        prod,
        verdict,
        expected_production_sha256=_sha(prod),
        expected_verdict_sha256=_sha(verdict),
        expected_scenario_id=scenario_id,
    )
    assert result.scenario_id == scenario_id
    assert result.sfm_snapshot_sha256 == sfm_sha
    assert result.frame_id == frame_id
    assert result.shared_frame.robust_diagonal > 0.0
    assert result.mvs_ply.is_file()
    assert result.da3_ply.is_file()


def test_generic_g2c_runner_is_g1_and_g2b_bound_and_never_reruns_providers():
    root = Path(__file__).resolve().parents[2]
    runner = (root / "scripts" / "run_i4_3g2c_matched_view_scenario.py").read_text(encoding="utf-8")
    assert "verify_g1_manual_review_requirement" in runner
    assert "verify_g2b_scenario_closure_for_g2c" in runner
    assert "load_matched_view_sources" in runner
    assert "equal_render_density_enforced" in runner
    assert "review_scope" in runner
    forbidden = ("run_sfm", "run_mvs", "run_da3", "bundle_adjust", "patch_match")
    assert all(token not in runner for token in forbidden)
