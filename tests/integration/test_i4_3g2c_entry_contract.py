from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from core.reconstruction.errors import ReconstructionValidationError
from core.reconstruction.mvs_da3_g2c_review import (
    blinded_render_label,
    resolve_equal_render_point_count,
    verify_g2b_closure_for_g2c,
)


def _write(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _closure(verdict_sha: str) -> dict:
    return {
        "phase": "I4.3G2B",
        "status": "closed",
        "i4_3g2b_closed": True,
        "i4_3g2c_ready": True,
        "all_five_g2_scenarios_accounted_for": True,
        "real_provider_execution_phase_complete": True,
        "scenarios": {
            "g2_temple_ring_mvs_da3": {
                "status": "closed_production_native_result",
                "authoritative_sfm_qualified": True,
                "production_native_comparison_valid": True,
                "rerun_authorized": False,
                "verdict_sha256": verdict_sha,
            }
        },
        "authority": {
            "camera_pose_authority_owner": "pycolmap_sfm",
            "provider_selected": False,
            "provider_promoted": False,
            "universal_provider_winner_assumed": False,
            "global_scalar_score_forbidden": True,
            "auto_policy_implemented": False,
            "hybrid_authorized": False,
            "metric_scale_claimed": False,
            "measurement_authoritative": False,
            "certified_metrology": False,
            "surface_mesh_generated": False,
            "permanent_global_view_cap": None,
        },
        "g2c_handoff": {
            "production_native_results_frozen": True,
            "provider_reexecution_required": False,
            "threshold_retuning_allowed": False,
            "source_rebinding_allowed": False,
            "matched_view_manual_review_required": True,
            "next_phase": "I4.3G2C",
        },
    }


def test_g2c_entry_accepts_only_formally_closed_frozen_g2b(tmp_path: Path):
    verdict_sha = "12" * 32
    path = _write(tmp_path / "closure.json", _closure(verdict_sha))
    result = verify_g2b_closure_for_g2c(
        path,
        expected_closure_sha256=_sha(path),
        expected_temple_ring_verdict_sha256=verdict_sha,
    )
    assert result["i4_3g2b_closed"] is True
    assert result["i4_3g2c_ready"] is True


def test_g2c_entry_rejects_provider_reexecution_or_authority_drift(tmp_path: Path):
    verdict_sha = "34" * 32
    payload = _closure(verdict_sha)
    payload["g2c_handoff"]["provider_reexecution_required"] = True
    path = _write(tmp_path / "closure_bad.json", payload)
    with pytest.raises(ReconstructionValidationError):
        verify_g2b_closure_for_g2c(
            path,
            expected_closure_sha256=_sha(path),
            expected_temple_ring_verdict_sha256=verdict_sha,
        )


def test_g2c_matched_view_public_render_contract_is_equal_density_and_blind():
    draw_count = resolve_equal_render_point_count(
        mvs_source_count=159_372,
        da3_source_count=1_500_000,
        requested_point_budget=250_000,
    )
    assert draw_count == 159_372
    a = blinded_render_label(blind_label="A", view="front", framing="context")
    b = blinded_render_label(blind_label="B", view="front", framing="context")
    assert a.replace("CLOUD A", "CLOUD X") == b.replace("CLOUD B", "CLOUD X")
    assert "MVS" not in a and "DA3" not in a
    assert "MVS" not in b and "DA3" not in b


def test_g2c_runner_normalizes_each_native_capture_before_pairing():
    root = Path(__file__).resolve().parents[2]
    runner = (root / "scripts" / "run_i4_3g2c_temple_ring_matched_view.py").read_text(encoding="utf-8")
    assert "window.setFixedSize(G2C_RENDER_WIDTH, G2C_RENDER_HEIGHT)" in runner
    assert "captured = window.grab().toImage()" in runner
    assert "Qt.AspectRatioMode.IgnoreAspectRatio" in runner
    assert "G2C matched-view rendered image pair extent mismatch" in runner
    assert '"canonical_render_raster": canonical_render_policy()' in runner
