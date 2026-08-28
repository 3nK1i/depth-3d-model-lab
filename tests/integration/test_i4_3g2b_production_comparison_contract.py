from __future__ import annotations

from pathlib import Path

from core.reconstruction.mvs_da3_production_comparison import build_g2b_runtime_preflight_snapshot
from tests.i4_3g2b_helpers import da3_runtime, g1_snapshot, g2a_snapshot, make_source, mvs_runtime


def test_preflight_preserves_g1_and_g2a_authority_contract(tmp_path: Path) -> None:
    source = make_source(tmp_path / "images", count=8)
    snapshot = build_g2b_runtime_preflight_snapshot(
        i4_3g1_snapshot=g1_snapshot(),
        i4_3g2a_snapshot=g2a_snapshot(source),
        scenario_id="g2_low_texture_mvs_da3",
        mvs_runtime=mvs_runtime(),
        da3_runtime=da3_runtime(),
        verify_files=True,
    )
    assert snapshot["phase"] == "I4.3G2B"
    assert snapshot["status"] == "ready"
    assert snapshot["authority"]["camera_pose_authority_owner"] == "pycolmap_sfm"
    assert snapshot["authority"]["permanent_global_view_cap"] is None
    assert snapshot["authority"]["provider_selected"] is False
    assert snapshot["authority"]["auto_policy_implemented"] is False
    assert snapshot["authority"]["hybrid_authorized"] is False
    assert snapshot["decision"]["global_scalar_score_forbidden"] is True
    assert snapshot["decision"]["universal_provider_winner_assumed"] is False
