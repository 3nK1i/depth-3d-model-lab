from __future__ import annotations

from core.reconstruction.i5_difficult_handheld_robustness import build_i5_1b_difficult_handheld_replay
from tests.i5_1b_helpers import (
    BASELINE_SHA,
    dense_ab_summary,
    i1_summary,
    localization_summary,
    passing_i5_1a,
    source_identity,
)


def _result():
    return build_i5_1b_difficult_handheld_replay(
        i5_1a_result=passing_i5_1a(),
        i5_1a_sha256=BASELINE_SHA,
        source_identity=source_identity(),
        i1_summary=i1_summary(),
        localization_summary=localization_summary(),
        dense_ab_summary=dense_ab_summary(),
        stage_return_codes={"i1": 2, "localization": 0, "dense_ab": 0},
        replay_parameters={
            "profile": "efficient", "matching": "exhaustive",
            "max_keyframes": 40, "max_dense_views": 40,
            "dense_candidate_cap": 120, "video_target_fps": 2.0,
            "local_target_fps": 2.0, "process_res": 392,
        },
    )


def test_i51b_contract_carries_fresh_source_identity_and_predecessor_binding() -> None:
    result = _result()
    assert result["provenance"]["i5_1a_sha256_bound"] is True
    assert result["provenance"]["fresh_source_sha256_bound"] is True
    assert result["source_identity"]["sha256"]


def test_i51b_contract_keeps_quality_separate_from_evidence_validity() -> None:
    result = _result()
    assert result["pass"] is True
    assert result["quality"]["status"] in {"pass", "degraded", "rejected"}
    assert result["quality"]["quality_status_is_not_provider_ranking"] is True


def test_i51b_contract_preserves_authority_boundary() -> None:
    result = _result()
    assert result["authority"]["camera_pose_authority_owner"] == "pycolmap_sfm"
    assert result["authority"]["permanent_global_view_cap"] is None
    assert result["authority_audit"]["default_reconstruction_scale_status"] == "arbitrary"


def test_i51b_contract_only_authorizes_i51c() -> None:
    result = _result()
    assert result["decision"]["i5_1c_authorized"] is True
    assert result["decision"]["i5_1_complete"] is False
    assert result["decision"]["i5_2_authorized"] is False
