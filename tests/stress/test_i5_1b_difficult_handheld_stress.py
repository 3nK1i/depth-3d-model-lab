from __future__ import annotations

import pytest

from core.reconstruction.errors import ReconstructionValidationError
from core.reconstruction.i5_difficult_handheld_robustness import build_i5_1b_difficult_handheld_replay
from tests.i5_1b_helpers import BASELINE_SHA, clone, passing_i5_1a, source_identity


@pytest.mark.parametrize("field", [
    "provider_selected", "provider_promoted", "provider_ranking_created",
    "universal_provider_winner_declared", "global_scalar_score_created",
    "auto_policy_implemented", "hybrid_authorized", "metric_scale_claimed",
    "measurement_authoritative", "certified_metrology", "surface_mesh_generated",
])
def test_i51b_predecessor_forbidden_decisions_fail_closed(field: str) -> None:
    baseline = clone(passing_i5_1a())
    baseline["decision"][field] = True
    with pytest.raises(ReconstructionValidationError):
        build_i5_1b_difficult_handheld_replay(
            i5_1a_result=baseline,
            i5_1a_sha256=BASELINE_SHA,
            source_identity=source_identity(),
            i1_summary={"phase": "I1", "status": "failed", "failed_stage": "sfm"},
            localization_summary=None,
            dense_ab_summary=None,
            stage_return_codes={"i1": 2, "localization": None, "dense_ab": None},
            replay_parameters={
                "profile": "efficient", "matching": "exhaustive", "max_keyframes": 40,
                "max_dense_views": 40, "dense_candidate_cap": 120,
                "video_target_fps": 2.0, "local_target_fps": 2.0, "process_res": 392,
            },
        )


def test_i51b_rejects_permanent_cap_drift() -> None:
    baseline = clone(passing_i5_1a())
    baseline["authority"]["permanent_global_view_cap"] = 40
    with pytest.raises(ReconstructionValidationError):
        build_i5_1b_difficult_handheld_replay(
            i5_1a_result=baseline,
            i5_1a_sha256=BASELINE_SHA,
            source_identity=source_identity(),
            i1_summary={"phase": "I1", "status": "failed", "failed_stage": "sfm"},
            localization_summary=None,
            dense_ab_summary=None,
            stage_return_codes={"i1": 2, "localization": None, "dense_ab": None},
            replay_parameters={
                "profile": "efficient", "matching": "exhaustive", "max_keyframes": 40,
                "max_dense_views": 40, "dense_candidate_cap": 120,
                "video_target_fps": 2.0, "local_target_fps": 2.0, "process_res": 392,
            },
        )
