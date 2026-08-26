from __future__ import annotations

import pytest

from core.reconstruction.errors import ReconstructionValidationError
from scripts.run_d4_2_weak_link_bridge_diagnostics import (
    _bridge_diagnosis,
    _derive_bridge_pairs,
    _thresholds_from_d4_1,
    _validate_d4_1_summary,
)


def _pair(index: int, *, status: str = "pass") -> dict:
    return {
        "pair": [index, index + 1],
        "pair_label": f"{index}-{index + 1}",
        "workspace_orders": [index + 6, index + 7],
        "keyframe_ids": [f"kf-{index}", f"kf-{index + 1}"],
        "status": status,
        "a_to_b": {"status": status},
        "b_to_a": {"status": status},
    }


def _summary(*, degraded_index: int = 1) -> dict:
    pairs = [_pair(index, status="degraded" if index == degraded_index else "pass") for index in range(9)]
    return {
        "phase": "D4.1",
        "pass": False,
        "status": "degraded",
        "view_count": 10,
        "adjacent_pairs": pairs,
        "thresholds": {
            "consistency_relative_tolerance": 0.12,
            "occlusion_relative_margin": 0.20,
            "minimum_comparable_samples": 1000,
            "minimum_consistent_ratio": 0.50,
            "maximum_median_relative_residual": 0.10,
            "maximum_p90_relative_residual": 0.25,
            "maximum_front_conflict_ratio": 0.25,
            "nn_support_scene_fraction": 0.05,
        },
    }


def test_validate_d4_1_summary_finds_single_interior_weak_pair() -> None:
    weak, pairs = _validate_d4_1_summary(_summary(degraded_index=1))
    assert weak == (1, 2)
    assert len(pairs) == 9


def test_validate_d4_1_summary_rejects_non_degraded_source() -> None:
    payload = _summary()
    payload["pass"] = True
    payload["status"] = "pass"
    with pytest.raises(ReconstructionValidationError):
        _validate_d4_1_summary(payload)


def test_validate_d4_1_summary_rejects_multiple_or_edge_weak_pairs() -> None:
    payload = _summary(degraded_index=1)
    payload["adjacent_pairs"][4]["status"] = "degraded"
    with pytest.raises(ReconstructionValidationError):
        _validate_d4_1_summary(payload)
    with pytest.raises(ReconstructionValidationError):
        _validate_d4_1_summary(_summary(degraded_index=0))
    with pytest.raises(ReconstructionValidationError):
        _validate_d4_1_summary(_summary(degraded_index=8))


def test_derive_bridge_pairs_for_7_8_weak_link() -> None:
    assert _derive_bridge_pairs((1, 2)) == {"left": (0, 2), "right": (1, 3)}


def test_derive_bridge_pairs_rejects_non_adjacent_or_boundary_pair() -> None:
    for weak in ((0, 1), (8, 9), (2, 4)):
        with pytest.raises(ReconstructionValidationError):
            _derive_bridge_pairs(weak)


def test_bridge_diagnosis_both_pass_is_pair_specific() -> None:
    result = _bridge_diagnosis(weak_pair=(1, 2), left_status="pass", right_status="pass")
    assert result["classification"] == "pair_specific_incompatibility"
    assert result["preferred_single_view_exclusion_index"] is None


def test_bridge_diagnosis_one_sided_evidence_suggests_minimal_exclusion() -> None:
    left_member = _bridge_diagnosis(weak_pair=(1, 2), left_status="pass", right_status="degraded")
    right_member = _bridge_diagnosis(weak_pair=(1, 2), left_status="degraded", right_status="pass")
    assert left_member["classification"] == "left_member_suspect"
    assert left_member["preferred_single_view_exclusion_index"] == 1
    assert right_member["classification"] == "right_member_suspect"
    assert right_member["preferred_single_view_exclusion_index"] == 2


def test_bridge_diagnosis_handles_inconclusive_and_both_degraded_without_overclaiming() -> None:
    inc = _bridge_diagnosis(weak_pair=(1, 2), left_status="inconclusive", right_status="pass")
    bad = _bridge_diagnosis(weak_pair=(1, 2), left_status="degraded", right_status="degraded")
    assert inc["classification"] == "inconclusive_bridge_evidence"
    assert inc["preferred_single_view_exclusion_index"] is None
    assert bad["classification"] == "span_instability"
    assert bad["preferred_single_view_exclusion_index"] is None


def test_bridge_diagnosis_rejects_unknown_status() -> None:
    with pytest.raises(ReconstructionValidationError):
        _bridge_diagnosis(weak_pair=(1, 2), left_status="pass", right_status="unknown")


def test_thresholds_are_reused_from_d4_1_record() -> None:
    thresholds = _thresholds_from_d4_1(_summary())
    assert thresholds["consistency_relative_tolerance"] == pytest.approx(0.12)
    assert thresholds["maximum_median_relative_residual"] == pytest.approx(0.10)
    assert thresholds["maximum_front_conflict_ratio"] == pytest.approx(0.25)
    assert thresholds["minimum_comparable_samples"] == 1000


def test_thresholds_reject_incomplete_d4_1_record() -> None:
    payload = _summary()
    del payload["thresholds"]["maximum_p90_relative_residual"]
    with pytest.raises(ReconstructionValidationError):
        _thresholds_from_d4_1(payload)
