from __future__ import annotations

import pytest

from core.reconstruction.errors import ReconstructionValidationError
from scripts.run_d4_1_ten_view_chain_diagnostics import (
    _D41_ADJACENT_PAIRS,
    _overall_chain_status,
    _validate_d4_summary,
)


def _summary() -> dict:
    return {
        "phase": "D4",
        "pass": True,
        "view_count": 10,
        "raw_accumulation": {
            "spatial_merge_applied": False,
            "voxel_fusion_applied": False,
            "multi_view_filtering_applied": False,
        },
        "views": [{"view_index": index} for index in range(10)],
    }


def test_adjacent_pairs_cover_all_nine_ten_view_links() -> None:
    assert _D41_ADJACENT_PAIRS == tuple((index, index + 1) for index in range(9))


def test_validate_d4_summary_accepts_unfused_ten_view_result() -> None:
    views = _validate_d4_summary(_summary())
    assert [item["view_index"] for item in views] == list(range(10))


@pytest.mark.parametrize(
    "field",
    ["spatial_merge_applied", "voxel_fusion_applied", "multi_view_filtering_applied"],
)
def test_validate_d4_summary_rejects_mutated_geometry(field: str) -> None:
    payload = _summary()
    payload["raw_accumulation"][field] = True
    with pytest.raises(ReconstructionValidationError):
        _validate_d4_summary(payload)


def test_validate_d4_summary_rejects_wrong_phase_or_failed_source() -> None:
    payload = _summary()
    payload["phase"] = "D3"
    with pytest.raises(ReconstructionValidationError):
        _validate_d4_summary(payload)
    payload = _summary()
    payload["pass"] = False
    with pytest.raises(ReconstructionValidationError):
        _validate_d4_summary(payload)


def test_validate_d4_summary_rejects_wrong_view_count_or_order() -> None:
    payload = _summary()
    payload["view_count"] = 9
    with pytest.raises(ReconstructionValidationError):
        _validate_d4_summary(payload)
    payload = _summary()
    payload["views"][7]["view_index"] = 8
    with pytest.raises(ReconstructionValidationError):
        _validate_d4_summary(payload)


def test_overall_chain_status_pass_requires_all_nine_pairs() -> None:
    assert _overall_chain_status(["pass"] * 9) == ("pass", True, 0)


def test_overall_chain_status_degraded_dominates() -> None:
    statuses = ["pass"] * 9
    statuses[3] = "inconclusive"
    statuses[7] = "degraded"
    assert _overall_chain_status(statuses) == ("degraded", False, 2)


def test_overall_chain_status_inconclusive_when_no_degraded_pair() -> None:
    statuses = ["pass"] * 9
    statuses[4] = "inconclusive"
    assert _overall_chain_status(statuses) == ("inconclusive", False, 3)


def test_overall_chain_status_rejects_wrong_count_or_unknown_statuses() -> None:
    with pytest.raises(ReconstructionValidationError):
        _overall_chain_status([])
    with pytest.raises(ReconstructionValidationError):
        _overall_chain_status(["pass"] * 8)
    with pytest.raises(ReconstructionValidationError):
        _overall_chain_status(["pass"] * 8 + ["unknown"])
