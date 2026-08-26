from __future__ import annotations

import pytest

from core.reconstruction.errors import ReconstructionValidationError
from scripts.run_d3_1_five_view_chain_diagnostics import (
    _overall_chain_status,
    _validate_d3_summary,
)


def _summary() -> dict:
    return {
        "phase": "D3",
        "pass": True,
        "view_count": 5,
        "raw_accumulation": {
            "spatial_merge_applied": False,
            "voxel_fusion_applied": False,
            "multi_view_filtering_applied": False,
        },
        "views": [{"view_index": index} for index in range(5)],
    }


def test_validate_d3_summary_accepts_unfused_five_view_result() -> None:
    views = _validate_d3_summary(_summary())
    assert [item["view_index"] for item in views] == [0, 1, 2, 3, 4]


@pytest.mark.parametrize("field", ["spatial_merge_applied", "voxel_fusion_applied", "multi_view_filtering_applied"])
def test_validate_d3_summary_rejects_mutated_geometry(field: str) -> None:
    payload = _summary()
    payload["raw_accumulation"][field] = True
    with pytest.raises(ReconstructionValidationError):
        _validate_d3_summary(payload)


def test_validate_d3_summary_rejects_wrong_view_count() -> None:
    payload = _summary()
    payload["view_count"] = 4
    with pytest.raises(ReconstructionValidationError):
        _validate_d3_summary(payload)


def test_overall_chain_status_pass_requires_all_four_pairs() -> None:
    assert _overall_chain_status(["pass", "pass", "pass", "pass"]) == ("pass", True, 0)


def test_overall_chain_status_degraded_dominates() -> None:
    assert _overall_chain_status(["pass", "degraded", "pass", "inconclusive"]) == ("degraded", False, 2)


def test_overall_chain_status_inconclusive_when_no_degraded_pair() -> None:
    assert _overall_chain_status(["pass", "inconclusive", "pass", "pass"]) == ("inconclusive", False, 3)


def test_overall_chain_status_rejects_empty_statuses() -> None:
    with pytest.raises(ReconstructionValidationError):
        _overall_chain_status([])
