from __future__ import annotations

from pathlib import Path

import pytest

from core.reconstruction.errors import ReconstructionValidationError
from core.reconstruction.i5_dynamic_scene_handling import validate_i5_5b_for_i5_5c
from tests.i5_5c_helpers import clone, passing_i5_5b


@pytest.mark.parametrize(
    "section,key,value",
    [
        ("scope", "dynamic_masking_added", True),
        ("scope", "semantic_segmentation_added", True),
        ("scope", "motion_outlier_filter_added", True),
        ("scope", "production_default_modified", True),
        ("decision", "mp_sfm_authorized", True),
        ("decision", "auto_policy_implemented", True),
        ("decision", "hybrid_authorized", True),
        ("authority", "metric_scale_claimed", True),
    ],
)
def test_i55c_rejects_forbidden_preclosure_drift(tmp_path: Path, section: str, key: str, value) -> None:
    payload = clone(passing_i5_5b(tmp_path))
    payload[section][key] = value
    with pytest.raises(ReconstructionValidationError):
        validate_i5_5b_for_i5_5c(payload)


def test_i55c_rejects_cross_run_absolute_geometry_claim(tmp_path: Path) -> None:
    payload = clone(passing_i5_5b(tmp_path))
    payload["paired_comparison"]["cross_run_absolute_geometry_comparison_permitted"] = True
    with pytest.raises(ReconstructionValidationError):
        validate_i5_5b_for_i5_5c(payload)


def test_i55c_validation_does_not_mutate_replay(tmp_path: Path) -> None:
    payload = passing_i5_5b(tmp_path)
    before = clone(payload)
    validate_i5_5b_for_i5_5c(payload)
    assert payload == before
