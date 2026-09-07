from __future__ import annotations

import pytest

from core.reconstruction.errors import ReconstructionValidationError
from core.reconstruction.i5_mvs_robustness import validate_i5_6b_for_i5_6c
from tests.i5_6c_helpers import clone, passing_i5_6b


@pytest.mark.parametrize(
    "section,key,value",
    [
        ("scope", "sfm_rerun", True),
        ("scope", "provider_comparison_started", True),
        ("scope", "thresholds_modified", True),
        ("scope", "production_default_modified", True),
        ("decision", "mp_sfm_authorized", True),
        ("decision", "auto_policy_implemented", True),
        ("decision", "hybrid_authorized", True),
        ("authority", "metric_scale_claimed", True),
        ("authority", "provider_ranking_created", True),
    ],
)
def test_i56c_rejects_forbidden_preclosure_drift(section: str, key: str, value) -> None:
    payload = clone(passing_i5_6b())
    payload[section][key] = value
    with pytest.raises(ReconstructionValidationError):
        validate_i5_6b_for_i5_6c(payload)


def test_i56c_rejects_missing_scenario_record() -> None:
    payload = clone(passing_i5_6b())
    payload["scenario_records"].pop("castle_symmetry")
    with pytest.raises(ReconstructionValidationError):
        validate_i5_6b_for_i5_6c(payload)


def test_i56c_rejects_native_visibility_mismatch() -> None:
    payload = clone(passing_i5_6b())
    payload["scenario_records"]["castle_symmetry"]["native_metrics"]["visibility_point_count"] += 1
    with pytest.raises(ReconstructionValidationError):
        validate_i5_6b_for_i5_6c(payload)


def test_i56c_rejects_recovery_without_authority_revalidation() -> None:
    payload = clone(passing_i5_6b())
    payload["recovery_provenance"]["authoritative_inputs_revalidated"] = False
    with pytest.raises(ReconstructionValidationError):
        validate_i5_6b_for_i5_6c(payload)


def test_i56c_rejects_recovery_scenario_patchmatch_rerun() -> None:
    payload = clone(passing_i5_6b())
    payload["scenario_records"]["castle_dynamic_proxy"]["recovery_provenance"]["patch_match_rerun"] = True
    with pytest.raises(ReconstructionValidationError):
        validate_i5_6b_for_i5_6c(payload)


def test_i56c_validation_does_not_mutate_replay() -> None:
    payload = passing_i5_6b()
    before = clone(payload)
    validate_i5_6b_for_i5_6c(payload)
    assert payload == before
