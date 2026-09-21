from __future__ import annotations

import pytest

from core.reconstruction.errors import ReconstructionValidationError
from core.reconstruction.i5_hybrid_admission import build_i5_12a_hybrid_admission_review, evaluate_g2_hybrid_evidence
from tests.i5_12a_helpers import build_args, clone, g2


def test_deterministic_decision_over_repeated_reviews():
    statuses = [build_i5_12a_hybrid_admission_review(**build_args())["quality_status"] for _ in range(20)]
    assert set(statuses) == {"hybrid_not_justified"}


@pytest.mark.parametrize("dimension", [
    "false_geometry_and_floaters",
    "surface_noise_and_roughness",
    "false_bridging_or_thickening",
])
def test_any_missing_mvs_cleanliness_signal_blocks_admission(dimension: str):
    payload = g2(justified=True)
    payload["scenario_synthesis"]["g2_weak_overlap_mvs_da3"]["provider_preferences"][dimension] = "tie"
    evidence = evaluate_g2_hybrid_evidence(payload)
    assert evidence["mvs_cleaner_observed"] is False
    assert evidence["historical_hybrid_condition_met"] is False


def test_direct_hole_signal_is_required_even_with_da3_completeness():
    evidence = evaluate_g2_hybrid_evidence(g2(justified=False, completeness_da3=True))
    assert evidence["global_completeness_da3_observed"] is True
    assert evidence["historical_hybrid_condition_met"] is False


def test_binding_drift_fails_closed():
    args = build_args(justified=True)
    args["i5_2a_admission"] = clone(args["i5_2a_admission"])
    args["i5_2a_admission"]["evidence_bindings"]["i4_3g2_final_closure"]["sha256"] = "f" * 64
    with pytest.raises(ReconstructionValidationError):
        build_i5_12a_hybrid_admission_review(**args)
