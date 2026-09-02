from __future__ import annotations

import pytest

from core.reconstruction.errors import ReconstructionValidationError
from core.reconstruction.i5_difficult_handheld_robustness import validate_i5_1b_for_i5_1c
from tests.i5_1c_helpers import clone, degraded_i5_1b


@pytest.mark.parametrize("location,field", [
    ("decision", "provider_selected"),
    ("decision", "provider_promoted"),
    ("decision", "provider_ranking_created"),
    ("decision", "auto_policy_implemented"),
    ("decision", "hybrid_authorized"),
    ("decision", "dense_consistency_threshold_relaxation_authorized"),
    ("decision", "metric_scale_claimed"),
    ("decision", "measurement_authoritative"),
    ("decision", "certified_metrology"),
    ("decision", "surface_mesh_generated"),
])
def test_i51c_fails_closed_on_forbidden_replay_drift(location: str, field: str) -> None:
    replay = clone(degraded_i5_1b())
    replay[location][field] = True
    with pytest.raises(ReconstructionValidationError):
        validate_i5_1b_for_i5_1c(replay)


def test_i51c_does_not_accept_i52_preauthorization_from_replay() -> None:
    replay = clone(degraded_i5_1b())
    replay["decision"]["i5_2_authorized"] = True
    with pytest.raises(ReconstructionValidationError):
        validate_i5_1b_for_i5_1c(replay)
