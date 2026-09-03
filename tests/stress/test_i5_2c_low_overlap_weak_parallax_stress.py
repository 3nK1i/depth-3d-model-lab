from __future__ import annotations

from pathlib import Path

import pytest

from core.reconstruction.errors import ReconstructionValidationError
from core.reconstruction.i5_low_overlap_weak_parallax import validate_i5_2b_for_i5_2c
from tests.i5_2c_helpers import clone, passing_i5_2b


@pytest.mark.parametrize(
    "path,value",
    [
        (("decision", "i5_3_authorized"), True),
        (("decision", "threshold_relaxation_authorized"), True),
        (("scope", "mp_sfm_strategy_evaluation_started"), True),
        (("provenance", "retry_until_quality_pass"), True),
        (("authority", "metric_scale_claimed"), True),
    ],
)
def test_i52c_stress_rejects_forbidden_preclosure_drift(tmp_path: Path, path: tuple[str, str], value) -> None:
    payload = clone(passing_i5_2b(tmp_path))
    payload[path[0]][path[1]] = value
    with pytest.raises(ReconstructionValidationError):
        validate_i5_2b_for_i5_2c(payload)


def test_i52c_stress_rejects_false_full_registration_rewrite(tmp_path: Path) -> None:
    payload = clone(passing_i5_2b(tmp_path))
    payload["scenarios"]["low_overlap"]["registered_images"] = 7
    payload["scenarios"]["low_overlap"]["registration_ratio"] = 1.0
    with pytest.raises(ReconstructionValidationError):
        validate_i5_2b_for_i5_2c(payload)
