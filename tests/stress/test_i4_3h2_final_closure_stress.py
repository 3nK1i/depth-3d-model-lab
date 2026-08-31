from __future__ import annotations

import pytest

from core.reconstruction.errors import ReconstructionValidationError
from core.reconstruction.i4_3_final_closure import H2EvidenceBinding, build_i4_3h2_final_closure
from tests.i4_3h2_helpers import clone, passing_h1


def build(h1: dict) -> dict:
    return build_i4_3h2_final_closure(
        i4_3h1_audit=h1,
        evidence_binding=H2EvidenceBinding(path="/evidence/h1.json", sha256="c" * 64),
        h1_bound_artifacts_reverified=True,
        g5b_predecessor_artifacts_reverified=True,
        current_architecture_matches_h1=True,
    )


def test_h2_repeated_closure_is_deterministic() -> None:
    first = build(passing_h1())
    for _ in range(100):
        assert build(passing_h1()) == first


@pytest.mark.parametrize(
    "path,value",
    [
        (("decision", "auto_policy_implemented"), True),
        (("decision", "hybrid_authorized"), True),
        (("authority", "metric_scale_claimed"), True),
        (("authority", "measurement_authoritative"), True),
        (("authority", "certified_metrology"), True),
        (("architecture_audit", "permanent_global_view_cap"), 40),
    ],
)
def test_h2_repeatedly_fails_closed_on_authority_or_policy_drift(path: tuple[str, str], value: object) -> None:
    for _ in range(20):
        h1 = clone(passing_h1())
        h1[path[0]][path[1]] = value
        with pytest.raises(ReconstructionValidationError):
            build(h1)


def test_h2_never_mutates_frozen_h1_under_repeated_calls() -> None:
    h1 = passing_h1()
    before = clone(h1)
    for _ in range(50):
        build(h1)
    assert h1 == before
