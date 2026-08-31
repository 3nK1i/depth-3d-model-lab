from __future__ import annotations

from copy import deepcopy

import pytest

from core.reconstruction.errors import ReconstructionValidationError
from core.reconstruction.i4_3_final_qualification import build_i4_3h1_final_qualification_audit
from tests.i4_3h1_helpers import passing_g5b, passing_h1_bindings, passing_regression_receipt


def _build(g5b=None):
    return build_i4_3h1_final_qualification_audit(
        i4_3g5b_closure=passing_g5b() if g5b is None else g5b,
        regression_receipt=passing_regression_receipt(),
        evidence_bindings=passing_h1_bindings(),
        g5b_predecessor_hashes_reverified=True,
    )


def test_h1_repeated_final_audit_is_deterministic() -> None:
    first = _build()
    for _ in range(30):
        assert _build() == first


@pytest.mark.parametrize("field", [
    "authoritative_sfm_modified", "provider_selected", "provider_promoted",
    "auto_policy_implemented", "hybrid_authorized", "metric_scale_claimed",
    "measurement_authoritative", "certified_metrology", "surface_mesh_generated",
])
def test_h1_repeatedly_fails_closed_on_g_authority_drift(field: str) -> None:
    for _ in range(3):
        g5b = passing_g5b(); g5b["authority"][field] = True
        with pytest.raises(ReconstructionValidationError):
            _build(g5b)


def test_h1_preserves_frozen_g5b_under_repeated_calls() -> None:
    g5b = passing_g5b(); before = deepcopy(g5b)
    for _ in range(20):
        _build(g5b)
    assert g5b == before
