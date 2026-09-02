from __future__ import annotations

from copy import deepcopy

import pytest

from core.reconstruction.errors import ReconstructionValidationError
from core.reconstruction.offline_dense_final_i4_closure import (
    FinalI4EvidenceBinding,
    build_i4_4d_final_i4_closure,
)
from tests.i4_4c_helpers import build_passing_i4_4c
from tests.i4_4d_helpers import passing_receipt


def _build(c=None, receipt=None, *, bound=True, authority=True):
    return build_i4_4d_final_i4_closure(
        i4_4c_result=build_passing_i4_4c() if c is None else c,
        i4_4c_binding=FinalI4EvidenceBinding("/evidence/c.json", "1" * 64),
        closure_receipt=passing_receipt() if receipt is None else receipt,
        closure_receipt_binding=FinalI4EvidenceBinding("/evidence/d_receipt.json", "2" * 64),
        i4_4c_bound_artifacts_reverified=bound,
        current_repository_authority_matches_i4_4c=authority,
    )


def test_i44d_repeated_build_is_deterministic() -> None:
    first = _build()
    for _ in range(40):
        assert _build() == first


@pytest.mark.parametrize("field", ["provider_selected", "provider_promoted", "auto_policy_implemented", "hybrid_authorized"])
def test_i44d_repeatedly_fails_closed_on_i44c_policy_promotion(field: str) -> None:
    for _ in range(10):
        c = build_passing_i4_4c(); c["decision"][field] = True
        with pytest.raises(ReconstructionValidationError):
            _build(c=c)


@pytest.mark.parametrize("field", [
    "h2_h1_chain_reverified",
    "h1_bound_artifacts_reverified",
    "g5b_predecessor_artifacts_reverified",
    "i4_4a_chain_reverified",
    "i4_4b_chain_reverified",
    "current_repository_authority_matches_frozen_h1",
])
def test_i44d_repeatedly_fails_closed_on_i44c_provenance_drift(field: str) -> None:
    for _ in range(10):
        c = build_passing_i4_4c(); c["provenance_chain_audit"][field] = False
        with pytest.raises(ReconstructionValidationError):
            _build(c=c)


def test_i44d_repeatedly_requires_bound_artifact_reverification() -> None:
    for _ in range(25):
        with pytest.raises(ReconstructionValidationError):
            _build(bound=False)


def test_i44d_repeatedly_requires_current_authority_match() -> None:
    for _ in range(25):
        with pytest.raises(ReconstructionValidationError):
            _build(authority=False)


def test_i44d_never_mutates_frozen_inputs_under_repeated_calls() -> None:
    c, receipt = build_passing_i4_4c(), passing_receipt()
    before_c, before_receipt = deepcopy(c), deepcopy(receipt)
    for _ in range(25):
        _build(c=c, receipt=receipt)
    assert c == before_c
    assert receipt == before_receipt
