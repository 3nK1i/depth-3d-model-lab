from __future__ import annotations

from copy import deepcopy

import pytest

from core.reconstruction.errors import ReconstructionValidationError
from core.reconstruction.offline_dense_provenance_authority_audit import (
    ProvenanceAuthorityEvidenceBinding,
    audit_current_offline_dense_authority_contracts,
    build_i4_4c_provenance_authority_audit,
)
from tests.i4_4a_helpers import passing_h2
from tests.i4_4b_helpers import passing_i4_4a
from tests.i4_4c_helpers import passing_i4_4b, passing_receipt


def _build(h2=None, i44a=None, i44b=None, receipt=None, **gates):
    defaults = {
        "h2_h1_chain_reverified": True,
        "h1_bound_artifacts_reverified": True,
        "g5b_predecessor_artifacts_reverified": True,
        "i4_4a_chain_reverified": True,
        "i4_4b_chain_reverified": True,
        "current_repository_authority_matches_frozen_h1": True,
    }
    defaults.update(gates)
    return build_i4_4c_provenance_authority_audit(
        i4_3h2_closure=passing_h2() if h2 is None else h2,
        i4_3h2_binding=ProvenanceAuthorityEvidenceBinding("/evidence/h2.json", "1" * 64),
        i4_4a_result=passing_i4_4a() if i44a is None else i44a,
        i4_4a_binding=ProvenanceAuthorityEvidenceBinding("/evidence/i44a.json", "2" * 64),
        i4_4b_result=passing_i4_4b() if i44b is None else i44b,
        i4_4b_binding=ProvenanceAuthorityEvidenceBinding("/evidence/i44b.json", "3" * 64),
        audit_receipt=passing_receipt() if receipt is None else receipt,
        audit_receipt_binding=ProvenanceAuthorityEvidenceBinding("/evidence/receipt.json", "4" * 64),
        **defaults,
    )


def test_i44c_current_authority_audit_is_deterministic() -> None:
    first = audit_current_offline_dense_authority_contracts()
    for _ in range(50):
        assert audit_current_offline_dense_authority_contracts() == first


def test_i44c_repeated_build_is_deterministic() -> None:
    first = _build()
    for _ in range(50):
        assert _build() == first


@pytest.mark.parametrize("field", ["provider_selected", "provider_promoted", "auto_policy_implemented", "hybrid_authorized"])
def test_i44c_repeatedly_fails_closed_on_i44b_policy_promotion(field: str) -> None:
    for _ in range(10):
        i44b = passing_i4_4b(); i44b["decision"][field] = True
        with pytest.raises(ReconstructionValidationError):
            _build(i44b=i44b)


@pytest.mark.parametrize("gate", [
    "h2_h1_chain_reverified",
    "h1_bound_artifacts_reverified",
    "g5b_predecessor_artifacts_reverified",
    "i4_4a_chain_reverified",
    "i4_4b_chain_reverified",
    "current_repository_authority_matches_frozen_h1",
])
def test_i44c_repeatedly_fails_closed_on_missing_reverification_gate(gate: str) -> None:
    for _ in range(10):
        with pytest.raises(ReconstructionValidationError):
            _build(**{gate: False})


def test_i44c_never_mutates_frozen_inputs_under_repeated_calls() -> None:
    h2 = passing_h2(); i44a = passing_i4_4a(); i44b = passing_i4_4b(); receipt = passing_receipt()
    before = [deepcopy(item) for item in (h2, i44a, i44b, receipt)]
    for _ in range(25):
        _build(h2=h2, i44a=i44a, i44b=i44b, receipt=receipt)
    assert [h2, i44a, i44b, receipt] == before
