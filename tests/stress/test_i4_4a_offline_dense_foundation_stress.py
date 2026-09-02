from __future__ import annotations

from copy import deepcopy

import pytest

from core.reconstruction.errors import ReconstructionValidationError
from core.reconstruction.offline_dense_foundation_regression import (
    RegressionEvidenceBinding,
    audit_offline_dense_foundation_contracts,
    build_i4_4a_focused_regression,
)
from tests.i4_4a_helpers import passing_h2, passing_receipt


def _build(h2=None, receipt=None):
    return build_i4_4a_focused_regression(
        i4_3h2_closure=passing_h2() if h2 is None else h2,
        i4_3h2_binding=RegressionEvidenceBinding("/evidence/h2.json", "a" * 64),
        regression_receipt=passing_receipt() if receipt is None else receipt,
        regression_binding=RegressionEvidenceBinding("/evidence/receipt.json", "b" * 64),
    )


def test_i44a_repeated_contract_audit_is_deterministic() -> None:
    first = audit_offline_dense_foundation_contracts()
    for _ in range(50):
        assert audit_offline_dense_foundation_contracts() == first


def test_i44a_repeated_build_is_deterministic() -> None:
    first = _build()
    for _ in range(50):
        assert _build() == first


@pytest.mark.parametrize("field", ["provider_selected", "provider_promoted", "auto_policy_implemented", "hybrid_authorized"])
def test_i44a_repeatedly_fails_closed_on_policy_promotion(field: str) -> None:
    for _ in range(10):
        h2 = passing_h2(); h2["decision"][field] = True
        with pytest.raises(ReconstructionValidationError):
            _build(h2=h2)


def test_i44a_never_mutates_h2_or_receipt_under_repeated_calls() -> None:
    h2, receipt = passing_h2(), passing_receipt(); before_h2, before_receipt = deepcopy(h2), deepcopy(receipt)
    for _ in range(25):
        _build(h2=h2, receipt=receipt)
    assert h2 == before_h2
    assert receipt == before_receipt
