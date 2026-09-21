from __future__ import annotations

from copy import deepcopy
import pytest

from core.reconstruction.errors import ReconstructionValidationError
from core.reconstruction.i5_final_closure import build_i5_14b_final_closure
from tests.i5_14b_helpers import make_binding, make_closure_receipt, make_readiness


def _build(readiness):
    return build_i5_14b_final_closure(
        readiness=readiness,
        readiness_binding=make_binding("i5_14a.json", "a"),
        post_i5_14a_checkpoint_binding=make_binding("post.tar.gz", "b"),
        closure_receipt=make_closure_receipt(),
        closure_receipt_binding=make_binding("receipt.json", "c"),
    )


@pytest.mark.parametrize("limitation", make_readiness()["bounded_limitations"])
def test_removing_any_bounded_limitation_is_fatal(limitation):
    readiness = make_readiness()
    readiness["bounded_limitations"].remove(limitation)
    with pytest.raises(ReconstructionValidationError):
        _build(readiness)


def test_repeat_final_closure_is_deterministic():
    a = _build(make_readiness())
    b = _build(make_readiness())
    assert a == b


def test_mutating_forward_motion_result_is_fatal():
    readiness = make_readiness()
    readiness["evidence_synthesis"]["sparse_sfm"]["forward_motion"] = "pass"
    with pytest.raises(ReconstructionValidationError):
        _build(readiness)


def test_mutating_hybrid_result_is_fatal():
    readiness = make_readiness()
    readiness["evidence_synthesis"]["hybrid"]["quality_status"] = "pass"
    with pytest.raises(ReconstructionValidationError):
        _build(readiness)


def test_mutating_production_auto_result_is_fatal():
    readiness = make_readiness()
    readiness["evidence_synthesis"]["auto_policy"]["production_auto_authorized"] = True
    with pytest.raises(ReconstructionValidationError):
        _build(readiness)
