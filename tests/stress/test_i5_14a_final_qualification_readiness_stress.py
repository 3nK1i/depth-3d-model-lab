from __future__ import annotations

from copy import deepcopy
import pytest

from core.reconstruction.errors import ReconstructionValidationError
from core.reconstruction.i5_final_qualification_readiness import build_i5_14a_final_qualification_readiness
from tests.i5_14a_helpers import make_binding, make_bindings, make_closures, make_receipt


def _build(closures):
    return build_i5_14a_final_qualification_readiness(
        closures=closures, closure_bindings=make_bindings(),
        post_i5_13d_checkpoint_binding=make_binding("post.tar.gz", "f"),
        readiness_receipt=make_receipt(), readiness_receipt_binding=make_binding("receipt.json", "e"),
    )


@pytest.mark.parametrize("key", list(make_closures()))
def test_every_missing_closure_is_fatal(key):
    closures = make_closures()
    closures.pop(key)
    with pytest.raises(ReconstructionValidationError):
        _build(closures)


@pytest.mark.parametrize("key", list(make_closures()))
def test_every_failed_closure_is_fatal(key):
    closures = make_closures()
    closures[key]["pass"] = False
    with pytest.raises(ReconstructionValidationError):
        _build(closures)


def test_repeat_build_is_deterministic():
    a = _build(make_closures())
    b = _build(make_closures())
    assert a == b
