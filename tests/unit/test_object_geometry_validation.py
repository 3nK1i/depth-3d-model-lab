from __future__ import annotations

import pytest

from core.contracts.validation import ContractValidationError
from core.object_geometry.pipeline import ObjectPointExtractionRequest
from tests.object_geometry_helpers import make_bundle


def test_request_selection_is_explicit_unique_and_known():
    bundle = make_bundle(masks=(bundle_mask := __import__('numpy').ones((36,48), bool), bundle_mask.copy()))
    request = ObjectPointExtractionRequest(bundle, ("object-1",))
    assert request.observation_ids == ("object-1",)
    with pytest.raises(ContractValidationError):
        ObjectPointExtractionRequest(bundle, ("missing",))
    with pytest.raises(ContractValidationError):
        ObjectPointExtractionRequest(bundle, ("object-0", "object-0"))
