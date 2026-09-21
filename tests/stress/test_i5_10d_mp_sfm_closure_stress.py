import pytest
from core.reconstruction.errors import ReconstructionValidationError
from core.reconstruction.i5_mp_sfm_qualification import validate_i5_10c_for_i5_10d
from tests.i5_10d_helpers import clone, synthetic_i510c_rejected


@pytest.mark.parametrize("field,value", [
    ("baseline_rerun", True),
    ("retry_until_pass", True),
    ("threshold_relaxation", True),
    ("production_default_modified", True),
])
def test_i510d_rejects_scope_drift(field, value):
    bad = clone(synthetic_i510c_rejected())
    bad["scope"][field] = value
    with pytest.raises(ReconstructionValidationError):
        validate_i5_10c_for_i5_10d(bad)
