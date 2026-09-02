from __future__ import annotations

from copy import deepcopy

import pytest

from core.reconstruction.errors import ReconstructionValidationError
from core.reconstruction.offline_dense_cross_provider_visual_verification import (
    VisualVerificationEvidenceBinding,
    build_i4_4b_cross_provider_visual_verification,
    validate_g2_visual_closure,
    validate_g3_visual_closure,
    validate_g4_visual_closure,
)
from tests.i4_4b_helpers import passing_g2, passing_g3, passing_g4, passing_h2, passing_i4_4a, passing_receipt


def _build(h2=None, i44a=None, g2=None, g3=None, g4=None, receipt=None):
    return build_i4_4b_cross_provider_visual_verification(
        i4_3h2_closure=passing_h2() if h2 is None else h2,
        i4_3h2_binding=VisualVerificationEvidenceBinding("/evidence/h2.json", "1" * 64),
        i4_4a_result=passing_i4_4a() if i44a is None else i44a,
        i4_4a_binding=VisualVerificationEvidenceBinding("/evidence/i44a.json", "2" * 64),
        i4_3g2_closure=passing_g2() if g2 is None else g2,
        i4_3g2_binding=VisualVerificationEvidenceBinding("/evidence/g2.json", "3" * 64),
        i4_3g3c_closure=passing_g3() if g3 is None else g3,
        i4_3g3c_binding=VisualVerificationEvidenceBinding("/evidence/g3.json", "4" * 64),
        i4_3g4c_closure=passing_g4() if g4 is None else g4,
        i4_3g4c_binding=VisualVerificationEvidenceBinding("/evidence/g4.json", "5" * 64),
        visual_receipt=passing_receipt() if receipt is None else receipt,
        visual_receipt_binding=VisualVerificationEvidenceBinding("/evidence/receipt.json", "6" * 64),
    )


def test_i44b_repeated_validation_is_deterministic() -> None:
    first = (
        validate_g2_visual_closure(passing_g2()),
        validate_g3_visual_closure(passing_g3()),
        validate_g4_visual_closure(passing_g4()),
    )
    for _ in range(25):
        assert (
            validate_g2_visual_closure(passing_g2()),
            validate_g3_visual_closure(passing_g3()),
            validate_g4_visual_closure(passing_g4()),
        ) == first


def test_i44b_repeated_build_is_deterministic() -> None:
    first = _build()
    for _ in range(25):
        assert _build() == first


@pytest.mark.parametrize("path", [
    ("g3_temporal_motion", "rolling_depth"),
    ("g3_temporal_motion", "video_depth_anything"),
    ("g4_single_image_low_texture", "depth_anything_v2"),
    ("g4_single_image_low_texture", "marigold"),
])
def test_i44b_repeatedly_fails_closed_on_terminal_event_drift(path: tuple[str, str]) -> None:
    scenario_id, provider = path
    for _ in range(10):
        g3 = passing_g3(); g4 = passing_g4()
        if scenario_id.startswith("g3_"):
            g3["scenario_synthesis"][scenario_id]["production_statuses"][provider] = "pass"
        else:
            g4["scenario_synthesis"][scenario_id]["production_statuses"][provider] = "pass"
        with pytest.raises(ReconstructionValidationError):
            _build(g3=g3, g4=g4)


def test_i44b_never_mutates_inputs_under_repeated_calls() -> None:
    h2 = passing_h2(); i44a = passing_i4_4a(); g2 = passing_g2(); g3 = passing_g3(); g4 = passing_g4(); receipt = passing_receipt()
    before = [deepcopy(item) for item in (h2, i44a, g2, g3, g4, receipt)]
    for _ in range(25):
        _build(h2=h2, i44a=i44a, g2=g2, g3=g3, g4=g4, receipt=receipt)
    assert [h2, i44a, g2, g3, g4, receipt] == before
