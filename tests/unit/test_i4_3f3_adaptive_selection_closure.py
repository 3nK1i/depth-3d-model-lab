from __future__ import annotations

import pytest

from core.reconstruction.adaptive_selection_closure import build_i4_3f3_closure_snapshot
from core.reconstruction.errors import ReconstructionValidationError
from core.reconstruction.keyframes import KeyframeSelectionPolicy


def _f1():
    return {
        "phase": "I4.3F1", "status": "pass", "pass": True,
        "decision": {"i4_3f2_production_wiring_authorized": True, "permanent_global_view_cap": None},
    }


def _f2():
    return {
        "phase": "I4.3F2", "status": "pass", "pass": True,
        "decision": {
            "i4_3f2_production_wiring_closed": True,
            "i4_3f3_production_validation_authorized": True,
            "permanent_global_view_cap": None,
        },
        "authority": {
            "camera_pose_authority_owner": "pycolmap_sfm",
            "authoritative_sfm_modified": False,
            "provider_selected": False,
            "provider_promoted": False,
            "auto_policy_implemented": False,
            "hybrid_authorized": False,
            "metric_scale_claimed": False,
            "measurement_authoritative": False,
            "certified_metrology": False,
            "surface_mesh_generated": False,
        },
    }


def test_core_keyframe_policy_has_no_implicit_default_cap() -> None:
    assert KeyframeSelectionPolicy().maximum_keyframes is None
    assert KeyframeSelectionPolicy(maximum_keyframes=17).maximum_keyframes == 17


def test_f3_closure_snapshot_authorizes_only_g_without_promotion() -> None:
    providers = [f"provider-{i}" for i in range(10)]
    snapshot = build_i4_3f3_closure_snapshot(
        i4_3f1_snapshot=_f1(),
        i4_3f2_snapshot=_f2(),
        source_contract_checks={"default_none": True},
        scenario_results=({"name": "synthetic", "pass": True},),
        validated_provider_implementations=providers,
    )
    assert snapshot["decision"]["i4_3f_closed"] is True
    assert snapshot["decision"]["i4_3g_cross_provider_qualification_authorized"] is True
    assert snapshot["authority"]["permanent_global_view_cap"] is None
    assert snapshot["authority"]["provider_selected"] is False
    assert snapshot["authority"]["auto_policy_implemented"] is False


@pytest.mark.parametrize("field", ["provider_selected", "auto_policy_implemented", "hybrid_authorized", "metric_scale_claimed"])
def test_f3_refuses_predecessor_authority_drift(field: str) -> None:
    f2 = _f2()
    f2["authority"][field] = True
    with pytest.raises(ReconstructionValidationError, match="authority drift"):
        build_i4_3f3_closure_snapshot(
            i4_3f1_snapshot=_f1(), i4_3f2_snapshot=f2,
            source_contract_checks={"ok": True},
            scenario_results=({"name": "x", "pass": True},),
            validated_provider_implementations=[f"p{i}" for i in range(10)],
        )


def test_f3_refuses_permanent_global_cap() -> None:
    f2 = _f2()
    f2["decision"]["permanent_global_view_cap"] = 80
    with pytest.raises(ReconstructionValidationError, match="permanent global view cap"):
        build_i4_3f3_closure_snapshot(
            i4_3f1_snapshot=_f1(), i4_3f2_snapshot=f2,
            source_contract_checks={"ok": True},
            scenario_results=({"name": "x", "pass": True},),
            validated_provider_implementations=[f"p{i}" for i in range(10)],
        )


def test_f3_refuses_incomplete_provider_coverage() -> None:
    with pytest.raises(ReconstructionValidationError, match="all 10"):
        build_i4_3f3_closure_snapshot(
            i4_3f1_snapshot=_f1(), i4_3f2_snapshot=_f2(),
            source_contract_checks={"ok": True},
            scenario_results=({"name": "x", "pass": True},),
            validated_provider_implementations=[f"p{i}" for i in range(9)],
        )
