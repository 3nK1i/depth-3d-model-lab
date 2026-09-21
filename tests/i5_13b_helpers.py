from __future__ import annotations

from copy import deepcopy

from core.reconstruction.dense_provider_architecture import (
    DenseProviderFamily,
    DenseProviderSelectionMode,
    DenseProviderSelectionRequest,
)
from core.reconstruction.i5_auto_policy_admission import build_i5_13a_auto_policy_admission
from tests.i5_13a_helpers import build_args as build_i513a_args


def clone(value):
    return deepcopy(value)


def admission() -> dict:
    out = build_i5_13a_auto_policy_admission(**build_i513a_args())
    anchors = out["evidence_synthesis"]["provider_operating_envelope"]["single_image_depth"]["historical_g4_anchors"]
    providers = ("metric3d_v2", "unidepth_v2", "moge_2", "depth_pro", "depth_anything_v2", "marigold")
    anchors["complete_room"] = {name: "pass" for name in providers}
    anchors["weak_overlap"] = {name: "pass" for name in providers}
    anchors["low_texture"] = {name: ("rejected" if name in {"depth_anything_v2", "marigold"} else "pass") for name in providers}
    return out


def request(*, override: str = "", allow_fallback: bool = False, sfm: bool = True) -> DenseProviderSelectionRequest:
    return DenseProviderSelectionRequest(
        mode=DenseProviderSelectionMode.AUTO,
        available_view_count=19,
        authoritative_sfm_available=sfm,
        authoritative_sfm_frame_id="sfm-frame-001" if sfm else "",
        allow_fallback=allow_fallback,
        engineer_override_provider_id=override,
    )


def context(
    *,
    sparse_context_id: str = "low_overlap_weak_parallax",
    evidence_context_id: str = "i5_6_current_mvs_envelope",
    family: DenseProviderFamily = DenseProviderFamily.MVS,
    available_provider_ids: tuple[str, ...] = ("colmap-mvs-external",),
    sfm_usable: bool = True,
    runtime_clean: bool = True,
):
    from core.reconstruction.i5_auto_policy_engine import I513BOfflineAutoContext

    return I513BOfflineAutoContext(
        sparse_context_id=sparse_context_id,
        evidence_context_id=evidence_context_id,
        provider_family=family,
        authoritative_sfm_production_usable=sfm_usable,
        runtime_integrity_clean=runtime_clean,
        available_provider_ids=available_provider_ids,
    )


def binding(path: str, char: str) -> dict[str, str]:
    return {"path": path, "sha256": char * 64}


def receipt() -> dict:
    return {
        "schema_version": 1,
        "phase": "I5.13B_ENGINE_RECEIPT",
        "status": "pass",
        "pass": True,
        "scope": {
            "focused_i5_13b_tests": True,
            "i5_13a_regression": True,
            "dense_selection_contract_regression": True,
            "determinism_stress": True,
            "import_probe": True,
            "runner_help_probe": True,
            "provider_execution": False,
            "sfm_execution": False,
            "mvs_execution": False,
            "production_auto_integration": False,
            "production_default_modified": False,
            "provider_promoted": False,
            "scene_classifier_implemented": False,
        },
        "counts": {"focused_tests": 34, "predecessor_total_tests": 40, "combined_exercised_tests": 74},
        "return_codes": {
            "focused": 0,
            "predecessor_unit": 0,
            "predecessor_integration": 0,
            "predecessor_stress": 0,
            "import_probe": 0,
            "runner_help_probe": 0,
        },
    }


def freeze_args() -> dict:
    return {
        "i5_13a_admission": admission(),
        "i5_13a_binding": binding("/e/i5_13a.json", "a"),
        "post_i5_13a_checkpoint_binding": binding("/e/post_i5_13a.tar.gz", "b"),
        "engine_module_binding": binding("/e/i5_auto_policy_engine.py", "c"),
        "engine_receipt": receipt(),
        "engine_receipt_binding": binding("/e/i5_13b_receipt.json", "d"),
    }
