from __future__ import annotations

from typing import Any

from tests.i5_13c_helpers import (
    make_binding,
    make_i5_13a_admission,
    make_i5_13b_freeze,
    make_i5_13c_receipt,
)
from core.reconstruction.i5_auto_policy_replay import build_i5_13c_replay_qualification


def make_i5_13c_qualification() -> dict[str, Any]:
    return build_i5_13c_replay_qualification(
        i5_13a_admission=make_i5_13a_admission(),
        i5_13a_binding=make_binding("i5_13a.json", "1"),
        i5_13b_freeze=make_i5_13b_freeze(),
        i5_13b_binding=make_binding("i5_13b.json", "2"),
        post_i5_13b_checkpoint_binding=make_binding("post_i5_13b.tar.gz", "3"),
        replay_module_binding=make_binding("i5_auto_policy_replay.py", "4"),
        engine_module_binding=make_binding("i5_auto_policy_engine.py", "3"),
        replay_receipt=make_i5_13c_receipt(),
        replay_receipt_binding=make_binding("i5_13c_receipt.json", "5"),
    )


def make_integration_inventory() -> dict[str, Any]:
    return {
        "auto_selection_mode_contract_exists": True,
        "offline_engine_requires_explicit_frozen_context_tokens": True,
        "production_scene_classifier_implemented": False,
        "live_context_to_evidence_token_mapping_qualified": False,
        "adaptive_production_planning_mode": "explicit_provider_only",
        "adaptive_planning_provider_selected_by_auto": False,
        "production_ui_provider_selection_mode": "inspection_only",
        "production_ui_provider_selection_activates_provider": False,
        "production_auto_engine_referenced_by_app": False,
        "production_auto_orchestration_entrypoint_implemented": False,
        "production_default_modified": False,
    }


def make_i5_13d_receipt() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "phase": "I5.13D_INTEGRATION_DECISION_RECEIPT",
        "status": "pass",
        "pass": True,
        "scope": {
            "focused_i5_13d_tests": True,
            "i5_13c_regression": True,
            "production_integration_surface_regression": True,
            "evidence_only_decision": True,
            "import_probe": True,
            "runner_help_probe": True,
            "provider_execution": False,
            "sfm_execution": False,
            "mvs_execution": False,
            "scene_classifier_execution": False,
            "production_auto_integration": False,
            "production_default_modified": False,
            "provider_promoted": False,
        },
        "counts": {
            "focused_tests": 1,
            "predecessor_unit_tests": 1,
            "predecessor_integration_tests": 1,
            "predecessor_stress_tests": 1,
            "predecessor_total_tests": 3,
            "combined_exercised_tests": 4,
        },
        "return_codes": {
            "focused": 0,
            "predecessor_unit": 0,
            "predecessor_integration": 0,
            "predecessor_stress": 0,
            "import_probe": 0,
            "runner_help_probe": 0,
        },
    }
