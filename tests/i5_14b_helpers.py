from __future__ import annotations

from copy import deepcopy
from typing import Any

from tests.i5_14a_helpers import make_bindings, make_closures, make_receipt as make_a_receipt
from core.reconstruction.i5_final_qualification_readiness import build_i5_14a_final_qualification_readiness


def make_binding(name: str, fill: str) -> dict[str, str]:
    return {"path": f"/evidence/{name}", "sha256": fill * 64}


def make_readiness() -> dict[str, Any]:
    return build_i5_14a_final_qualification_readiness(
        closures=make_closures(),
        closure_bindings=make_bindings(),
        post_i5_13d_checkpoint_binding=make_binding("post_i5_13d.tar.gz", "f"),
        readiness_receipt=make_a_receipt(),
        readiness_receipt_binding=make_binding("i5_14a_receipt.json", "e"),
    )


def make_closure_receipt() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "phase": "I5.14B_FINAL_I5_CLOSURE_RECEIPT",
        "status": "pass",
        "pass": True,
        "scope": {
            "focused_i5_14b_tests": True,
            "full_i5_closure_regression": True,
            "evidence_only_final_closure": True,
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


def mutated_readiness() -> dict[str, Any]:
    return deepcopy(make_readiness())
