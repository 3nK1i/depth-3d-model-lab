from __future__ import annotations

from copy import deepcopy

from core.reconstruction.offline_dense_final_i4_closure import (
    FinalI4EvidenceBinding,
    build_i4_4d_final_i4_closure,
)
from tests.i4_4c_helpers import build_passing_i4_4c


def passing_receipt() -> dict:
    return {
        "schema_version": 1,
        "phase": "I4.4D_FINAL_CLOSURE_RECEIPT",
        "status": "pass",
        "pass": True,
        "scope": {
            "focused_i4_4d_tests": True,
            "predecessor_i4_4_unit_regression": True,
            "predecessor_i4_4_integration_regression": True,
            "predecessor_i4_4_stress_regression": True,
            "import_probe": True,
            "runner_help_probe": True,
            "provider_execution_started": False,
            "provider_runtime_probe_started": False,
            "sfm_rerun": False,
            "matched_view_rerun": False,
            "manual_review_rerun": False,
            "new_provider_comparison_started": False,
            "i5_work_started": False,
        },
        "counts": {
            "focused_tests": 47,
            "predecessor_test_files": 9,
            "predecessor_unit_tests": 51,
            "predecessor_integration_tests": 15,
            "predecessor_stress_tests": 27,
            "predecessor_total_tests": 93,
            "combined_exercised_tests": 140,
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


def build_passing_i4_4d() -> dict:
    return build_i4_4d_final_i4_closure(
        i4_4c_result=build_passing_i4_4c(),
        i4_4c_binding=FinalI4EvidenceBinding("/evidence/i44c.json", "1" * 64),
        closure_receipt=passing_receipt(),
        closure_receipt_binding=FinalI4EvidenceBinding("/evidence/i44d_receipt.json", "2" * 64),
        i4_4c_bound_artifacts_reverified=True,
        current_repository_authority_matches_i4_4c=True,
    )


def clone(value: dict) -> dict:
    return deepcopy(value)
