from __future__ import annotations

from copy import deepcopy

from core.reconstruction.i4_3_final_closure import H2EvidenceBinding, build_i4_3h2_final_closure
from tests.i4_3h2_helpers import passing_h1


def passing_h2() -> dict:
    return build_i4_3h2_final_closure(
        i4_3h1_audit=passing_h1(),
        evidence_binding=H2EvidenceBinding(path="/evidence/h1.json", sha256="c" * 64),
        h1_bound_artifacts_reverified=True,
        g5b_predecessor_artifacts_reverified=True,
        current_architecture_matches_h1=True,
    )


def passing_receipt() -> dict:
    return {
        "schema_version": 1,
        "phase": "I4.4A_REGRESSION_RECEIPT",
        "status": "pass",
        "pass": True,
        "scope": {
            "focused_i4_4a_tests": True,
            "affected_unit_regression": True,
            "affected_integration_regression": True,
            "affected_stress_regression": True,
            "import_probe": True,
            "runner_help_probe": True,
            "provider_execution_started": False,
            "sfm_rerun": False,
            "visual_verification_started": False,
        },
        "counts": {
            "focused_tests": 27,
            "affected_test_files": 54,
            "affected_unit_tests": 355,
            "affected_integration_tests": 45,
            "affected_stress_tests": 79,
            "affected_total_tests": 479,
        },
        "return_codes": {
            "focused": 0,
            "affected_unit": 0,
            "affected_integration": 0,
            "affected_stress": 0,
            "import_probe": 0,
            "runner_help_probe": 0,
        },
    }


def clone(value: dict) -> dict:
    return deepcopy(value)
