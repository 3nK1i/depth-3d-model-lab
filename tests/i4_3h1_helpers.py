from __future__ import annotations

from copy import deepcopy

from core.reconstruction.cross_provider_qualification_closure import (
    G5BClosureBinding,
    build_i4_3g5b_final_closure,
)
from core.reconstruction.i4_3_final_qualification import H1EvidenceBinding
from tests.i4_3g5b_helpers import passing_g5a, passing_g5b_bindings


def passing_g5b() -> dict:
    return build_i4_3g5b_final_closure(
        i4_3g5a_synthesis=passing_g5a(),
        evidence_bindings=passing_g5b_bindings(),
        chain_rebuild_verified=True,
    )


def passing_regression_receipt(*, focused_tests: int = 24) -> dict:
    return {
        "schema_version": 1,
        "phase": "I4.3H1_REGRESSION_AUDIT",
        "status": "pass",
        "pass": True,
        "scope": {
            "focused_h1_contract_tests": True,
            "predecessor_i4_3_unit_tests": True,
            "predecessor_i4_3_integration_tests": True,
            "predecessor_i4_3_stress_tests": True,
            "import_probe": True,
            "runner_help_probe": True,
        },
        "counts": {
            "focused_h1_tests": focused_tests,
            "predecessor_test_file_count": 91,
            "predecessor_unit_tests": 444,
            "predecessor_integration_tests": 86,
            "predecessor_stress_tests": 138,
            "predecessor_total_tests": 668,
        },
        "return_codes": {
            "focused_h1": 0,
            "predecessor_unit": 0,
            "predecessor_integration": 0,
            "predecessor_stress": 0,
            "import_probe": 0,
            "runner_help_probe": 0,
        },
    }


def passing_h1_bindings() -> dict[str, H1EvidenceBinding]:
    return {
        "i4_3g5b_closure": H1EvidenceBinding(path="/evidence/g5b.json", sha256="a" * 64),
        "i4_3h1_regression_receipt": H1EvidenceBinding(path="/evidence/h1-regression.json", sha256="b" * 64),
    }


def clone(value: dict) -> dict:
    return deepcopy(value)
