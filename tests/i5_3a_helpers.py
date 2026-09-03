from __future__ import annotations

from copy import deepcopy
from pathlib import Path

from core.reconstruction.i5_difficult_handheld_robustness import I51EvidenceBinding
from core.reconstruction.i5_low_overlap_weak_parallax import (
    build_i5_2c_low_overlap_weak_parallax_closure,
    build_ordered_source_identity,
)
from tests.i5_2b_helpers import I52A_SHA, passing_i5_2a
from tests.i5_2c_helpers import I52B_SHA, RECEIPT_52C_SHA, passing_i5_2b, passing_i5_2c_receipt

I52C_SHA = "b" * 64
RECEIPT_53A_SHA = "c" * 64


def clone(value):
    return deepcopy(value)


def _binding(path: str, sha: str) -> I51EvidenceBinding:
    return I51EvidenceBinding(path, sha)


def source_dir(tmp_path: Path) -> Path:
    # passing_i5_2a uses this exact path/name and produces the exact source identity
    # that is carried through I5.2C.
    passing_i5_2a(tmp_path)
    return tmp_path / "castle-P19"


def passing_i5_2c(tmp_path: Path) -> dict:
    admission = passing_i5_2a(tmp_path)
    replay = passing_i5_2b(tmp_path)
    return build_i5_2c_low_overlap_weak_parallax_closure(
        i5_2a_result=admission,
        i5_2a_binding=_binding("/evidence/i5_2a.json", I52A_SHA),
        i5_2b_result=replay,
        i5_2b_binding=_binding("/evidence/i5_2b.json", I52B_SHA),
        qualification_receipt=passing_i5_2c_receipt(),
        qualification_receipt_binding=_binding("/evidence/i5_2c_receipt.json", RECEIPT_52C_SHA),
    )


def passing_i5_3a_receipt() -> dict:
    return {
        "schema_version": 1,
        "phase": "I5.3A_ADMISSION_RECEIPT",
        "status": "pass",
        "pass": True,
        "scope": {
            "focused_i5_3a_tests": True,
            "predecessor_i5_2c_regression": True,
            "source_identity_regression": True,
            "stress_design_regression": True,
            "import_probe": True,
            "runner_help_probe": True,
            "stress_images_materialized": False,
            "sfm_rerun": False,
            "dense_provider_rerun": False,
            "thresholds_modified": False,
            "production_default_modified": False,
            "mp_sfm_started": False,
            "auto_policy_implemented": False,
            "hybrid_authorized": False,
        },
        "counts": {
            "focused_tests": 18,
            "predecessor_unit_tests": 24,
            "predecessor_integration_tests": 5,
            "predecessor_stress_tests": 12,
            "predecessor_total_tests": 41,
            "combined_exercised_tests": 59,
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
