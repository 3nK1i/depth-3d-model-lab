from __future__ import annotations

from copy import deepcopy
from pathlib import Path

from core.reconstruction.i5_difficult_handheld_robustness import I51EvidenceBinding
from core.reconstruction.i5_low_overlap_weak_parallax import (
    analyze_i5_2b_sfm_report,
    build_i5_2b_low_overlap_weak_parallax_replay,
)
from tests.i5_2b_helpers import (
    I52A_SHA,
    RECEIPT_52B_SHA,
    passing_i5_2a,
    passing_i5_2b_receipt,
    synthetic_sfm_report,
)

I52B_SHA = "9" * 64
RECEIPT_52C_SHA = "a" * 64


def clone(value):
    return deepcopy(value)


def _binding(path: str, sha: str) -> I51EvidenceBinding:
    return I51EvidenceBinding(path, sha)


def passing_i5_2b(tmp_path: Path, *, low_registered: int = 6) -> dict:
    admission = passing_i5_2a(tmp_path)
    control = analyze_i5_2b_sfm_report(
        synthetic_sfm_report(
            admission,
            "control",
            camera_step=1.0,
            common_landmarks_per_adjacent_pair=40,
            registered_count=19,
        ),
        scenario_plan=admission["stress_plan"]["control"],
    )
    low = analyze_i5_2b_sfm_report(
        synthetic_sfm_report(
            admission,
            "low_overlap_proxy",
            camera_step=2.0,
            common_landmarks_per_adjacent_pair=12,
            registered_count=low_registered,
        ),
        scenario_plan=admission["stress_plan"]["low_overlap_proxy"],
    )
    weak = analyze_i5_2b_sfm_report(
        synthetic_sfm_report(
            admission,
            "weak_parallax_proxy",
            camera_step=0.18,
            common_landmarks_per_adjacent_pair=35,
            registered_count=7,
        ),
        scenario_plan=admission["stress_plan"]["weak_parallax_proxy"],
    )
    return build_i5_2b_low_overlap_weak_parallax_replay(
        i5_2a_result=admission,
        i5_2a_binding=_binding("/evidence/i5_2a.json", I52A_SHA),
        replay_receipt=passing_i5_2b_receipt(),
        replay_receipt_binding=_binding("/evidence/i5_2b_receipt.json", RECEIPT_52B_SHA),
        control_analysis=control,
        low_overlap_analysis=low,
        weak_parallax_analysis=weak,
        source_reverified_before=True,
        source_reverified_after=True,
    )


def passing_i5_2c_receipt() -> dict:
    return {
        "schema_version": 1,
        "phase": "I5.2C_QUALIFICATION_RECEIPT",
        "status": "pass",
        "pass": True,
        "scope": {
            "focused_i5_2c_tests": True,
            "predecessor_i5_2_regression": True,
            "real_replay_contract_regression": True,
            "stress_geometry_regression": True,
            "import_probe": True,
            "runner_help_probe": True,
            "sfm_rerun": False,
            "dense_provider_rerun": False,
            "quality_replay_rerun": False,
            "thresholds_modified": False,
            "production_default_modified": False,
            "mp_sfm_started": False,
            "auto_policy_implemented": False,
            "hybrid_authorized": False,
        },
        "counts": {
            "focused_tests": 25,
            "predecessor_unit_tests": 45,
            "predecessor_integration_tests": 8,
            "predecessor_stress_tests": 12,
            "predecessor_total_tests": 65,
            "combined_exercised_tests": 90,
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
