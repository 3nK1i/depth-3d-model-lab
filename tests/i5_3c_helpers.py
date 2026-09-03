from __future__ import annotations

from copy import deepcopy
from pathlib import Path

from core.reconstruction.i5_blur_exposure_clutter_symmetry import (
    analyze_i5_3b_pre_sfm_ingestion_rejection,
    build_i5_3b_blur_exposure_clutter_symmetry_replay,
)
from core.reconstruction.i5_difficult_handheld_robustness import I51EvidenceBinding
from tests.i5_3b_helpers import (
    I53A_SHA,
    RECEIPT_53B_SHA,
    passing_i5_3a,
    passing_i5_3b_receipt,
    passing_image_domain_comparison,
    synthetic_i53_sfm_analysis,
)

I53B_SHA = "f" * 64
RECEIPT_53C_SHA = "1" * 64


def clone(value):
    return deepcopy(value)


def _binding(path: str, sha: str) -> I51EvidenceBinding:
    return I51EvidenceBinding(path, sha)


def degraded_i5_3b(tmp_path: Path) -> dict:
    admission = passing_i5_3a(tmp_path)
    comparison = passing_image_domain_comparison()
    comparison["clutter"]["stress_bottom_band_orb_feature_density_p50"] = comparison["clutter"]["control_bottom_band_orb_feature_density_p50"]
    comparison["clutter"]["stress_to_control_ratio"] = 1.0
    comparison["clutter"]["semantic_stress_revalidated"] = False
    comparison["all_semantic_stresses_revalidated"] = False

    def _executed(scenario_key: str, registered_count: int) -> dict:
        value = synthetic_i53_sfm_analysis(admission, scenario_key, registered_count=registered_count)
        value["registration_attempted"] = True
        value["sfm_executed"] = True
        value["pre_sfm_ingestion_rejected"] = False
        value["rejection_stage"] = None
        value["rejection_reason"] = None
        return value

    analyses = {
        "control": _executed("control", 19),
        "blur": analyze_i5_3b_pre_sfm_ingestion_rejection(
            scenario_plan=admission["stress_plan"]["blur_proxy"],
            staged_names=[f"{i:04d}.png" for i in range(19)],
            reason="Fewer than three valid keyframes survived ingestion. Use a sharper, better exposed, overlapping sequence.",
        ),
        "underexposure": analyze_i5_3b_pre_sfm_ingestion_rejection(
            scenario_plan=admission["stress_plan"]["underexposure_proxy"],
            staged_names=[f"{i:04d}.png" for i in range(19)],
            reason="Fewer than three valid keyframes survived ingestion. Use a sharper, better exposed, overlapping sequence.",
        ),
        "overexposure": _executed("overexposure_proxy", 19),
        "clutter": _executed("clutter_proxy", 19),
        "symmetry": _executed("symmetry_proxy", 12),
    }
    return build_i5_3b_blur_exposure_clutter_symmetry_replay(
        i5_3a_result=admission,
        i5_3a_binding=_binding("/evidence/i5_3a.json", I53A_SHA),
        replay_receipt=passing_i5_3b_receipt(),
        replay_receipt_binding=_binding("/evidence/i5_3b_receipt.json", RECEIPT_53B_SHA),
        image_domain_comparison=comparison,
        scenario_analyses=analyses,
        source_reverified_before=True,
        source_reverified_after=True,
        staged_inputs_reverified=True,
    )


def passing_i5_3c_receipt() -> dict:
    return {
        "schema_version": 1,
        "phase": "I5.3C_QUALIFICATION_RECEIPT",
        "status": "pass",
        "pass": True,
        "scope": {
            "focused_i5_3c_tests": True,
            "predecessor_i5_3_regression": True,
            "real_replay_contract_regression": True,
            "stress_conclusion_regression": True,
            "import_probe": True,
            "runner_help_probe": True,
            "sfm_rerun": False,
            "stress_images_rematerialized": False,
            "dense_provider_rerun": False,
            "quality_replay_rerun": False,
            "thresholds_modified": False,
            "production_default_modified": False,
            "mp_sfm_started": False,
            "auto_policy_implemented": False,
            "hybrid_authorized": False,
        },
        "counts": {
            "focused_tests": 14,
            "predecessor_unit_tests": 47,
            "predecessor_integration_tests": 10,
            "predecessor_stress_tests": 12,
            "predecessor_total_tests": 69,
            "combined_exercised_tests": 83,
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
