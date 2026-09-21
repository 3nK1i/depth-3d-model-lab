from __future__ import annotations

from copy import deepcopy
from typing import Any

from tests.i5_10b_helpers import synthetic_i510a_admission
from tests.i5_10c_helpers import synthetic_preflight, synthetic_receipt, synthetic_report
from core.reconstruction.i5_mp_sfm_real_evaluation import (
    analyze_i5_10c_sfm_report,
    build_i5_10c_real_evaluation,
)

I510A_SHA = "fca08621136404bd5ecdb7b861123f28f4c624dd3b38be0c35e51989a4f4e2db"
I510B_SHA = "0db52edcf9805d66656f874256b5219ded26afe2b25666024c7b9bd3ed02dae4"
I510C_SHA = "f40c915856be93daef782b6d793af7cc3ffa7247280df32009be67c2c80cb98c"
I510D_RECEIPT_SHA = "d" * 64


def synthetic_i510b_preflight() -> dict[str, Any]:
    return synthetic_preflight()


def synthetic_i510c_rejected() -> dict[str, Any]:
    p = synthetic_i510b_preflight()
    report = synthetic_report(quality="rejected", registered=3)
    # Align the synthetic report with the observed C quality evidence.
    report["sfm_result"]["statistics"]["landmarks"] = 42
    report["sfm_result"]["statistics"]["observations"] = 109
    report["sfm_result"]["statistics"]["mean_reprojection_error_px"] = 1.0129207143082428
    report["quality_assessment"]["mean_track_length"] = 2.5952380952380953
    report["sfm_result"]["metadata"]["quality_assessment"]["mean_track_length"] = 2.5952380952380953
    report["quality_assessment"]["reasons"] = [
        "Registration ratio 5.0% is below the 30.0% rejection threshold.",
        "Landmark density 14.0 per registered keyframe is low.",
    ]
    report["sfm_result"]["metadata"]["quality_assessment"]["reasons"] = list(report["quality_assessment"]["reasons"])
    analysis = analyze_i5_10c_sfm_report(report, preflight=p)
    # Force the exact observed scale-invariant diagnostic cardinality; D validates
    # the bounded interpretation, not absolute pose values.
    analysis["forward_motion_diagnostic"] = {
        "scale_invariant": True,
        "adjacent_registered_motion_pairs": 1,
        "forward_axis_dominant": False,
        "median_forward_component_fraction": 0.4823517536978073,
        "median_lateral_component_fraction": 0.8759776171253749,
        "absolute_translation_magnitude_cross_run_comparison_permitted": False,
        "records": [{
            "name_a": "0000000009.png",
            "name_b": "0000000010.png",
            "forward_component_fraction": 0.4823517536978073,
            "lateral_component_fraction": 0.8759776171253749,
        }],
    }
    return build_i5_10c_real_evaluation(
        preflight=p,
        preflight_binding={"path": "/evidence/i5_10b.json", "sha256": I510B_SHA},
        post_i5_10b_delta_binding={"path": "/checkpoint/i510b_delta.tar.gz", "sha256": "5fc21a70bc9825036037c5aa953aa67f44dc94f154a795282a50a590388bcf3a"},
        evaluation_receipt=synthetic_receipt(),
        evaluation_receipt_binding={"path": "/evidence/i5_10c_receipt.json", "sha256": "c" * 64},
        execution_analysis=analysis,
        execution_record={
            "candidate_command": ["python", "scripts/run_reconstruction_pycolmap_smoke.py", "--images", "/stage", "--sfm-python", "/sfm/python", "--output-dir", "/out", "--profile", "efficient", "--matching", "exhaustive", "--max-keyframes", "60", "--image-sequence-fps", "10.0", "--timeout-minutes", "60.0"],
            "attempt_started_utc": "2026-09-14T08:59:06Z",
            "attempt_finished_utc": "2026-09-14T08:59:22Z",
            "return_code": 1,
            "console_log_sha256": "a" * 64,
            "report_sha256": "b" * 64,
            "source_integrity_after": True,
            "staged_integrity_after": True,
            "worker_integrity_after": True,
            "attempt_directory": "/attempt",
        },
        source_integrity_before=True,
        source_integrity_after=True,
        staged_integrity_after=True,
        worker_integrity_before=True,
        worker_integrity_after=True,
    )


def synthetic_i510d_receipt() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "phase": "I5.10D_CLOSURE_RECEIPT",
        "status": "pass",
        "pass": True,
        "scope": {
            "focused_i5_10d_tests": True,
            "predecessor_i5_10c_regression": True,
            "evidence_only": True,
            "sfm_execution": False,
            "mvs_execution": False,
            "provider_inference": False,
        },
        "counts": {
            "focused_tests": 1,
            "predecessor_total_tests": 1,
            "combined_exercised_tests": 2,
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


def clone(value: Any) -> Any:
    return deepcopy(value)
