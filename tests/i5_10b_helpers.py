from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

from core.reconstruction.i5_difficult_handheld_robustness import I51EvidenceBinding
from core.reconstruction.i5_forward_motion_kitti import build_i5_4c_forward_motion_kitti_closure
from core.reconstruction.i5_mp_sfm_evaluation import build_i5_10a_mp_sfm_admission
from tests.i5_10a_helpers import i59c_binding, receipt_binding, synthetic_i510a_receipt, synthetic_i59c_closure
from tests.i5_4b_helpers import passing_i5_4a
from tests.i5_4c_helpers import binding as i54_binding, passing_i5_4c_receipt, rejected_i5_4b

I510A_SHA = "fca08621136404bd5ecdb7b861123f28f4c624dd3b38be0c35e51989a4f4e2db"
I54A_SHA = "4" * 64
I54B_SHA = "7" * 64
I54C_SHA = "9" * 64
I510B_RECEIPT_SHA = "b" * 64
CHECKPOINT_SHA = "c" * 64


def clone(value: Any) -> Any:
    return deepcopy(value)


def evidence_binding(path: str, sha: str) -> I51EvidenceBinding:
    return I51EvidenceBinding(path, sha)


def synthetic_i510a_admission() -> dict[str, Any]:
    return build_i5_10a_mp_sfm_admission(
        i5_9c_closure=synthetic_i59c_closure(),
        i5_9c_binding=i59c_binding(),
        admission_receipt=synthetic_i510a_receipt(),
        admission_receipt_binding=receipt_binding(),
    )


def historical_chain(tmp_path: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], Path]:
    a = passing_i5_4a(tmp_path)
    source_hash = a["source_identity"]["asset_set_sha256"]
    window_hash = a["stress_plan"]["forward_motion_window"]["window_asset_set_sha256"]

    b = rejected_i5_4b()
    b["evidence_bindings"]["i5_4a_admission"]["sha256"] = I54A_SHA
    b["source_binding"]["source_asset_set_sha256"] = source_hash
    b["source_binding"]["window_asset_set_sha256"] = window_hash

    c = build_i5_4c_forward_motion_kitti_closure(
        i5_4b_result=b,
        i5_4b_binding=i54_binding("/evidence/i5_4b.json", I54B_SHA),
        closure_receipt=passing_i5_4c_receipt(),
        closure_receipt_binding=i54_binding("/evidence/i5_4c_receipt.json", "8" * 64),
    )
    dataset = Path(a["source_identity"]["directory"])
    return a, b, c, dataset


def passing_runtime_probe() -> dict[str, Any]:
    return {
        "python_executable": "/opt/conda/envs/depth-sfm/bin/python",
        "available": True,
        "pycolmap_version": "4.1.1",
        "distribution": "pycolmap-cuda12",
        "distribution_version": "4.1.1",
        "cuda_devices": 1,
        "apis": {
            "extract_features": True,
            "match_exhaustive": True,
            "match_sequential": True,
            "incremental_mapping": True,
            "Reconstruction": True,
            "Camera": True,
            "Rigid3d": True,
            "Rotation3d": True,
            "FeatureExtractionOptions": True,
            "FeatureMatchingOptions": True,
            "IncrementalPipelineOptions": True,
            "FeatureExtractor": True,
            "ImageReaderOptions": True,
            "SequentialPairingOptions": True,
            "Device": True,
            "CameraMode": True,
            "get_num_cuda_devices": True,
        },
        "required_apis_available": True,
        "recommended_version": True,
        "cuda_ready": True,
        "backend_ready": True,
        "error": "",
    }


def synthetic_i510b_receipt() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "phase": "I5.10B_PREFLIGHT_RECEIPT",
        "status": "pass",
        "pass": True,
        "scope": {
            "focused_i5_10b_tests": True,
            "i5_10a_regression": True,
            "i5_4_chain_regression": True,
            "source_identity_regression": True,
            "pycolmap_environment_regression": True,
            "candidate_command_regression": True,
            "authority_guard_regression": True,
            "roadmap_guard_regression": True,
            "import_probe": True,
            "runner_help_probe": True,
            "real_mp_sfm_execution": False,
            "sfm_rerun": False,
            "mvs_rerun": False,
            "thresholds_modified": False,
            "production_default_modified": False,
            "retry_until_pass": False,
            "auto_policy_implemented": False,
            "hybrid_authorized": False,
        },
        "counts": {
            "focused_tests": 24,
            "predecessor_unit_tests": 60,
            "predecessor_integration_tests": 12,
            "predecessor_stress_tests": 12,
            "predecessor_total_tests": 84,
            "combined_exercised_tests": 108,
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
