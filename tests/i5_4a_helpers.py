from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import cv2
import numpy as np

from core.reconstruction.i5_blur_exposure_clutter_symmetry import (
    build_i5_3c_blur_exposure_clutter_symmetry_closure,
)
from core.reconstruction.i5_difficult_handheld_robustness import I51EvidenceBinding
from tests.i5_3c_helpers import I53B_SHA, RECEIPT_53C_SHA, degraded_i5_3b, passing_i5_3c_receipt
from tests.i5_3b_helpers import I53A_SHA, passing_i5_3a

I53C_SHA = "2" * 64
RECEIPT_54A_SHA = "3" * 64


def clone(value):
    return deepcopy(value)


def binding(path: str, sha: str) -> I51EvidenceBinding:
    return I51EvidenceBinding(path, sha)


def passing_i5_3c(tmp_path: Path) -> dict:
    admission = passing_i5_3a(tmp_path)
    replay = degraded_i5_3b(tmp_path)
    return build_i5_3c_blur_exposure_clutter_symmetry_closure(
        i5_3a_result=admission,
        i5_3a_binding=binding("/evidence/i5_3a.json", I53A_SHA),
        i5_3b_result=replay,
        i5_3b_binding=binding("/evidence/i5_3b.json", I53B_SHA),
        qualification_receipt=passing_i5_3c_receipt(),
        qualification_receipt_binding=binding("/evidence/i5_3c_receipt.json", RECEIPT_53C_SHA),
    )


def kitti_source_dir(tmp_path: Path, count: int = 80) -> Path:
    root = tmp_path / "kitti-seq00-image_2"
    root.mkdir(parents=True, exist_ok=True)
    for index in range(count):
        # Deterministic non-empty image payloads, natural-name ordered like KITTI.
        image = np.full((20, 30, 3), (index * 7) % 255, dtype=np.uint8)
        cv2.putText(image, str(index), (2, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
        ok = cv2.imwrite(str(root / f"{index:06d}.png"), image)
        assert ok
    return root


def passing_i5_4a_receipt() -> dict:
    return {
        "schema_version": 1,
        "phase": "I5.4A_ADMISSION_RECEIPT",
        "status": "pass",
        "pass": True,
        "scope": {
            "focused_i5_4a_tests": True,
            "predecessor_i5_3c_regression": True,
            "ordered_source_identity_regression": True,
            "forward_motion_admission_design_regression": True,
            "import_probe": True,
            "runner_help_probe": True,
            "sfm_rerun": False,
            "images_transformed": False,
            "dense_provider_rerun": False,
            "thresholds_modified": False,
            "production_default_modified": False,
            "mp_sfm_started": False,
            "auto_policy_implemented": False,
            "hybrid_authorized": False,
        },
        "counts": {
            "focused_tests": 14,
            "predecessor_unit_tests": 40,
            "predecessor_integration_tests": 9,
            "predecessor_stress_tests": 12,
            "predecessor_total_tests": 61,
            "combined_exercised_tests": 75,
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
