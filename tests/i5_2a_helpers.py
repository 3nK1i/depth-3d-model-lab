from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import cv2
import numpy as np

from core.reconstruction.i5_difficult_handheld_robustness import (
    I51EvidenceBinding,
    build_i5_1c_difficult_handheld_closure,
)
from tests.i5_1b_helpers import passing_i5_1a
from tests.i5_1c_helpers import (
    I51A_SHA,
    I51B_SHA,
    OOM_SHA,
    RECEIPT_SHA,
    degraded_i5_1b,
    oom_diagnostic_text,
    passing_i5_1c_receipt,
)

I51C_SHA = "1" * 64
G2_FINAL_SHA = "2" * 64
G2A3_DOC_SHA = "3" * 64
G2B_DOC_SHA = "4" * 64
G2C_DOC_SHA = "5" * 64
RECEIPT_52A_SHA = "6" * 64


def clone(value):
    return deepcopy(value)


def passing_i5_1c() -> dict:
    return build_i5_1c_difficult_handheld_closure(
        i5_1a_result=passing_i5_1a(),
        i5_1a_binding=I51EvidenceBinding("/evidence/i5_1a.json", I51A_SHA),
        i5_1b_result=degraded_i5_1b(),
        i5_1b_binding=I51EvidenceBinding("/evidence/i5_1b.json", I51B_SHA),
        oom_diagnostic=oom_diagnostic_text(),
        oom_diagnostic_binding=I51EvidenceBinding("/evidence/i5_1b_failed_attempt_diagnostic.txt", OOM_SHA),
        qualification_receipt=passing_i5_1c_receipt(),
        qualification_receipt_binding=I51EvidenceBinding("/evidence/i5_1c_receipt.json", RECEIPT_SHA),
    )



def g2_final() -> dict:
    return {
        "phase": "I4.3G2",
        "status": "formally_closed",
        "all_required_g2_scenarios_accounted": True,
        "all_required_manual_reviews_complete": True,
        "g2a_source_binding_complete": True,
        "g2b_production_native_complete": True,
        "g2c_manual_qualification_complete": True,
        "i4_3g2_closed": True,
        "scenario_synthesis": {
            "g2_weak_overlap_mvs_da3": {
                "controlled_reveal_complete": True,
                "evidence_mode": "fresh_production_plus_matched_view",
                "full_scenario_claim_permitted": False,
                "global_provider_winner_declared": False,
                "manual_review_complete": True,
                "partial_registration_preserved": True,
                "review_scope": "authoritative_registered_views",
            }
        },
        "synthesis": {
            "camera_pose_authority_owner": "pycolmap_sfm",
            "provider_selected": False,
            "provider_promoted": False,
            "universal_provider_winner_declared": False,
            "auto_provider_selection_implemented": False,
            "hybrid_provider_authorized": False,
            "metric_scale_claimed": False,
            "measurement_authoritative": False,
            "certified_metrology": False,
            "surface_mesh_generated": False,
        },
    }

def g2a3_doc() -> str:
    return """# I4.3G2A.3 — Fresh scenario binding and G2A closure gate
### Weak overlap
By default, G2A.3 derives an ordered, increased-spacing subset from the confirmed complete-room capture.
ORB evidence is only a cheap pre-provider connectivity proxy. It is not claimed as authoritative overlap measurement.
G2B must revalidate weak-overlap behavior from the authoritative PyCOLMAP SfM/covisibility evidence.
Failure to produce a valid authoritative SfM remains a legitimate scenario result; G2A.3 does not weaken SfM gates to make the scenario pass.
"""


def g2b_doc() -> str:
    return """# I4.3G2B — MVS vs DA3 production comparison
For weak overlap, authoritative PyCOLMAP co-visibility connectivity is persisted and must qualify before either dense provider starts.
A connectivity failure is preserved as the scenario outcome; matching/source/provider thresholds are not relaxed to erase it.
camera_pose_authority_owner = pycolmap_sfm
permanent_global_view_cap = null
"""


def g2c_doc() -> str:
    return """# I4.3G2C Remaining Static Manual Review Coverage
Weak Overlap preserves its connected five-view authoritative registered scope.
Its review scope is authoritative_registered_views.
No provider or SfM rerun is authorized.
"""


def passing_receipt() -> dict:
    return {
        "schema_version": 1,
        "phase": "I5.2A_ADMISSION_RECEIPT",
        "status": "pass",
        "pass": True,
        "scope": {
            "focused_i5_2a_tests": True,
            "predecessor_i5_1c_regression": True,
            "weak_overlap_methodology_regression": True,
            "source_identity_regression": True,
            "import_probe": True,
            "runner_help_probe": True,
            "provider_execution_started": False,
            "sfm_rerun": False,
            "dense_provider_rerun": False,
            "thresholds_modified": False,
            "production_default_modified": False,
            "auto_policy_implemented": False,
            "hybrid_authorized": False,
        },
        "counts": {
            "focused_tests": 24,
            "predecessor_unit_tests": 30,
            "predecessor_integration_tests": 10,
            "predecessor_stress_tests": 8,
            "predecessor_total_tests": 48,
            "combined_exercised_tests": 72,
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


def make_ordered_dataset(root: Path, count: int = 19) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    for index in range(count):
        image = np.zeros((72, 96, 3), dtype=np.uint8)
        image[:, :, 0] = (index * 7) % 255
        cv2.circle(image, (20 + (index % 12), 30), 8, (255, 255, 255), -1)
        cv2.putText(image, str(index), (42, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2, cv2.LINE_AA)
        assert cv2.imwrite(str(root / f"castle{index:02d}.png"), image)
    return root
