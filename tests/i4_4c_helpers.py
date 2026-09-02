from __future__ import annotations

from copy import deepcopy

from core.reconstruction.offline_dense_cross_provider_visual_verification import (
    VisualVerificationEvidenceBinding,
    build_i4_4b_cross_provider_visual_verification,
)
from core.reconstruction.offline_dense_provenance_authority_audit import (
    ProvenanceAuthorityEvidenceBinding,
    build_i4_4c_provenance_authority_audit,
)
from tests.i4_4a_helpers import passing_h2
from tests.i4_4b_helpers import passing_g2, passing_g3, passing_g4, passing_i4_4a, passing_receipt as passing_i44b_receipt


def passing_i4_4b() -> dict:
    return build_i4_4b_cross_provider_visual_verification(
        i4_3h2_closure=passing_h2(),
        i4_3h2_binding=VisualVerificationEvidenceBinding("/evidence/h2.json", "1" * 64),
        i4_4a_result=passing_i4_4a(),
        i4_4a_binding=VisualVerificationEvidenceBinding("/evidence/i44a.json", "2" * 64),
        i4_3g2_closure=passing_g2(),
        i4_3g2_binding=VisualVerificationEvidenceBinding("/evidence/g2.json", "3" * 64),
        i4_3g3c_closure=passing_g3(),
        i4_3g3c_binding=VisualVerificationEvidenceBinding("/evidence/g3c.json", "4" * 64),
        i4_3g4c_closure=passing_g4(),
        i4_3g4c_binding=VisualVerificationEvidenceBinding("/evidence/g4c.json", "5" * 64),
        visual_receipt=passing_i44b_receipt(),
        visual_receipt_binding=VisualVerificationEvidenceBinding("/evidence/i44b_receipt.json", "6" * 64),
    )


def passing_receipt() -> dict:
    return {
        "schema_version": 1,
        "phase": "I4.4C_PROVENANCE_AUTHORITY_AUDIT_RECEIPT",
        "status": "pass",
        "pass": True,
        "scope": {
            "focused_i4_4c_tests": True,
            "affected_provenance_unit_regression": True,
            "affected_provenance_integration_regression": True,
            "affected_provenance_stress_regression": True,
            "import_probe": True,
            "runner_help_probe": True,
            "provider_execution_started": False,
            "provider_runtime_probe_started": False,
            "sfm_rerun": False,
            "matched_view_rerun": False,
            "manual_review_rerun": False,
            "final_i4_closure_started": False,
        },
        "counts": {
            "focused_tests": 39,
            "affected_test_files": 25,
            "affected_unit_tests": 170,
            "affected_integration_tests": 31,
            "affected_stress_tests": 69,
            "affected_total_tests": 270,
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


def build_passing_i4_4c() -> dict:
    return build_i4_4c_provenance_authority_audit(
        i4_3h2_closure=passing_h2(),
        i4_3h2_binding=ProvenanceAuthorityEvidenceBinding("/evidence/h2.json", "1" * 64),
        i4_4a_result=passing_i4_4a(),
        i4_4a_binding=ProvenanceAuthorityEvidenceBinding("/evidence/i44a.json", "2" * 64),
        i4_4b_result=passing_i4_4b(),
        i4_4b_binding=ProvenanceAuthorityEvidenceBinding("/evidence/i44b.json", "3" * 64),
        audit_receipt=passing_receipt(),
        audit_receipt_binding=ProvenanceAuthorityEvidenceBinding("/evidence/i44c_receipt.json", "4" * 64),
        h2_h1_chain_reverified=True,
        h1_bound_artifacts_reverified=True,
        g5b_predecessor_artifacts_reverified=True,
        i4_4a_chain_reverified=True,
        i4_4b_chain_reverified=True,
        current_repository_authority_matches_frozen_h1=True,
    )


def clone(value: dict) -> dict:
    return deepcopy(value)
