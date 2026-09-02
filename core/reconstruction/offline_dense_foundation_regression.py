"""I4.4A focused regression contracts for the frozen offline dense foundation.

This boundary is execution-light.  It consumes the formally closed I4.3H2
artifact, re-audits the current in-repository offline dense contract surfaces,
and binds a focused regression receipt.  It deliberately does not rerun
providers/SfM or begin I4.4B visual verification, I4.4C provenance/authority
final audit, or I4.4D final I4 closure.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import re
from typing import Any, Mapping, Sequence

from core.reconstruction.adaptive_production_planning import (
    prepare_adaptive_provider_execution,
    project_authoritative_sfm_for_provider,
)
from core.reconstruction.adaptive_view_selection import (
    I43F1_PERMANENT_GLOBAL_VIEW_CAP,
    build_adaptive_view_selection_plan,
)
from core.reconstruction.dense_fusion import DenseDepthUnit, fuse_dense_frames
from core.reconstruction.dense_scene_persistence import (
    read_persistent_dense_rgb_scene,
    verify_persistent_dense_scene_files,
    write_persistent_dense_rgb_scene,
)
from core.reconstruction.errors import ReconstructionValidationError
from core.reconstruction.i4_3_final_qualification import audit_i4_3_provider_architecture
from core.reconstruction.persistent_scene import (
    read_persistent_reconstruction_scene,
    verify_persistent_scene_file_references,
    write_persistent_reconstruction_scene,
)
from core.reconstruction.sfm_contracts import SfmScaleStatus
from core.reconstruction.sfm_dense_alignment import align_sfm_keyframe_dense_depth
from core.reconstruction.sfm_dense_projection import project_sfm_keyframe_dense_rgb

I44A_SCHEMA_VERSION = 1
I44A_PHASE = "I4.4A"
I44A_RECEIPT_PHASE = "I4.4A_REGRESSION_RECEIPT"
I44A_MIN_AFFECTED_TESTS = 200
I44A_MIN_AFFECTED_TEST_FILES = 20

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_EXPECTED_TERMINAL_QUALITY_EVENTS = {
    ("temporal_depth", "g3_temporal_static_sweep", "rolling_depth", "degraded"),
    ("temporal_depth", "g3_temporal_motion", "rolling_depth", "rejected"),
    ("temporal_depth", "g3_temporal_motion", "video_depth_anything", "degraded"),
    ("single_image_depth", "g4_single_image_low_texture", "depth_anything_v2", "rejected"),
    ("single_image_depth", "g4_single_image_low_texture", "marigold", "rejected"),
}


def _mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ReconstructionValidationError(f"I4.4A {label} must be an object.")
    return value


def _sequence(value: Any, *, label: str) -> Sequence[Any]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise ReconstructionValidationError(f"I4.4A {label} must be an array.")
    return value


def _require(mapping: Mapping[str, Any], key: str, expected: Any, *, label: str) -> None:
    if mapping.get(key) != expected:
        raise ReconstructionValidationError(
            f"I4.4A {label}.{key} contract drift: expected={expected!r} actual={mapping.get(key)!r}."
        )


def _positive_int(value: Any, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ReconstructionValidationError(f"I4.4A {label} must be a non-negative integer.")
    return value


@dataclass(frozen=True, slots=True)
class RegressionEvidenceBinding:
    path: str
    sha256: str

    def as_dict(self) -> dict[str, str]:
        path = str(self.path).strip()
        sha = str(self.sha256).strip().lower()
        if not path or not _SHA256_RE.fullmatch(sha):
            raise ReconstructionValidationError("I4.4A evidence binding is malformed.")
        return {"path": path, "sha256": sha}


def _terminal_events(value: Any) -> set[tuple[str, str, str, str]]:
    events: set[tuple[str, str, str, str]] = set()
    for raw in _sequence(value, label="terminal_quality_events"):
        item = _mapping(raw, label="terminal quality event")
        events.add((
            str(item.get("family_id", "")),
            str(item.get("scenario_id", "")),
            str(item.get("provider_implementation", "")),
            str(item.get("production_status", "")),
        ))
    return events


def validate_i4_3h2_for_i4_4a(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Fail closed unless H2 is the exact formal authorization boundary for I4.4A."""
    if payload.get("phase") != "I4.3H2" or payload.get("status") != "formally_closed" or payload.get("pass") is not True:
        raise ReconstructionValidationError("I4.4A requires a passing formal I4.3H2 closure.")

    scope = _mapping(payload.get("scope"), label="H2 scope")
    for key in ("provider_execution_started", "provider_runtime_probe_started", "sfm_rerun", "matched_view_rerun", "manual_review_rerun", "i4_4_work_started"):
        _require(scope, key, False, label="H2 scope")
    _require(scope, "formal_i4_3_closure_performed", True, label="H2 scope")

    coverage = _mapping(payload.get("coverage"), label="H2 coverage")
    expected_coverage = {
        "provider_family_count": 5,
        "executable_provider_family_count": 4,
        "provider_implementation_count": 10,
        "qualification_family_count": 3,
        "qualification_scenario_count": 11,
        "terminal_quality_event_count": 5,
    }
    for key, expected in expected_coverage.items():
        _require(coverage, key, expected, label="H2 coverage")

    decision = _mapping(payload.get("decision"), label="H2 decision")
    for key in ("i4_3h2_formal_closure_complete", "i4_3_production_dense_provider_architecture_frozen", "i4_3_closed", "i4_4_authorized"):
        _require(decision, key, True, label="H2 decision")
    for key in ("provider_selected", "provider_promoted", "provider_ranking_created", "universal_provider_winner_declared", "global_scalar_score_created", "auto_policy_implemented", "hybrid_authorized"):
        _require(decision, key, False, label="H2 decision")

    authority = _mapping(payload.get("authority"), label="H2 authority")
    _require(authority, "camera_pose_authority_owner", "pycolmap_sfm", label="H2 authority")
    _require(authority, "permanent_global_view_cap", None, label="H2 authority")
    for key in ("authoritative_sfm_modified", "provider_selected", "provider_promoted", "auto_policy_implemented", "hybrid_authorized", "metric_scale_claimed", "measurement_authoritative", "certified_metrology", "surface_mesh_generated"):
        _require(authority, key, False, label="H2 authority")

    qualification = _mapping(payload.get("qualification_closure"), label="H2 qualification_closure")
    _require(qualification, "production_native_evidence_primary", True, label="H2 qualification_closure")
    _require(qualification, "matched_view_diagnostic_secondary", True, label="H2 qualification_closure")
    _require(qualification, "scenario_specific_operating_envelopes_preserved", True, label="H2 qualification_closure")
    _require(qualification, "quality_event_counts_are_not_provider_scores", True, label="H2 qualification_closure")
    _require(qualification, "direct_cross_family_provider_ranking_valid", False, label="H2 qualification_closure")
    events = _terminal_events(qualification.get("terminal_quality_events"))
    if events != _EXPECTED_TERMINAL_QUALITY_EVENTS or len(events) != 5:
        raise ReconstructionValidationError("I4.4A H2 terminal-quality event set drifted.")

    handoff = _mapping(payload.get("handoff_to_i4_4"), label="H2 handoff_to_i4_4")
    _require(handoff, "i4_4_authorized", True, label="H2 handoff_to_i4_4")
    _require(handoff, "i4_4_work_started", False, label="H2 handoff_to_i4_4")
    _require(handoff, "first_boundary", "I4.4A focused regression", label="H2 handoff_to_i4_4")
    if payload.get("next") != "I4.4A focused regression.":
        raise ReconstructionValidationError("I4.4A H2 next-boundary contract drifted.")

    return {
        "coverage": deepcopy(dict(coverage)),
        "decision": deepcopy(dict(decision)),
        "authority": deepcopy(dict(authority)),
        "terminal_quality_events": deepcopy(list(qualification["terminal_quality_events"])),
    }


def validate_i4_4a_regression_receipt(payload: Mapping[str, Any]) -> dict[str, Any]:
    if payload.get("phase") != I44A_RECEIPT_PHASE or payload.get("status") != "pass" or payload.get("pass") is not True:
        raise ReconstructionValidationError("I4.4A requires a passing focused-regression receipt.")
    scope = _mapping(payload.get("scope"), label="regression receipt scope")
    for key in ("focused_i4_4a_tests", "affected_unit_regression", "affected_integration_regression", "affected_stress_regression", "import_probe", "runner_help_probe"):
        _require(scope, key, True, label="regression receipt scope")
    for key in ("provider_execution_started", "sfm_rerun", "visual_verification_started"):
        _require(scope, key, False, label="regression receipt scope")

    counts = _mapping(payload.get("counts"), label="regression receipt counts")
    focused = _positive_int(counts.get("focused_tests"), label="focused_tests")
    file_count = _positive_int(counts.get("affected_test_files"), label="affected_test_files")
    unit = _positive_int(counts.get("affected_unit_tests"), label="affected_unit_tests")
    integration = _positive_int(counts.get("affected_integration_tests"), label="affected_integration_tests")
    stress = _positive_int(counts.get("affected_stress_tests"), label="affected_stress_tests")
    total = _positive_int(counts.get("affected_total_tests"), label="affected_total_tests")
    if focused < 20:
        raise ReconstructionValidationError("I4.4A focused-test count is unexpectedly small.")
    if file_count < I44A_MIN_AFFECTED_TEST_FILES or total < I44A_MIN_AFFECTED_TESTS:
        raise ReconstructionValidationError("I4.4A affected regression is too narrow for the frozen offline dense foundation.")
    if total != unit + integration + stress:
        raise ReconstructionValidationError("I4.4A affected regression counts do not sum to affected_total_tests.")

    return_codes = _mapping(payload.get("return_codes"), label="regression receipt return_codes")
    for key in ("focused", "affected_unit", "affected_integration", "affected_stress", "import_probe", "runner_help_probe"):
        _require(return_codes, key, 0, label="regression receipt return_codes")
    return {"counts": deepcopy(dict(counts)), "scope": deepcopy(dict(scope))}


def audit_offline_dense_foundation_contracts() -> dict[str, Any]:
    provider = audit_i4_3_provider_architecture()
    if provider["provider_family_count"] != 5 or provider["provider_implementation_count"] != 10:
        raise ReconstructionValidationError("I4.4A provider architecture inventory drifted.")
    if provider["permanent_global_view_cap"] is not None:
        raise ReconstructionValidationError("I4.4A detected a permanent global view cap.")
    return {
        "provider_architecture": deepcopy(provider),
        "sfm_authority": {
            "camera_pose_authority_owner": "pycolmap_sfm",
            "default_scale_status": SfmScaleStatus.ARBITRARY.value,
            "metric_scale_assumed": False,
        },
        "dense_alignment_projection": {
            "alignment_callable": callable(align_sfm_keyframe_dense_depth),
            "projection_callable": callable(project_sfm_keyframe_dense_rgb),
            "provider_native_pose_becomes_canonical": False,
        },
        "adaptive_planning": {
            "permanent_global_view_cap": I43F1_PERMANENT_GLOBAL_VIEW_CAP,
            "view_selection_callable": callable(build_adaptive_view_selection_plan),
            "sfm_projection_callable": callable(project_authoritative_sfm_for_provider),
            "provider_preparation_callable": callable(prepare_adaptive_provider_execution),
        },
        "fusion_persistence": {
            "fusion_callable": callable(fuse_dense_frames),
            "reconstruction_depth_unit_available": DenseDepthUnit.RECONSTRUCTION_UNITS.value == "reconstruction_units",
            "dense_scene_roundtrip_surfaces_importable": all(callable(x) for x in (read_persistent_dense_rgb_scene, write_persistent_dense_rgb_scene, verify_persistent_dense_scene_files)),
            "canonical_scene_roundtrip_surfaces_importable": all(callable(x) for x in (read_persistent_reconstruction_scene, write_persistent_reconstruction_scene, verify_persistent_scene_file_references)),
        },
        "heavy_execution_started": False,
    }


def build_i4_4a_focused_regression(
    *,
    i4_3h2_closure: Mapping[str, Any],
    i4_3h2_binding: RegressionEvidenceBinding,
    regression_receipt: Mapping[str, Any],
    regression_binding: RegressionEvidenceBinding,
) -> dict[str, Any]:
    validated_h2 = validate_i4_3h2_for_i4_4a(i4_3h2_closure)
    validated_receipt = validate_i4_4a_regression_receipt(regression_receipt)
    contract_audit = audit_offline_dense_foundation_contracts()
    authority = deepcopy(validated_h2["authority"])
    return {
        "schema_version": I44A_SCHEMA_VERSION,
        "phase": I44A_PHASE,
        "status": "focused_regression_passed",
        "pass": True,
        "scope": {
            "focused_regression_only": True,
            "provider_execution_started": False,
            "provider_runtime_probe_started": False,
            "sfm_rerun": False,
            "matched_view_rerun": False,
            "manual_visual_verification_started": False,
            "provenance_authority_final_audit_started": False,
            "final_i4_closure_started": False,
        },
        "evidence_bindings": {
            "i4_3h2_closure": i4_3h2_binding.as_dict(),
            "i4_4a_regression_receipt": regression_binding.as_dict(),
        },
        "foundation_contract_audit": contract_audit,
        "regression_summary": deepcopy(validated_receipt["counts"]),
        "qualification_guard": {
            "terminal_quality_event_count": 5,
            "terminal_quality_events": deepcopy(validated_h2["terminal_quality_events"]),
            "quality_event_counts_are_not_provider_scores": True,
            "provider_ranking_created": False,
            "universal_provider_winner_declared": False,
        },
        "authority": authority,
        "decision": {
            "i4_3_closed": True,
            "i4_4a_complete": True,
            "i4_4b_authorized": True,
            "i4_4b_started": False,
            "i4_4c_started": False,
            "i4_4d_started": False,
            "auto_policy_implemented": False,
            "hybrid_authorized": False,
        },
        "next": "I4.4B cross-provider visual verification.",
    }


__all__ = [
    "I44A_SCHEMA_VERSION", "I44A_PHASE", "I44A_RECEIPT_PHASE",
    "RegressionEvidenceBinding", "validate_i4_3h2_for_i4_4a",
    "validate_i4_4a_regression_receipt", "audit_offline_dense_foundation_contracts",
    "build_i4_4a_focused_regression",
]
