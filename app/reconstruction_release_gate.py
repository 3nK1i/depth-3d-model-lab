"""Phase 9.6H final Digital Twin release-gate contracts.

This module is deliberately dependency-light.  It validates normalized evidence
produced by already-qualified Phase 9.6 workflows; it never starts DA3,
Nerfstudio, Gaussian Splatting, CUDA, or reconstruction backends.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

PHASE96H = "9.6H"
RELEASE_GATE_SCHEMA = "depth3d.digital-twin.phase9-6h-release-gate.v1"
STATE_EXPORT_SCHEMA = "depth3d.digital-twin.state-export.v1"
MIN_REAL_SCENE_DURATION_S = 120.0
MAX_REAL_SCENE_DURATION_S = 180.0
MIN_VIDEO_WIDTH = 640
MIN_VIDEO_HEIGHT = 360
MIN_SELECTED_KEYFRAMES = 24
MAX_SELECTED_KEYFRAMES = 96
QUALIFIED_NERF_STEPS = frozenset({64, 128, 256})

MANUAL_BOOLEAN_FIELDS: tuple[str, ...] = (
    "canonical_scene_visible",
    "camera_trajectory_visible",
    "estimated_metric_ruler_uncertainty_visible",
    "restart_restore_pass",
    "nerf_interactive_pass",
    "static_preview_nonowning_pass",
    "resource_handoff_pass",
    "no_orphan_cuda_process_after_exit",
    "authority_wording_pass",
)


@dataclass(frozen=True, slots=True)
class Phase96HGateResult:
    passed: bool
    reasons: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return {"pass": self.passed, "reasons": list(self.reasons)}


def _mapping(value: object) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _sha256(value: object) -> bool:
    text = str(value or "").strip().lower()
    return len(text) == 64 and all(ch in "0123456789abcdef" for ch in text)


def _authority_flags_false(payload: Mapping[str, Any], reasons: list[str], prefix: str) -> None:
    for key in (
        "canonical_geometry_authoritative",
        "metric_scale_authoritative",
        "measurement_authoritative",
        "qc_authoritative",
        "certified_metrology",
    ):
        if payload.get(key) is not False:
            reasons.append(f"{prefix}_authority_violation:{key}")


def validate_phase96h_evidence(evidence: Mapping[str, Any]) -> Phase96HGateResult:
    """Validate final machine + explicit-manual release evidence.

    A PASS is a functional Digital Twin release gate.  It intentionally does not
    certify photorealistic convergence, global loop closure, physical metric
    scale, QC authority, or metrology.
    """

    reasons: list[str] = []
    video = _mapping(evidence.get("video"))
    e2 = _mapping(evidence.get("e2_full_scene"))
    e3 = _mapping(evidence.get("e3_nerf"))
    scene = _mapping(evidence.get("persistent_scene"))
    state_export = _mapping(evidence.get("state_export"))
    integrity = _mapping(evidence.get("integrity"))
    manual = _mapping(evidence.get("manual"))
    limitations = _mapping(evidence.get("limitations"))

    # Real-scene source admission.  H intentionally qualifies a 2–3 minute
    # fixed-focal recorded-video campaign rather than another tiny sequence.
    if video.get("readable") is not True:
        reasons.append("source_video_not_readable")
    try:
        duration = float(video.get("duration_s", 0.0))
        fps = float(video.get("fps", 0.0))
        width = int(video.get("width", 0))
        height = int(video.get("height", 0))
        frame_count = int(video.get("frame_count", 0))
    except (TypeError, ValueError):
        duration = fps = 0.0
        width = height = frame_count = 0
    if not (MIN_REAL_SCENE_DURATION_S <= duration <= MAX_REAL_SCENE_DURATION_S):
        reasons.append("source_video_duration_outside_2_to_3_minute_gate")
    if fps <= 0.0 or width < MIN_VIDEO_WIDTH or height < MIN_VIDEO_HEIGHT or frame_count < 1:
        reasons.append("source_video_metadata_invalid_or_too_small")

    # E2: complete recorded-video DA3 full-scene supporting geometry.
    if e2.get("phase") != "9.6E2" or e2.get("pass") is not True:
        reasons.append("e2_full_scene_not_passed")
    selected = int(e2.get("selected_keyframes", 0) or 0)
    windows = int(e2.get("window_count", 0) or 0)
    if not (MIN_SELECTED_KEYFRAMES <= selected <= MAX_SELECTED_KEYFRAMES):
        reasons.append("e2_keyframe_coverage_outside_release_gate")
    if selected >= 3 and windows != selected - 2:
        reasons.append("e2_overlapping_window_count_mismatch")
    if e2.get("geometry_scale") != "relative_arbitrary_scale" or e2.get("unit_to_metre") is not None:
        reasons.append("e2_relative_scale_contract_changed")
    if e2.get("loop_closure") is not False or e2.get("quality_certified") is not False:
        reasons.append("e2_quality_or_loop_closure_overclaimed")
    if e2.get("camera_calibrated") is not False:
        reasons.append("e2_camera_calibration_overclaimed")
    _authority_flags_false(e2, reasons, "e2")

    # Persistent canonical scene produced by E2 and retained through E3.
    if scene.get("canonical_geometry_authoritative") is not True:
        reasons.append("persistent_canonical_geometry_not_authoritative")
    if scene.get("certified_metrology") is not False:
        reasons.append("persistent_scene_metrology_overclaimed")
    if not _sha256(scene.get("scene_fingerprint_sha256")):
        reasons.append("persistent_scene_fingerprint_invalid")
    try:
        if int(scene.get("camera_count", 0) or 0) != selected:
            reasons.append("persistent_camera_count_not_bound_to_e2_views")
    except (TypeError, ValueError):
        reasons.append("persistent_camera_count_invalid")
    if scene.get("references_verified") is not True:
        reasons.append("persistent_scene_references_not_verified")

    # E3: appearance-only immutable NeRF revision bound to the E2 scene/views.
    if e3.get("phase") != "9.6E3" or e3.get("pass") is not True:
        reasons.append("e3_nerf_not_passed")
    if e3.get("source_scene_id") != e2.get("scene_id"):
        reasons.append("e3_source_scene_lineage_mismatch")
    if int(e3.get("input_frame_count", 0) or 0) != selected:
        reasons.append("e3_input_views_not_bound_to_e2_keyframes")
    if int(e3.get("training_steps", 0) or 0) not in QUALIFIED_NERF_STEPS:
        reasons.append("e3_training_profile_outside_qualified_bounds")
    if e3.get("nerf_scene_role") != "appearance_only" or e3.get("quality_certified") is not False:
        reasons.append("e3_appearance_or_quality_contract_changed")
    _authority_flags_false(e3, reasons, "e3")
    if not str(e3.get("persistent_scene_id", "")).strip():
        reasons.append("e3_persistent_revision_missing")

    # F: audit/state export must describe rather than absorb authority/payloads.
    if state_export.get("schema") != STATE_EXPORT_SCHEMA:
        reasons.append("state_export_schema_mismatch")
    if state_export.get("scene_id") != e3.get("persistent_scene_id"):
        reasons.append("state_export_scene_lineage_mismatch")
    if state_export.get("artifact_payloads_embedded") is not False:
        reasons.append("state_export_payload_embedding_contract_changed")
    if state_export.get("portable_payload_bundle") is not False:
        reasons.append("state_export_portability_overclaimed")
    if state_export.get("restore_requires_referenced_files") is not True:
        reasons.append("state_export_restore_dependency_not_declared")
    _authority_flags_false(state_export, reasons, "state_export")
    if not _sha256(state_export.get("scene_fingerprint_sha256")) or not _sha256(state_export.get("workspace_fingerprint_sha256")):
        reasons.append("state_export_fingerprint_invalid")
    referenced = state_export.get("referenced_payloads")
    if not isinstance(referenced, Sequence) or isinstance(referenced, (str, bytes)):
        reasons.append("state_export_referenced_payloads_missing")
    else:
        if int(state_export.get("referenced_payload_count", -1) or -1) != len(referenced):
            reasons.append("state_export_referenced_payload_count_mismatch")
        if len(referenced) < 3:
            reasons.append("state_export_dependency_graph_too_small")
        for item in referenced:
            row = _mapping(item)
            if not str(row.get("path", "")).strip() or not str(row.get("role", "")).strip() or not _sha256(row.get("sha256")):
                reasons.append("state_export_reference_metadata_invalid")
                break

    if integrity.get("persistent_scene_manifest_exists") is not True:
        reasons.append("persistent_scene_manifest_missing")
    if integrity.get("state_export_references_verified") is not True:
        reasons.append("state_export_reference_integrity_failed")

    # Explicit real-UI confirmations.  These cannot be inferred from manifests.
    for key in MANUAL_BOOLEAN_FIELDS:
        if manual.get(key) is not True:
            reasons.append(f"manual_gate_missing:{key}")
    if manual.get("appearance_path") not in {"nerf", "gaussian+nerf"}:
        reasons.append("manual_gate_missing:appearance_path")

    # Release truthfulness: these capabilities remain deliberately un-certified.
    for key in (
        "photorealistic_quality_certified",
        "global_loop_closure_certified",
        "metric_scale_authoritative",
        "certified_metrology",
    ):
        if limitations.get(key) is not False:
            reasons.append(f"release_limitation_overclaimed:{key}")

    unique = tuple(dict.fromkeys(reasons))
    return Phase96HGateResult(passed=not unique, reasons=unique)


def default_manual_evidence(*, appearance_path: str = "nerf") -> dict[str, Any]:
    payload: dict[str, Any] = {key: False for key in MANUAL_BOOLEAN_FIELDS}
    payload["appearance_path"] = appearance_path
    return payload


def qualified_release_limitations() -> dict[str, bool]:
    return {
        "photorealistic_quality_certified": False,
        "global_loop_closure_certified": False,
        "metric_scale_authoritative": False,
        "certified_metrology": False,
    }


__all__ = [
    "MAX_REAL_SCENE_DURATION_S",
    "MAX_SELECTED_KEYFRAMES",
    "MIN_REAL_SCENE_DURATION_S",
    "MIN_SELECTED_KEYFRAMES",
    "MANUAL_BOOLEAN_FIELDS",
    "PHASE96H",
    "QUALIFIED_NERF_STEPS",
    "RELEASE_GATE_SCHEMA",
    "STATE_EXPORT_SCHEMA",
    "Phase96HGateResult",
    "default_manual_evidence",
    "qualified_release_limitations",
    "validate_phase96h_evidence",
]
