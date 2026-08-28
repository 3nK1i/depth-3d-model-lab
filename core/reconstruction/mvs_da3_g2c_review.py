"""I4.3G2C blinded matched-view review contracts for MVS vs DA3.

This layer begins only after formal I4.3G2B closure.  It binds G2C review to
that frozen closure, preserves the PyCOLMAP authority contract, and keeps the
A/B provider mapping out of the public review package until review completion.

It does not rerun SfM or any dense provider and does not select/promote a
provider.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from core.reconstruction.errors import ReconstructionValidationError
from core.reconstruction.mvs_da3_matched_view import (
    MATCHED_VIEW_REVIEW_DIMENSIONS,
    blind_assignment,
    read_json_object,
    verify_sha256,
)

G2C_SCHEMA_VERSION = 1
G2C_PHASE = "I4.3G2C"
G2C_TEMPLE_RING_SCENARIO = "g2_temple_ring_mvs_da3"
G2C_ALLOWED_RATINGS: tuple[str, ...] = ("A", "B", "tie", "not_assessable")
G2C_RENDER_WIDTH = 1366
G2C_RENDER_HEIGHT = 768


G2C_FRESH_MATCHED_VIEW_SCENARIOS: tuple[str, ...] = (
    "g2_complete_room_mvs_da3",
    "g2_temple_ring_mvs_da3",
    "g2_low_texture_mvs_da3",
    "g2_weak_overlap_mvs_da3",
)


def verify_g1_manual_review_requirement(
    g1_snapshot_path: str | Path,
    *,
    expected_g1_sha256: str,
    scenario_id: str,
) -> dict[str, Any]:
    """Bind G2C to the frozen G1 manual-review/matched-view requirement."""
    path = Path(g1_snapshot_path).expanduser().resolve()
    verify_sha256(path, expected_g1_sha256, label="I4.3G1 qualification snapshot")
    g1 = read_json_object(path, label="I4.3G1 qualification snapshot")
    _require(g1, "phase", "I4.3G1", label="G1 qualification")
    _require(g1, "pass", True, label="G1 qualification")
    matrix = g1.get("scenario_matrix")
    if not isinstance(matrix, list):
        raise ReconstructionValidationError("G1 scenario matrix is missing.")
    entry = next(
        (item for item in matrix if isinstance(item, Mapping) and item.get("scenario_id") == scenario_id),
        None,
    )
    if not isinstance(entry, Mapping):
        raise ReconstructionValidationError(f"G1 scenario is missing: {scenario_id}")
    _require(entry, "required", True, label=f"G1 {scenario_id}")
    _require(entry, "manual_review_required", True, label=f"G1 {scenario_id}")
    modes = entry.get("comparison_modes")
    if not isinstance(modes, list) or "matched_view_diagnostic" not in modes:
        raise ReconstructionValidationError(
            f"G1 {scenario_id} does not authorize matched_view_diagnostic."
        )
    return dict(entry)


def verify_g2b_scenario_closure_for_g2c(
    closure_path: str | Path,
    *,
    expected_closure_sha256: str,
    scenario_id: str,
    expected_verdict_sha256: str,
) -> dict[str, Any]:
    """Verify a fresh G2 scenario is frozen in the formal G2B closure."""
    path = Path(closure_path).expanduser().resolve()
    verify_sha256(path, expected_closure_sha256, label="I4.3G2B final closure")
    closure = read_json_object(path, label="I4.3G2B final closure")
    for key, expected in {
        "phase": "I4.3G2B",
        "status": "closed",
        "i4_3g2b_closed": True,
        "i4_3g2c_ready": True,
        "all_five_g2_scenarios_accounted_for": True,
        "real_provider_execution_phase_complete": True,
    }.items():
        _require(closure, key, expected, label="G2B closure")

    authority = closure.get("authority")
    if not isinstance(authority, Mapping):
        raise ReconstructionValidationError("G2B closure authority block is missing.")
    for key, expected in {
        "camera_pose_authority_owner": "pycolmap_sfm",
        "provider_selected": False,
        "provider_promoted": False,
        "universal_provider_winner_assumed": False,
        "global_scalar_score_forbidden": True,
        "auto_policy_implemented": False,
        "hybrid_authorized": False,
        "metric_scale_claimed": False,
        "measurement_authoritative": False,
        "certified_metrology": False,
        "surface_mesh_generated": False,
        "permanent_global_view_cap": None,
    }.items():
        _require(authority, key, expected, label="G2B closure authority")

    scenarios = closure.get("scenarios")
    if not isinstance(scenarios, Mapping):
        raise ReconstructionValidationError("G2B closure scenarios block is missing.")
    scenario = scenarios.get(scenario_id)
    if not isinstance(scenario, Mapping):
        raise ReconstructionValidationError(f"G2B closure scenario missing: {scenario_id}")
    _require(
        scenario,
        "verdict_sha256",
        str(expected_verdict_sha256).lower(),
        label=f"G2B closure {scenario_id}",
    )
    _require(scenario, "rerun_authorized", False, label=f"G2B closure {scenario_id}")

    if scenario_id in {"g2_complete_room_mvs_da3", "g2_low_texture_mvs_da3"}:
        _require(
            scenario,
            "status",
            "closed_operating_envelope_result",
            label=f"G2B closure {scenario_id}",
        )
        _require(
            scenario,
            "full_scenario_sfm_qualified",
            False,
            label=f"G2B closure {scenario_id}",
        )
        _require(
            scenario,
            "localized_provider_evidence_valid",
            True,
            label=f"G2B closure {scenario_id}",
        )
    elif scenario_id == "g2_temple_ring_mvs_da3":
        _require(scenario, "status", "closed_production_native_result", label="TempleRing closure")
        _require(scenario, "authoritative_sfm_qualified", True, label="TempleRing closure")
        _require(scenario, "production_native_comparison_valid", True, label="TempleRing closure")
    elif scenario_id == "g2_weak_overlap_mvs_da3":
        _require(scenario, "status", "closed_production_native_result", label="Weak Overlap closure")
        _require(
            scenario,
            "authoritative_sfm_connectivity_qualified",
            True,
            label="Weak Overlap closure",
        )
        _require(
            scenario,
            "production_native_comparison_valid",
            True,
            label="Weak Overlap closure",
        )
    else:
        raise ReconstructionValidationError(f"Unsupported fresh G2C scenario: {scenario_id}")

    return closure


def canonical_render_policy() -> dict[str, Any]:
    """Return the fixed reviewer-facing raster contract for every A/B cloud.

    QMainWindow resize requests are not guaranteed to produce identical grabbed
    raster sizes across repeated native/VisPy windows.  G2C therefore freezes
    one canonical output extent and normalizes each successful capture to it
    before the A/B contact sheet is assembled.
    """
    return {
        "width_px": G2C_RENDER_WIDTH,
        "height_px": G2C_RENDER_HEIGHT,
        "capture_normalized_to_canonical_extent": True,
        "independent_image_auto_fit": False,
    }


def _require(mapping: Mapping[str, Any], key: str, expected: Any, *, label: str) -> None:
    actual = mapping.get(key)
    if actual != expected:
        raise ReconstructionValidationError(
            f"{label} contract violation: {key}={actual!r}, expected={expected!r}"
        )


def verify_g2b_closure_for_g2c(
    closure_path: str | Path,
    *,
    expected_closure_sha256: str,
    expected_temple_ring_verdict_sha256: str,
) -> dict[str, Any]:
    """Verify the formal G2B closure is an immutable valid G2C predecessor."""
    path = Path(closure_path).expanduser().resolve()
    verify_sha256(path, expected_closure_sha256, label="I4.3G2B final closure")
    closure = read_json_object(path, label="I4.3G2B final closure")

    _require(closure, "phase", "I4.3G2B", label="G2B closure")
    _require(closure, "status", "closed", label="G2B closure")
    _require(closure, "i4_3g2b_closed", True, label="G2B closure")
    _require(closure, "i4_3g2c_ready", True, label="G2B closure")
    _require(closure, "all_five_g2_scenarios_accounted_for", True, label="G2B closure")
    _require(closure, "real_provider_execution_phase_complete", True, label="G2B closure")

    scenarios = closure.get("scenarios")
    if not isinstance(scenarios, Mapping):
        raise ReconstructionValidationError("G2B closure scenarios block is missing.")
    temple = scenarios.get(G2C_TEMPLE_RING_SCENARIO)
    if not isinstance(temple, Mapping):
        raise ReconstructionValidationError("G2B closure TempleRing scenario is missing.")
    _require(temple, "status", "closed_production_native_result", label="TempleRing G2B closure")
    _require(temple, "authoritative_sfm_qualified", True, label="TempleRing G2B closure")
    _require(temple, "production_native_comparison_valid", True, label="TempleRing G2B closure")
    _require(temple, "rerun_authorized", False, label="TempleRing G2B closure")
    _require(
        temple,
        "verdict_sha256",
        str(expected_temple_ring_verdict_sha256).lower(),
        label="TempleRing G2B closure",
    )

    authority = closure.get("authority")
    if not isinstance(authority, Mapping):
        raise ReconstructionValidationError("G2B closure authority block is missing.")
    required_authority = {
        "camera_pose_authority_owner": "pycolmap_sfm",
        "provider_selected": False,
        "provider_promoted": False,
        "universal_provider_winner_assumed": False,
        "global_scalar_score_forbidden": True,
        "auto_policy_implemented": False,
        "hybrid_authorized": False,
        "metric_scale_claimed": False,
        "measurement_authoritative": False,
        "certified_metrology": False,
        "surface_mesh_generated": False,
        "permanent_global_view_cap": None,
    }
    for key, expected in required_authority.items():
        _require(authority, key, expected, label="G2B closure authority")

    handoff = closure.get("g2c_handoff")
    if not isinstance(handoff, Mapping):
        raise ReconstructionValidationError("G2B closure G2C handoff block is missing.")
    required_handoff = {
        "production_native_results_frozen": True,
        "provider_reexecution_required": False,
        "threshold_retuning_allowed": False,
        "source_rebinding_allowed": False,
        "matched_view_manual_review_required": True,
        "next_phase": "I4.3G2C",
    }
    for key, expected in required_handoff.items():
        _require(handoff, key, expected, label="G2B->G2C handoff")

    return closure


def assignment_commitment(*, production_snapshot_sha256: str, mapping: Mapping[str, str]) -> str:
    canonical = json.dumps(dict(mapping), sort_keys=True, separators=(",", ":"))
    payload = (
        canonical
        + "|"
        + str(production_snapshot_sha256).lower()
        + "|I4.3G2C|blind-commit-v1"
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True, slots=True)
class G2CBlindPlan:
    scenario_id: str
    production_snapshot_sha256: str
    assignment: dict[str, str]
    commitment_sha256: str

    def public_dict(self) -> dict[str, Any]:
        """Return review metadata without the A/B provider mapping."""
        return {
            "schema_version": G2C_SCHEMA_VERSION,
            "phase": G2C_PHASE,
            "scenario_id": self.scenario_id,
            "blinded": True,
            "labels": ["A", "B"],
            "assignment_commitment_sha256": self.commitment_sha256,
            "mapping_in_public_review_package": False,
            "reveal_only_after_completed_manual_review": True,
            "provider_selection_performed": False,
        }


def build_blind_plan(
    production_snapshot_sha256: str,
    *,
    scenario_id: str = G2C_TEMPLE_RING_SCENARIO,
) -> G2CBlindPlan:
    mapping = blind_assignment(production_snapshot_sha256)
    commitment = assignment_commitment(
        production_snapshot_sha256=production_snapshot_sha256,
        mapping=mapping,
    )
    return G2CBlindPlan(
        scenario_id=scenario_id,
        production_snapshot_sha256=str(production_snapshot_sha256).lower(),
        assignment=dict(mapping),
        commitment_sha256=commitment,
    )




def resolve_equal_render_point_count(
    *,
    mvs_source_count: int,
    da3_source_count: int,
    requested_point_budget: int,
) -> int:
    """Return one identical deterministic draw count for both blinded clouds.

    Matched-view review must not encode provider identity through unequal
    render density.  The common draw count is therefore bounded by the
    smaller frozen source cloud and the requested display budget.
    """
    counts = (int(mvs_source_count), int(da3_source_count), int(requested_point_budget))
    if any(value <= 0 for value in counts):
        raise ReconstructionValidationError(
            "G2C matched-view source counts and requested point budget must be positive."
        )
    return min(counts)


def blinded_render_label(*, blind_label: str, view: str, framing: str) -> str:
    """Build a reviewer-facing label that contains no provider/count hints."""
    if blind_label not in {"A", "B"}:
        raise ReconstructionValidationError(f"Unsupported G2C blind label: {blind_label}")
    return (
        f"I4.3G2C BLINDED MATCHED VIEW — CLOUD {blind_label} | {view} | {framing} | "
        "SAME SfM FRAME + SAME CAMERA + SAME DRAW COUNT"
    )

def build_manual_review_template(
    *,
    scenario_id: str,
    views: Sequence[str],
    framings: Sequence[str],
    assignment_commitment_sha256: str,
) -> dict[str, Any]:
    pair_observations = {
        f"{framing}:{view}": {
            "assessable": None,
            "notes": "",
        }
        for framing in framings
        for view in views
    }
    return {
        "schema_version": G2C_SCHEMA_VERSION,
        "phase": G2C_PHASE,
        "scenario_id": scenario_id,
        "comparison_mode": "matched_view_diagnostic",
        "blinded": True,
        "assignment_commitment_sha256": assignment_commitment_sha256,
        "allowed_values": list(G2C_ALLOWED_RATINGS),
        "review_dimensions": list(MATCHED_VIEW_REVIEW_DIMENSIONS),
        "views": list(views),
        "framings": list(framings),
        "pair_observations": pair_observations,
        "scenario_ratings": {
            dimension: None
            for dimension in MATCHED_VIEW_REVIEW_DIMENSIONS
        },
        "scenario_notes": "",
        "reviewer_confirmation": False,
        "blind_key_revealed": False,
        "global_winner_forbidden": True,
        "provider_selection_performed": False,
    }


def review_is_complete(review: Mapping[str, Any]) -> bool:
    if review.get("phase") != G2C_PHASE or review.get("blinded") is not True:
        return False
    ratings = review.get("scenario_ratings")
    if not isinstance(ratings, Mapping):
        return False
    if set(ratings) != set(MATCHED_VIEW_REVIEW_DIMENSIONS):
        return False
    if any(value not in G2C_ALLOWED_RATINGS for value in ratings.values()):
        return False
    return review.get("reviewer_confirmation") is True and review.get("blind_key_revealed") is False


def reveal_assignment_after_review(
    review: Mapping[str, Any],
    *,
    production_snapshot_sha256: str,
) -> dict[str, str]:
    """Reveal only after a completed blinded review and verify its commitment."""
    if not review_is_complete(review):
        raise ReconstructionValidationError(
            "G2C blind assignment cannot be revealed before the manual review is complete."
        )
    plan = build_blind_plan(
        production_snapshot_sha256,
        scenario_id=str(review.get("scenario_id", G2C_TEMPLE_RING_SCENARIO)),
    )
    if review.get("assignment_commitment_sha256") != plan.commitment_sha256:
        raise ReconstructionValidationError("G2C blind assignment commitment mismatch.")
    return dict(plan.assignment)
