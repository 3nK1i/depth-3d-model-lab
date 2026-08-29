"""I4.3G3C matched-view temporal-provider review and closure contracts.

G3C consumes only frozen I4.3G3B production-native evidence. It never reruns
SfM, RollingDepth, or Video Depth Anything. The matched-view diagnostic uses the
same contiguous ordered provider-output positions, the same authoritative SfM
workspace RGB frames, and one fixed sequence-global display normalization per
provider. Per-frame visual auto-fit is forbidden.

The diagnostic remains secondary to production-native evidence. A G3B
``rejected`` provider stays rejected even if a blinded review prefers some of
its visible temporal behavior. G3C does not select/promote a provider and does
not create Auto, hybrid, metric-scale, measurement, metrology, or mesh
authority.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from core.reconstruction.errors import ReconstructionValidationError
from core.reconstruction.temporal_provider_production_qualification import (
    I43G3B_PROVIDER_IMPLEMENTATIONS,
    I43G3B_SCENARIOS,
    read_json_object,
    validate_temporal_provider_run_evidence,
)

I43G3C_SCHEMA_VERSION = 1
I43G3C_PHASE = "I4.3G3C"
I43G3C_SCENARIOS = I43G3B_SCENARIOS
I43G3C_PROVIDER_IMPLEMENTATIONS = I43G3B_PROVIDER_IMPLEMENTATIONS
I43G3C_REVIEW_DIMENSIONS = (
    "temporal_flicker",
    "temporal_drift",
    "static_geometry_consistency",
    "motion_boundary_stability",
    "transient_artifacts",
    "textureless_region_stability",
)
I43G3C_ALLOWED_RATINGS = ("A", "B", "tie", "not_assessable")
I43G3C_DEFAULT_WINDOW_FRAMES = 32

_REQUIRED_AUTHORITY_FALSE = (
    "authoritative_sfm_modified",
    "provider_selected",
    "provider_promoted",
    "auto_policy_implemented",
    "hybrid_authorized",
    "metric_scale_claimed",
    "measurement_authoritative",
    "certified_metrology",
    "surface_mesh_generated",
)


def sha256_file(path: str | Path) -> str:
    source = Path(path).expanduser().resolve()
    digest = hashlib.sha256()
    with source.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_sha256(path: str | Path, expected: str, *, label: str) -> str:
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise ReconstructionValidationError(f"I4.3G3C {label} is missing: {source}")
    expected_text = str(expected).strip().lower()
    if len(expected_text) != 64 or any(ch not in "0123456789abcdef" for ch in expected_text):
        raise ReconstructionValidationError(f"I4.3G3C {label} expected SHA256 is invalid.")
    actual = sha256_file(source)
    if actual != expected_text:
        raise ReconstructionValidationError(
            f"I4.3G3C {label} SHA256 mismatch: expected={expected_text} actual={actual}"
        )
    return actual


def _mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ReconstructionValidationError(f"I4.3G3C requires mapping section: {label}")
    return value


def _sequence(value: Any, *, label: str) -> tuple[Any, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ReconstructionValidationError(f"I4.3G3C requires sequence section: {label}")
    return tuple(value)


def verify_g1_temporal_review_requirement(
    g1_snapshot: Mapping[str, Any], *, scenario_id: str
) -> Mapping[str, Any]:
    if g1_snapshot.get("phase") != "I4.3G1" or g1_snapshot.get("pass") is not True:
        raise ReconstructionValidationError("I4.3G3C requires a passing I4.3G1 qualification snapshot.")
    matrix = _sequence(g1_snapshot.get("scenario_matrix"), label="G1 scenario_matrix")
    matches = [item for item in matrix if isinstance(item, Mapping) and item.get("scenario_id") == scenario_id]
    if len(matches) != 1:
        raise ReconstructionValidationError(f"I4.3G3C requires exactly one G1 scenario entry for {scenario_id}.")
    entry = matches[0]
    if entry.get("substep") != "I4.3G3":
        raise ReconstructionValidationError("I4.3G3C G1 scenario substep drifted.")
    if entry.get("manual_review_required") is not True or entry.get("new_real_execution_required") is not True:
        raise ReconstructionValidationError("I4.3G3C requires the frozen G1 temporal manual-review contract.")
    modes = _sequence(entry.get("comparison_modes"), label="G1 comparison_modes")
    if "matched_view_diagnostic" not in modes or "production_native" not in modes:
        raise ReconstructionValidationError("I4.3G3C requires both G1 production-native and matched-view modes.")
    dimensions = tuple(str(value) for value in _sequence(entry.get("manual_review_dimensions"), label="G1 manual_review_dimensions"))
    if dimensions != I43G3C_REVIEW_DIMENSIONS:
        raise ReconstructionValidationError("I4.3G3C temporal manual-review dimensions drifted from G1.")
    return entry


@dataclass(frozen=True, slots=True)
class TemporalProviderReviewSource:
    implementation: str
    status: str
    evidence_path: Path
    evidence_sha256: str
    summary_path: Path
    summary_sha256: str
    depth_npz_path: Path
    depth_npz_sha256: str
    ordered_keyframe_ids: tuple[str, ...]
    ordered_source_sequence_indices: tuple[int, ...]
    ordered_timestamps_ns: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class TemporalMatchedViewSources:
    scenario_id: str
    production_snapshot_path: Path
    production_snapshot_sha256: str
    source_file_sha256: str
    sfm_snapshot_path: Path
    sfm_snapshot_sha256: str
    sfm_frame_id: str
    workspace_root: Path
    workspace_images: tuple[Mapping[str, Any], ...]
    rolling_depth: TemporalProviderReviewSource
    video_depth_anything: TemporalProviderReviewSource

    @property
    def view_count(self) -> int:
        return len(self.rolling_depth.ordered_keyframe_ids)


def _artifact_by_basename(evidence: Mapping[str, Any], basename: str, *, label: str) -> tuple[Path, str]:
    artifacts = _sequence(evidence.get("artifacts"), label=f"{label} artifacts")
    matches: list[tuple[Path, str]] = []
    for item in artifacts:
        if not isinstance(item, Mapping):
            continue
        path = Path(str(item.get("path", ""))).expanduser().resolve()
        expected = str(item.get("sha256", "")).strip().lower()
        if path.name == basename:
            matches.append((path, expected))
    if len(matches) != 1:
        raise ReconstructionValidationError(
            f"I4.3G3C {label} requires exactly one frozen {basename} artifact; found {len(matches)}."
        )
    path, expected = matches[0]
    verify_sha256(path, expected, label=f"{label} {basename}")
    return path, expected


def _load_provider_source(
    provider_ref: Mapping[str, Any], *, implementation: str
) -> TemporalProviderReviewSource:
    evidence_ref = _mapping(provider_ref.get("evidence"), label=f"{implementation} evidence reference")
    evidence_path = Path(str(evidence_ref.get("path", ""))).expanduser().resolve()
    evidence_sha = verify_sha256(
        evidence_path,
        str(evidence_ref.get("sha256", "")),
        label=f"{implementation} G3B evidence",
    )
    evidence = read_json_object(evidence_path, label=f"I4.3G3C {implementation} G3B evidence")
    validate_temporal_provider_run_evidence(evidence)
    if evidence.get("provider_implementation") != implementation:
        raise ReconstructionValidationError(f"I4.3G3C provider evidence implementation drift for {implementation}.")

    summary_basename = (
        "i4_3d2_rolling_depth_execution_summary.json"
        if implementation == "rolling_depth"
        else "i4_3d1_vda_execution_summary.json"
    )
    depth_basename = "rolling_depth_evidence.npz" if implementation == "rolling_depth" else "vda_depth_evidence.npz"
    summary_path, summary_sha = _artifact_by_basename(evidence, summary_basename, label=implementation)
    depth_path, depth_sha = _artifact_by_basename(evidence, depth_basename, label=implementation)

    temporal = _mapping(evidence.get("temporal_sequence"), label=f"{implementation} temporal_sequence")
    keyframes = tuple(str(value) for value in _sequence(temporal.get("ordered_keyframe_ids"), label=f"{implementation} keyframes"))
    indices = tuple(int(value) for value in _sequence(temporal.get("ordered_source_sequence_indices"), label=f"{implementation} source indices"))
    timestamps = tuple(int(value) for value in _sequence(temporal.get("ordered_timestamps_ns"), label=f"{implementation} timestamps"))
    if not keyframes or not (len(keyframes) == len(indices) == len(timestamps)):
        raise ReconstructionValidationError(f"I4.3G3C {implementation} temporal sequence length is invalid.")
    if len(set(keyframes)) != len(keyframes):
        raise ReconstructionValidationError(f"I4.3G3C {implementation} keyframe ids are not unique.")
    if any(b <= a for a, b in zip(indices, indices[1:])):
        raise ReconstructionValidationError(f"I4.3G3C {implementation} source sequence order is not strictly increasing.")
    if any(b <= a for a, b in zip(timestamps, timestamps[1:])):
        raise ReconstructionValidationError(f"I4.3G3C {implementation} timestamps are not strictly increasing.")

    return TemporalProviderReviewSource(
        implementation=implementation,
        status=str(evidence.get("status")),
        evidence_path=evidence_path,
        evidence_sha256=evidence_sha,
        summary_path=summary_path,
        summary_sha256=summary_sha,
        depth_npz_path=depth_path,
        depth_npz_sha256=depth_sha,
        ordered_keyframe_ids=keyframes,
        ordered_source_sequence_indices=indices,
        ordered_timestamps_ns=timestamps,
    )


def load_g3b_matched_view_sources(
    production_snapshot_path: str | Path,
    *,
    expected_sha256: str,
    scenario_id: str,
) -> TemporalMatchedViewSources:
    if scenario_id not in I43G3C_SCENARIOS:
        raise ReconstructionValidationError(f"I4.3G3C unsupported temporal scenario: {scenario_id}")
    prod_path = Path(production_snapshot_path).expanduser().resolve()
    prod_sha = verify_sha256(prod_path, expected_sha256, label="G3B production snapshot")
    prod = read_json_object(prod_path, label="I4.3G3C G3B production snapshot")
    if prod.get("phase") != "I4.3G3B" or prod.get("pass") is not True:
        raise ReconstructionValidationError("I4.3G3C requires a passing I4.3G3B production snapshot.")
    if prod.get("status") not in {"production_native_complete", "production_native_complete_with_provider_rejection"}:
        raise ReconstructionValidationError("I4.3G3C requires terminal G3B production-native evidence.")
    scenario = _mapping(prod.get("scenario"), label="G3B scenario")
    if scenario.get("scenario_id") != scenario_id:
        raise ReconstructionValidationError("I4.3G3C production scenario id does not match the requested scenario.")

    decision = _mapping(prod.get("decision"), label="G3B decision")
    if decision.get("production_native_complete") is not True:
        raise ReconstructionValidationError("I4.3G3C requires complete G3B production-native evidence.")
    if decision.get("matched_view_diagnostic_authorized_for_g3c") is not True:
        raise ReconstructionValidationError("I4.3G3C matched-view diagnostic is not authorized by G3B.")
    if decision.get("scenario_verdict_pending_g3c_manual_review") is not True:
        raise ReconstructionValidationError("I4.3G3C requires the G3B scenario verdict to remain pending manual review.")

    authority = _mapping(prod.get("authority"), label="G3B authority")
    if authority.get("camera_pose_authority_owner") != "pycolmap_sfm":
        raise ReconstructionValidationError("I4.3G3C requires PyCOLMAP camera/scene authority.")
    if authority.get("same_authoritative_sfm_shared_by_both_providers") is not True:
        raise ReconstructionValidationError("I4.3G3C requires one shared authoritative SfM within the scenario.")
    if authority.get("permanent_global_view_cap") is not None:
        raise ReconstructionValidationError("I4.3G3C refuses a permanent global view cap.")
    drifted = [key for key in _REQUIRED_AUTHORITY_FALSE if authority.get(key) is not False]
    if drifted:
        raise ReconstructionValidationError(f"I4.3G3C G3B authority drift: {drifted}")

    providers = _mapping(prod.get("providers"), label="G3B providers")
    rolling_ref = _mapping(providers.get("rolling_depth"), label="RollingDepth provider block")
    vda_ref = _mapping(providers.get("video_depth_anything"), label="VDA provider block")
    rolling = _load_provider_source(rolling_ref, implementation="rolling_depth")
    vda = _load_provider_source(vda_ref, implementation="video_depth_anything")

    # G3B snapshots created before the terminal-provider-rejection correction did
    # not persist provider_terminal_outcomes_complete. They are still valid G3C
    # predecessors when the snapshot is the clean production_native_complete form
    # and both independently validated provider evidence records have terminal
    # non-rejection statuses. Explicit False always fails closed, and rejection
    # snapshots must carry the post-correction terminal-outcome assertion.
    provider_statuses = {rolling.status, vda.status}
    terminal_outcomes = decision.get("provider_terminal_outcomes_complete")
    production_status = str(prod.get("status"))
    if terminal_outcomes is False:
        raise ReconstructionValidationError("I4.3G3C requires complete G3B provider terminal outcomes.")
    if terminal_outcomes is not True:
        legacy_clean_snapshot = (
            terminal_outcomes is None
            and production_status == "production_native_complete"
            and "rejected" not in provider_statuses
        )
        if not legacy_clean_snapshot:
            raise ReconstructionValidationError("I4.3G3C requires complete G3B provider terminal outcomes.")
    if production_status == "production_native_complete_with_provider_rejection" and "rejected" not in provider_statuses:
        raise ReconstructionValidationError("I4.3G3C G3B rejection status is inconsistent with provider evidence.")
    if production_status == "production_native_complete" and "rejected" in provider_statuses:
        raise ReconstructionValidationError("I4.3G3C clean G3B status cannot contain rejected provider evidence.")

    if rolling.ordered_keyframe_ids != vda.ordered_keyframe_ids:
        raise ReconstructionValidationError("I4.3G3C providers do not share the same ordered keyframe ids.")
    if rolling.ordered_source_sequence_indices != vda.ordered_source_sequence_indices:
        raise ReconstructionValidationError("I4.3G3C providers do not share the same source-sequence indices.")
    if rolling.ordered_timestamps_ns != vda.ordered_timestamps_ns:
        raise ReconstructionValidationError("I4.3G3C providers do not share the same canonical timestamps.")

    rolling_evidence = read_json_object(rolling.evidence_path, label="RollingDepth G3B evidence")
    vda_evidence = read_json_object(vda.evidence_path, label="VDA G3B evidence")
    rolling_source = _mapping(rolling_evidence.get("source_identity"), label="RollingDepth source identity")
    vda_source = _mapping(vda_evidence.get("source_identity"), label="VDA source identity")
    source_sha = str(rolling_source.get("source_file_sha256", ""))
    if source_sha != str(vda_source.get("source_file_sha256", "")) or source_sha != str(scenario.get("source_file_sha256", "")):
        raise ReconstructionValidationError("I4.3G3C source identity differs across frozen G3B evidence.")

    rolling_sfm = _mapping(rolling_evidence.get("authoritative_sfm_identity"), label="RollingDepth SfM identity")
    vda_sfm = _mapping(vda_evidence.get("authoritative_sfm_identity"), label="VDA SfM identity")
    sfm_sha = str(rolling_sfm.get("snapshot_sha256", ""))
    frame_id = str(rolling_sfm.get("frame_id", ""))
    if sfm_sha != str(vda_sfm.get("snapshot_sha256", "")) or frame_id != str(vda_sfm.get("frame_id", "")):
        raise ReconstructionValidationError("I4.3G3C providers do not share the same frozen authoritative SfM identity.")
    sfm_path = Path(str(rolling_sfm.get("snapshot_path", ""))).expanduser().resolve()
    verify_sha256(sfm_path, sfm_sha, label="authoritative SfM snapshot")
    sfm = read_json_object(sfm_path, label="I4.3G3C authoritative SfM snapshot")
    workspace = _mapping(sfm.get("workspace"), label="authoritative SfM workspace")
    workspace_root = Path(str(workspace.get("workspace_root", ""))).expanduser().resolve()
    images = tuple(item for item in _sequence(workspace.get("images"), label="authoritative SfM workspace images") if isinstance(item, Mapping))
    if len(images) < len(rolling.ordered_keyframe_ids):
        raise ReconstructionValidationError("I4.3G3C authoritative SfM workspace image coverage is incomplete.")

    return TemporalMatchedViewSources(
        scenario_id=scenario_id,
        production_snapshot_path=prod_path,
        production_snapshot_sha256=prod_sha,
        source_file_sha256=source_sha,
        sfm_snapshot_path=sfm_path,
        sfm_snapshot_sha256=sfm_sha,
        sfm_frame_id=frame_id,
        workspace_root=workspace_root,
        workspace_images=images,
        rolling_depth=rolling,
        video_depth_anything=vda,
    )


def load_depth_stack(path: str | Path, *, expected_view_count: int, label: str) -> np.ndarray:
    source = Path(path).expanduser().resolve()
    try:
        with np.load(source, allow_pickle=False) as archive:
            if "depth" not in archive.files:
                raise ReconstructionValidationError(f"I4.3G3C {label} NPZ lacks depth array.")
            depth = np.asarray(archive["depth"], dtype=np.float32)
    except (OSError, ValueError) as exc:
        raise ReconstructionValidationError(f"I4.3G3C could not load {label} depth evidence: {exc}") from exc
    if depth.ndim != 3 or depth.shape[0] != int(expected_view_count):
        raise ReconstructionValidationError(
            f"I4.3G3C {label} depth stack must be NxHxW for all matched views; got {depth.shape}."
        )
    if not np.isfinite(depth).all():
        raise ReconstructionValidationError(f"I4.3G3C {label} depth evidence contains non-finite values.")
    return depth


def rolling_depth_display_multiplier(summary: Mapping[str, Any]) -> float:
    canonicalization = _mapping(summary.get("depth_canonicalization"), label="RollingDepth depth_canonicalization")
    if canonicalization.get("transform") != "provider_native_polarity_sign_flip":
        raise ReconstructionValidationError("I4.3G3C requires the frozen RollingDepth polarity canonicalization.")
    multiplier = float(canonicalization.get("multiplier", 0.0))
    if multiplier != -1.0 or canonicalization.get("sequence_order_changed") is not False:
        raise ReconstructionValidationError("I4.3G3C RollingDepth canonicalization provenance drifted.")
    if canonicalization.get("physical_metric_scale_claimed") is not False:
        raise ReconstructionValidationError("I4.3G3C refuses metric authority from RollingDepth display canonicalization.")
    return multiplier


def centered_contiguous_window(view_count: int, requested_frames: int = I43G3C_DEFAULT_WINDOW_FRAMES) -> tuple[int, ...]:
    count = int(view_count)
    requested = int(requested_frames)
    if count < 2 or requested < 2:
        raise ReconstructionValidationError("I4.3G3C matched-view window requires at least two frames.")
    length = min(count, requested)
    start = (count - length) // 2
    return tuple(range(start, start + length))


def sequence_global_display_range(depth: np.ndarray, *, multiplier: float = 1.0) -> tuple[float, float]:
    data = np.asarray(depth, dtype=np.float32)
    if data.ndim != 3 or not np.isfinite(data).all():
        raise ReconstructionValidationError("I4.3G3C display normalization requires finite NxHxW depth.")
    # Deterministic spatial sampling keeps quantile memory bounded while the
    # normalization remains one fixed range for the full frozen sequence.
    sampled = (data[:, ::8, ::8] * float(multiplier)).reshape(-1)
    low, high = np.quantile(sampled.astype(np.float64, copy=False), [0.02, 0.98])
    low_value = float(low)
    high_value = float(high)
    if not np.isfinite(low_value) or not np.isfinite(high_value) or high_value <= low_value:
        low_value = float(np.min(sampled))
        high_value = float(np.max(sampled))
    if high_value <= low_value:
        raise ReconstructionValidationError("I4.3G3C sequence-global display range is degenerate.")
    return low_value, high_value


def blind_assignment(seed_sha256: str, *, scenario_id: str) -> dict[str, str]:
    seed = str(seed_sha256).strip().lower()
    if len(seed) != 64 or any(ch not in "0123456789abcdef" for ch in seed):
        raise ReconstructionValidationError("I4.3G3C blind seed must be a SHA256 value.")
    if scenario_id not in I43G3C_SCENARIOS:
        raise ReconstructionValidationError("I4.3G3C blind plan scenario is invalid.")
    digest = hashlib.sha256((seed + "|" + scenario_id + "|I4.3G3C|temporal-blind-v1").encode("utf-8")).digest()
    if digest[0] & 1:
        return {"A": "video_depth_anything", "B": "rolling_depth"}
    return {"A": "rolling_depth", "B": "video_depth_anything"}


def assignment_commitment(assignment: Mapping[str, str], *, seed_sha256: str, scenario_id: str) -> str:
    canonical = json.dumps(
        {
            "scenario_id": scenario_id,
            "seed_sha256": str(seed_sha256).lower(),
            "assignment": {"A": assignment["A"], "B": assignment["B"]},
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def build_manual_review_template(
    *, scenario_id: str, assignment_commitment_sha256: str, window_positions: Sequence[int]
) -> dict[str, Any]:
    return {
        "schema_version": I43G3C_SCHEMA_VERSION,
        "phase": I43G3C_PHASE,
        "scenario_id": scenario_id,
        "comparison_mode": "matched_view_diagnostic",
        "blinded": True,
        "assignment_commitment_sha256": str(assignment_commitment_sha256),
        "allowed_values": list(I43G3C_ALLOWED_RATINGS),
        "review_dimensions": list(I43G3C_REVIEW_DIMENSIONS),
        "matched_window_positions": [int(value) for value in window_positions],
        "scenario_ratings": {dimension: None for dimension in I43G3C_REVIEW_DIMENSIONS},
        "scenario_notes": "",
        "reviewer_confirmation": False,
        "blind_key_revealed": False,
        "global_winner_forbidden": True,
        "provider_selection_performed": False,
        "production_native_evidence_remains_primary": True,
    }


def review_is_complete(review: Mapping[str, Any]) -> bool:
    if review.get("phase") != I43G3C_PHASE or review.get("blinded") is not True:
        return False
    ratings = review.get("scenario_ratings")
    if not isinstance(ratings, Mapping) or set(ratings) != set(I43G3C_REVIEW_DIMENSIONS):
        return False
    if any(value not in I43G3C_ALLOWED_RATINGS for value in ratings.values()):
        return False
    return review.get("reviewer_confirmation") is True and review.get("blind_key_revealed") is False


def build_controlled_reveal(
    *,
    review: Mapping[str, Any],
    production_snapshot_sha256: str,
    production_statuses: Mapping[str, str],
    prepare_summary_sha256: str,
) -> dict[str, Any]:
    if not review_is_complete(review):
        raise ReconstructionValidationError("I4.3G3C blind assignment cannot be revealed before completed review.")
    scenario_id = str(review.get("scenario_id", ""))
    assignment = blind_assignment(production_snapshot_sha256, scenario_id=scenario_id)
    commitment = assignment_commitment(
        assignment,
        seed_sha256=production_snapshot_sha256,
        scenario_id=scenario_id,
    )
    if review.get("assignment_commitment_sha256") != commitment:
        raise ReconstructionValidationError("I4.3G3C blind assignment commitment mismatch.")
    expected_status_keys = set(I43G3C_PROVIDER_IMPLEMENTATIONS)
    if set(production_statuses) != expected_status_keys:
        raise ReconstructionValidationError("I4.3G3C controlled reveal requires both frozen provider statuses.")
    if any(value not in {"pass", "degraded", "rejected"} for value in production_statuses.values()):
        raise ReconstructionValidationError("I4.3G3C controlled reveal provider status is invalid.")

    raw_ratings = _mapping(review.get("scenario_ratings"), label="manual review ratings")
    preferences: dict[str, str | None] = {}
    for dimension in I43G3C_REVIEW_DIMENSIONS:
        rating = str(raw_ratings[dimension])
        preferences[dimension] = assignment.get(rating) if rating in {"A", "B"} else None

    return {
        "schema_version": I43G3C_SCHEMA_VERSION,
        "phase": I43G3C_PHASE,
        "status": "scenario_review_complete",
        "pass": True,
        "scenario_id": scenario_id,
        "comparison_mode": "matched_view_diagnostic",
        "production_native_evidence_primary": True,
        "matched_view_diagnostic_secondary": True,
        "prepare_summary_sha256": str(prepare_summary_sha256).lower(),
        "production_snapshot_sha256": str(production_snapshot_sha256).lower(),
        "blind_assignment": dict(assignment),
        "blind_assignment_commitment_sha256": commitment,
        "raw_blinded_ratings": {dimension: str(raw_ratings[dimension]) for dimension in I43G3C_REVIEW_DIMENSIONS},
        "provider_preferences": preferences,
        "production_statuses": {key: str(production_statuses[key]) for key in I43G3C_PROVIDER_IMPLEMENTATIONS},
        "decision": {
            "scenario_manual_review_complete": True,
            "production_rejection_overridden_by_manual_review": False,
            "provider_selected": False,
            "provider_promoted": False,
            "universal_provider_winner_declared": False,
            "global_scalar_score_created": False,
        },
        "authority": {
            "camera_pose_authority_owner": "pycolmap_sfm",
            "authoritative_sfm_modified": False,
            "provider_selected": False,
            "provider_promoted": False,
            "auto_policy_implemented": False,
            "hybrid_authorized": False,
            "metric_scale_claimed": False,
            "measurement_authoritative": False,
            "certified_metrology": False,
            "surface_mesh_generated": False,
            "permanent_global_view_cap": None,
        },
    }


def validate_controlled_reveal(payload: Mapping[str, Any]) -> None:
    if payload.get("phase") != I43G3C_PHASE or payload.get("status") != "scenario_review_complete" or payload.get("pass") is not True:
        raise ReconstructionValidationError("I4.3G3C controlled reveal is not complete/passing.")
    if payload.get("scenario_id") not in I43G3C_SCENARIOS:
        raise ReconstructionValidationError("I4.3G3C controlled reveal scenario is invalid.")
    decision = _mapping(payload.get("decision"), label="controlled reveal decision")
    if decision.get("scenario_manual_review_complete") is not True or decision.get("production_rejection_overridden_by_manual_review") is not False:
        raise ReconstructionValidationError("I4.3G3C controlled reveal decision drifted.")
    for key in ("provider_selected", "provider_promoted", "universal_provider_winner_declared", "global_scalar_score_created"):
        if decision.get(key) is not False:
            raise ReconstructionValidationError(f"I4.3G3C controlled reveal decision authority drift: {key}")
    authority = _mapping(payload.get("authority"), label="controlled reveal authority")
    if authority.get("camera_pose_authority_owner") != "pycolmap_sfm" or authority.get("permanent_global_view_cap") is not None:
        raise ReconstructionValidationError("I4.3G3C controlled reveal authority drifted.")
    drifted = [key for key in _REQUIRED_AUTHORITY_FALSE if authority.get(key) is not False]
    if drifted:
        raise ReconstructionValidationError(f"I4.3G3C controlled reveal forbidden authority drift: {drifted}")


def build_i4_3g3c_final_closure(reveals: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    payloads = tuple(reveals)
    if len(payloads) != len(I43G3C_SCENARIOS):
        raise ReconstructionValidationError("I4.3G3C final closure requires exactly three scenario reveals.")
    by_id: dict[str, Mapping[str, Any]] = {}
    for payload in payloads:
        validate_controlled_reveal(payload)
        scenario_id = str(payload.get("scenario_id"))
        if scenario_id in by_id:
            raise ReconstructionValidationError("I4.3G3C final closure received duplicate scenario evidence.")
        by_id[scenario_id] = payload
    if tuple(sid for sid in I43G3C_SCENARIOS if sid in by_id) != I43G3C_SCENARIOS:
        raise ReconstructionValidationError("I4.3G3C final closure scenario coverage is incomplete.")

    synthesis: dict[str, Any] = {}
    for scenario_id in I43G3C_SCENARIOS:
        payload = by_id[scenario_id]
        synthesis[scenario_id] = {
            "production_statuses": dict(_mapping(payload.get("production_statuses"), label=f"{scenario_id} production_statuses")),
            "provider_preferences": dict(_mapping(payload.get("provider_preferences"), label=f"{scenario_id} provider_preferences")),
            "manual_review_complete": True,
            "matched_view_diagnostic_secondary": True,
            "production_rejection_overridden": False,
            "global_provider_winner_declared": False,
        }

    return {
        "schema_version": I43G3C_SCHEMA_VERSION,
        "phase": I43G3C_PHASE,
        "status": "formally_closed",
        "pass": True,
        "scenario_count": len(I43G3C_SCENARIOS),
        "all_required_temporal_scenarios_accounted": True,
        "all_required_manual_reviews_complete": True,
        "i4_3g3_closed": True,
        "i4_3g4_authorized": True,
        "scenario_synthesis": synthesis,
        "decision": {
            "production_native_evidence_primary": True,
            "matched_view_diagnostic_secondary": True,
            "scenario_specific_verdicts_preserved": True,
            "quality_failures_preserved": True,
            "provider_selected": False,
            "provider_promoted": False,
            "universal_provider_winner_declared": False,
            "global_scalar_score_created": False,
            "auto_policy_implemented": False,
            "hybrid_authorized": False,
        },
        "authority": {
            "camera_pose_authority_owner": "pycolmap_sfm",
            "authoritative_sfm_modified": False,
            "provider_selected": False,
            "provider_promoted": False,
            "auto_policy_implemented": False,
            "hybrid_authorized": False,
            "metric_scale_claimed": False,
            "measurement_authoritative": False,
            "certified_metrology": False,
            "surface_mesh_generated": False,
            "permanent_global_view_cap": None,
        },
        "next": "I4.3G4 single-image provider qualification.",
    }


__all__ = [
    "I43G3C_SCHEMA_VERSION",
    "I43G3C_PHASE",
    "I43G3C_SCENARIOS",
    "I43G3C_PROVIDER_IMPLEMENTATIONS",
    "I43G3C_REVIEW_DIMENSIONS",
    "I43G3C_ALLOWED_RATINGS",
    "I43G3C_DEFAULT_WINDOW_FRAMES",
    "TemporalProviderReviewSource",
    "TemporalMatchedViewSources",
    "sha256_file",
    "verify_sha256",
    "verify_g1_temporal_review_requirement",
    "load_g3b_matched_view_sources",
    "load_depth_stack",
    "rolling_depth_display_multiplier",
    "centered_contiguous_window",
    "sequence_global_display_range",
    "blind_assignment",
    "assignment_commitment",
    "build_manual_review_template",
    "review_is_complete",
    "build_controlled_reveal",
    "validate_controlled_reveal",
    "build_i4_3g3c_final_closure",
]
