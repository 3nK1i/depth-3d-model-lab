"""I4.3G4C blinded matched-view review and formal G4 closure contracts.

G4C consumes only frozen I4.3G4B production-native evidence. It never reruns
PyCOLMAP or a depth provider. Reviewer-facing material uses a deterministic
six-way A-F blind assignment, one shared PyCOLMAP-derived 3D visual frame,
equal canonical point draw counts where canonical geometry exists, and the
same frozen provider-native keyframes for 2D depth diagnostics.

Production-native status remains primary evidence. A G4B ``rejected`` result
stays rejected regardless of manual preference. G4C never selects/promotes a
provider and never creates Auto, hybrid, metric-scale, measurement, metrology,
or mesh authority.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from core.reconstruction.errors import ReconstructionValidationError
from core.reconstruction.single_image_provider_production_qualification import (
    I43G4B_MANUAL_REVIEW_DIMENSIONS,
    I43G4B_PROVIDER_IMPLEMENTATIONS,
    I43G4B_SCENARIOS,
    validate_single_image_provider_run_evidence,
)

I43G4C_SCHEMA_VERSION = 1
I43G4C_PHASE = "I4.3G4C"
I43G4C_SCENARIOS = I43G4B_SCENARIOS
I43G4C_PROVIDER_IMPLEMENTATIONS = I43G4B_PROVIDER_IMPLEMENTATIONS
I43G4C_REVIEW_DIMENSIONS = I43G4B_MANUAL_REVIEW_DIMENSIONS
I43G4C_BLIND_LABELS = ("A", "B", "C", "D", "E", "F")
I43G4C_ALLOWED_PRODUCTION_STATUSES = ("pass", "degraded", "rejected")

_WORKER_RESULT_MANIFESTS = {
    "metric3d_v2": "single_image_depth_result_manifest.json",
    "unidepth_v2": "unidepth_v2_result_manifest.json",
    "moge_2": "moge2_result_manifest.json",
    "depth_pro": "depth_pro_result_manifest.json",
    "depth_anything_v2": "depth_anything_v2_result_manifest.json",
    "marigold": "marigold_result_manifest.json",
}
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


def read_json_object(path: str | Path, *, label: str) -> dict[str, Any]:
    source = Path(path).expanduser().resolve()
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ReconstructionValidationError(f"I4.3G4C could not read {label} {source}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ReconstructionValidationError(f"I4.3G4C {label} must contain a JSON object: {source}")
    return payload


def verify_sha256(path: str | Path, expected: str, *, label: str) -> str:
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise ReconstructionValidationError(f"I4.3G4C {label} is missing: {source}")
    expected_text = str(expected).strip().lower()
    if len(expected_text) != 64 or any(ch not in "0123456789abcdef" for ch in expected_text):
        raise ReconstructionValidationError(f"I4.3G4C {label} expected SHA256 is invalid.")
    actual = sha256_file(source)
    if actual != expected_text:
        raise ReconstructionValidationError(
            f"I4.3G4C {label} SHA256 mismatch: expected={expected_text} actual={actual}"
        )
    return actual


def _mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ReconstructionValidationError(f"I4.3G4C requires mapping section: {label}")
    return value


def _sequence(value: Any, *, label: str) -> tuple[Any, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ReconstructionValidationError(f"I4.3G4C requires sequence section: {label}")
    return tuple(value)


def verify_g1_single_image_review_requirement(
    g1_snapshot: Mapping[str, Any], *, scenario_id: str
) -> Mapping[str, Any]:
    if g1_snapshot.get("phase") != "I4.3G1" or g1_snapshot.get("pass") is not True:
        raise ReconstructionValidationError("I4.3G4C requires a passing I4.3G1 qualification snapshot.")
    matrix = _sequence(g1_snapshot.get("scenario_matrix"), label="G1 scenario_matrix")
    matches = [item for item in matrix if isinstance(item, Mapping) and item.get("scenario_id") == scenario_id]
    if len(matches) != 1:
        raise ReconstructionValidationError(f"I4.3G4C requires exactly one G1 scenario entry for {scenario_id}.")
    entry = matches[0]
    if entry.get("substep") != "I4.3G4":
        raise ReconstructionValidationError("I4.3G4C G1 scenario substep drifted.")
    if entry.get("manual_review_required") is not True or entry.get("new_real_execution_required") is not True:
        raise ReconstructionValidationError("I4.3G4C requires the frozen G1 single-image manual-review contract.")
    modes = _sequence(entry.get("comparison_modes"), label="G1 comparison_modes")
    if "matched_view_diagnostic" not in modes or "production_native" not in modes:
        raise ReconstructionValidationError("I4.3G4C requires both G1 production-native and matched-view modes.")
    dimensions = tuple(
        str(value)
        for value in _sequence(entry.get("manual_review_dimensions"), label="G1 manual_review_dimensions")
    )
    if dimensions != I43G4C_REVIEW_DIMENSIONS:
        raise ReconstructionValidationError("I4.3G4C manual-review dimensions drifted from frozen G1.")
    return entry


@dataclass(frozen=True, slots=True)
class SingleImageProviderReviewSource:
    implementation: str
    status: str
    evidence_path: Path
    evidence_sha256: str
    selected_keyframe_ids: tuple[str, ...]
    worker_manifest_path: Path
    worker_manifest_sha256: str
    native_depth_artifacts: Mapping[str, tuple[Path, str]]
    canonical_geometry_path: Path | None
    canonical_geometry_sha256: str | None


@dataclass(frozen=True, slots=True)
class SingleImageMatchedViewSources:
    scenario_id: str
    production_snapshot_path: Path
    production_snapshot_sha256: str
    authoritative_sfm_path: Path
    authoritative_sfm_sha256: str
    authoritative_sfm_frame_id: str
    providers: Mapping[str, SingleImageProviderReviewSource]
    common_keyframe_ids: tuple[str, ...]

    @property
    def production_statuses(self) -> dict[str, str]:
        return {name: self.providers[name].status for name in I43G4C_PROVIDER_IMPLEMENTATIONS}


def _artifact_index(evidence: Mapping[str, Any], *, implementation: str) -> dict[Path, str]:
    artifacts = _sequence(evidence.get("artifacts"), label=f"{implementation} artifacts")
    result: dict[Path, str] = {}
    for raw in artifacts:
        item = _mapping(raw, label=f"{implementation} artifact")
        path = Path(str(item.get("path", ""))).expanduser().resolve()
        expected = str(item.get("sha256", "")).strip().lower()
        if not path.is_file() or len(expected) != 64:
            raise ReconstructionValidationError(f"I4.3G4C {implementation} frozen artifact identity is incomplete: {path}")
        verify_sha256(path, expected, label=f"{implementation} artifact")
        result[path] = expected
    return result


def _provider_review_source(provider_ref: Mapping[str, Any], *, implementation: str) -> SingleImageProviderReviewSource:
    ref_path = Path(str(provider_ref.get("path", ""))).expanduser().resolve()
    ref_sha = verify_sha256(ref_path, str(provider_ref.get("sha256", "")), label=f"{implementation} G4B evidence")
    evidence = read_json_object(ref_path, label=f"{implementation} G4B evidence")
    validate_single_image_provider_run_evidence(evidence)
    if evidence.get("provider_implementation") != implementation:
        raise ReconstructionValidationError(f"I4.3G4C provider evidence implementation drift for {implementation}.")
    status = str(evidence.get("status", ""))
    if status not in I43G4C_ALLOWED_PRODUCTION_STATUSES:
        raise ReconstructionValidationError(f"I4.3G4C {implementation} production status is invalid: {status}")

    adaptive = _mapping(evidence.get("adaptive_plan_identity"), label=f"{implementation} adaptive plan")
    selected = tuple(str(value) for value in _sequence(adaptive.get("selected_view_ids"), label=f"{implementation} selected views"))
    if not selected or len(set(selected)) != len(selected):
        raise ReconstructionValidationError(f"I4.3G4C {implementation} selected-view identity is invalid.")

    artifacts = _artifact_index(evidence, implementation=implementation)
    worker_basename = _WORKER_RESULT_MANIFESTS[implementation]
    worker_matches = [path for path in artifacts if path.name == worker_basename]
    if len(worker_matches) != 1:
        raise ReconstructionValidationError(
            f"I4.3G4C {implementation} requires exactly one frozen worker manifest {worker_basename}."
        )
    worker_path = worker_matches[0]
    worker = read_json_object(worker_path, label=f"{implementation} worker manifest")
    records = _sequence(worker.get("frame_records"), label=f"{implementation} worker frame_records")
    if len(records) != len(selected):
        raise ReconstructionValidationError(f"I4.3G4C {implementation} worker/selected-view count drifted.")
    native: dict[str, tuple[Path, str]] = {}
    worker_base = worker_path.parent
    for raw in records:
        record = _mapping(raw, label=f"{implementation} worker frame record")
        keyframe_id = str(record.get("keyframe_id", "")).strip()
        path = Path(str(record.get("evidence_npz_path", ""))).expanduser()
        if not path.is_absolute():
            path = (worker_base / path).resolve()
        else:
            path = path.resolve()
        if keyframe_id not in selected:
            raise ReconstructionValidationError(f"I4.3G4C {implementation} worker frame is outside the frozen adaptive plan.")
        # Successful E-series runs freeze the worker manifest and canonical
        # fused outputs, but historically did not list every worker-frame NPZ
        # as a top-level G4B artifact. Rejected G4B runs intentionally do.
        # G4C must not retroactively pretend an unlisted frame file was
        # content-addressed by G4B, so only explicitly hash-bound native frames
        # are admitted as review evidence.
        if path in artifacts:
            native[keyframe_id] = (path, artifacts[path])

    canonical_matches = [
        path for path in artifacts
        if path.name.startswith("canonical_") and path.name.endswith("_fused.npz")
    ]
    if status == "rejected":
        if canonical_matches:
            raise ReconstructionValidationError(f"I4.3G4C rejected {implementation} unexpectedly exposes canonical fused geometry.")
        if tuple(key for key in selected if key in native) != selected:
            raise ReconstructionValidationError(
                f"I4.3G4C rejected {implementation} requires hash-bound native frame evidence for every selected view."
            )
        canonical_path = None
        canonical_sha = None
    else:
        if len(canonical_matches) != 1:
            raise ReconstructionValidationError(
                f"I4.3G4C {implementation} requires exactly one frozen canonical fused NPZ; found {len(canonical_matches)}."
            )
        canonical_path = canonical_matches[0]
        canonical_sha = artifacts[canonical_path]

    return SingleImageProviderReviewSource(
        implementation=implementation,
        status=status,
        evidence_path=ref_path,
        evidence_sha256=ref_sha,
        selected_keyframe_ids=selected,
        worker_manifest_path=worker_path,
        worker_manifest_sha256=artifacts[worker_path],
        native_depth_artifacts=native,
        canonical_geometry_path=canonical_path,
        canonical_geometry_sha256=canonical_sha,
    )


def load_g4b_matched_view_sources(
    production_snapshot_path: str | Path,
    *,
    expected_sha256: str,
    scenario_id: str,
) -> SingleImageMatchedViewSources:
    if scenario_id not in I43G4C_SCENARIOS:
        raise ReconstructionValidationError(f"I4.3G4C unsupported scenario: {scenario_id}")
    prod_path = Path(production_snapshot_path).expanduser().resolve()
    prod_sha = verify_sha256(prod_path, expected_sha256, label="G4B production snapshot")
    prod = read_json_object(prod_path, label="G4B production snapshot")
    if prod.get("phase") != "I4.3G4B" or prod.get("pass") is not True:
        raise ReconstructionValidationError("I4.3G4C requires a passing I4.3G4B production snapshot.")
    if prod.get("status") not in {"production_native_complete", "production_native_complete_with_provider_rejection"}:
        raise ReconstructionValidationError("I4.3G4C G4B snapshot is not terminally production-complete.")
    scenario = _mapping(prod.get("scenario"), label="G4B scenario")
    if scenario.get("scenario_id") != scenario_id:
        raise ReconstructionValidationError("I4.3G4C G4B scenario id mismatch.")
    accounting = _mapping(prod.get("accounting"), label="G4B accounting")
    if accounting.get("all_required_providers_accounted") is not True or int(accounting.get("terminal_provider_count", 0)) != 6:
        raise ReconstructionValidationError("I4.3G4C requires all six G4B provider terminal outcomes.")
    manual = _mapping(prod.get("manual_review"), label="G4B manual_review")
    if manual.get("required") is not True or manual.get("deferred_to") != I43G4C_PHASE:
        raise ReconstructionValidationError("I4.3G4C G4B manual-review handoff is not authorized.")
    if tuple(manual.get("dimensions", ())) != I43G4C_REVIEW_DIMENSIONS:
        raise ReconstructionValidationError("I4.3G4C G4B manual-review dimensions drifted.")
    if manual.get("production_rejection_cannot_be_overridden") is not True:
        raise ReconstructionValidationError("I4.3G4C requires immutable G4B production rejection semantics.")

    authority = _mapping(prod.get("authority"), label="G4B authority")
    if authority.get("camera_pose_authority_owner") != "pycolmap_sfm" or authority.get("permanent_global_view_cap") is not None:
        raise ReconstructionValidationError("I4.3G4C G4B camera/view-cap authority drifted.")
    if authority.get("same_authoritative_sfm_shared_by_all_six_providers") is not True:
        raise ReconstructionValidationError("I4.3G4C requires one shared authoritative SfM for all six providers.")
    drifted = [key for key in _REQUIRED_AUTHORITY_FALSE if authority.get(key) is not False]
    if drifted:
        raise ReconstructionValidationError(f"I4.3G4C forbidden G4B authority drift: {drifted}")

    providers_ref = _mapping(prod.get("providers"), label="G4B providers")
    if set(providers_ref) != set(I43G4C_PROVIDER_IMPLEMENTATIONS):
        raise ReconstructionValidationError("I4.3G4C G4B provider evidence coverage is incomplete.")
    providers = {
        implementation: _provider_review_source(
            _mapping(providers_ref[implementation], label=f"{implementation} provider ref"),
            implementation=implementation,
        )
        for implementation in I43G4C_PROVIDER_IMPLEMENTATIONS
    }
    frozen_statuses = _mapping(prod.get("production_statuses"), label="G4B production_statuses")
    for implementation in I43G4C_PROVIDER_IMPLEMENTATIONS:
        if str(frozen_statuses.get(implementation, "")) != providers[implementation].status:
            raise ReconstructionValidationError(f"I4.3G4C {implementation} production status drifted across G4B evidence.")

    common = set(providers[I43G4C_PROVIDER_IMPLEMENTATIONS[0]].selected_keyframe_ids)
    for implementation in I43G4C_PROVIDER_IMPLEMENTATIONS[1:]:
        common.intersection_update(providers[implementation].selected_keyframe_ids)
    ordered_reference = providers[I43G4C_PROVIDER_IMPLEMENTATIONS[0]].selected_keyframe_ids
    common_ordered = tuple(key for key in ordered_reference if key in common)
    if not common_ordered:
        raise ReconstructionValidationError("I4.3G4C six-provider matched-view intersection is empty.")

    sfm = _mapping(prod.get("authoritative_sfm"), label="G4B authoritative_sfm")
    sfm_path = Path(str(sfm.get("path", scenario.get("authoritative_sfm_snapshot_path", "")))).expanduser().resolve()
    sfm_sha = str(sfm.get("sha256", scenario.get("authoritative_sfm_snapshot_sha256", ""))).lower()
    verify_sha256(sfm_path, sfm_sha, label="frozen authoritative SfM")
    frame_id = str(scenario.get("authoritative_sfm_frame_id", ""))
    if not frame_id:
        raise ReconstructionValidationError("I4.3G4C authoritative SfM frame id is missing.")

    return SingleImageMatchedViewSources(
        scenario_id=scenario_id,
        production_snapshot_path=prod_path,
        production_snapshot_sha256=prod_sha,
        authoritative_sfm_path=sfm_path,
        authoritative_sfm_sha256=sfm_sha,
        authoritative_sfm_frame_id=frame_id,
        providers=providers,
        common_keyframe_ids=common_ordered,
    )


def blind_assignment(seed_sha256: str, *, scenario_id: str) -> dict[str, str]:
    seed = str(seed_sha256).strip().lower()
    if len(seed) != 64 or any(ch not in "0123456789abcdef" for ch in seed):
        raise ReconstructionValidationError("I4.3G4C blind seed must be a SHA256 value.")
    if scenario_id not in I43G4C_SCENARIOS:
        raise ReconstructionValidationError("I4.3G4C blind plan scenario is invalid.")
    ranked = sorted(
        I43G4C_PROVIDER_IMPLEMENTATIONS,
        key=lambda provider: hashlib.sha256(
            f"{seed}|{scenario_id}|{provider}|I4.3G4C|six-way-blind-v1".encode("utf-8")
        ).hexdigest(),
    )
    return {label: provider for label, provider in zip(I43G4C_BLIND_LABELS, ranked, strict=True)}


def assignment_commitment(
    assignment: Mapping[str, str], *, seed_sha256: str, scenario_id: str
) -> str:
    if set(assignment) != set(I43G4C_BLIND_LABELS) or set(assignment.values()) != set(I43G4C_PROVIDER_IMPLEMENTATIONS):
        raise ReconstructionValidationError("I4.3G4C blind assignment is not a six-provider permutation.")
    canonical = json.dumps(
        {
            "scenario_id": scenario_id,
            "seed_sha256": str(seed_sha256).lower(),
            "assignment": {label: str(assignment[label]) for label in I43G4C_BLIND_LABELS},
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def build_manual_review_template(
    *,
    scenario_id: str,
    assignment_commitment_sha256: str,
    common_keyframe_ids: Sequence[str],
    canonical_available_labels: Sequence[str],
) -> dict[str, Any]:
    canonical_available = tuple(str(value) for value in canonical_available_labels)
    if any(value not in I43G4C_BLIND_LABELS for value in canonical_available):
        raise ReconstructionValidationError("I4.3G4C canonical-availability blind label is invalid.")
    return {
        "schema_version": I43G4C_SCHEMA_VERSION,
        "phase": I43G4C_PHASE,
        "scenario_id": scenario_id,
        "comparison_mode": "matched_view_diagnostic",
        "blinded": True,
        "blind_labels": list(I43G4C_BLIND_LABELS),
        "assignment_commitment_sha256": str(assignment_commitment_sha256).lower(),
        "review_dimensions": list(I43G4C_REVIEW_DIMENSIONS),
        "common_keyframe_ids": [str(value) for value in common_keyframe_ids],
        "canonical_geometry_available_blind_labels": list(canonical_available),
        "dimension_reviews": {
            dimension: {
                "preferred_blind_labels": [],
                "not_assessable_blind_labels": [],
                "notes": "",
                "dimension_confirmation": False,
            }
            for dimension in I43G4C_REVIEW_DIMENSIONS
        },
        "scenario_notes": "",
        "reviewer_confirmation": False,
        "blind_key_revealed": False,
        "global_winner_forbidden": True,
        "provider_ranking_forbidden": True,
        "provider_selection_performed": False,
        "production_native_evidence_remains_primary": True,
        "production_rejection_cannot_be_overridden": True,
    }


def review_is_complete(review: Mapping[str, Any]) -> bool:
    if review.get("phase") != I43G4C_PHASE or review.get("blinded") is not True:
        return False
    if tuple(review.get("blind_labels", ())) != I43G4C_BLIND_LABELS:
        return False
    dimensions = review.get("dimension_reviews")
    if not isinstance(dimensions, Mapping) or set(dimensions) != set(I43G4C_REVIEW_DIMENSIONS):
        return False
    labels = set(I43G4C_BLIND_LABELS)
    for dimension in I43G4C_REVIEW_DIMENSIONS:
        item = dimensions.get(dimension)
        if not isinstance(item, Mapping):
            return False
        preferred = item.get("preferred_blind_labels")
        not_assessable = item.get("not_assessable_blind_labels")
        if not isinstance(preferred, list) or not isinstance(not_assessable, list):
            return False
        if len(set(preferred)) != len(preferred) or len(set(not_assessable)) != len(not_assessable):
            return False
        if any(value not in labels for value in preferred + not_assessable):
            return False
        if set(preferred).intersection(not_assessable):
            return False
        if item.get("dimension_confirmation") is not True:
            return False
        if set(not_assessable) == labels and not str(item.get("notes", "")).strip():
            return False
    return (
        review.get("reviewer_confirmation") is True
        and review.get("blind_key_revealed") is False
        and review.get("global_winner_forbidden") is True
        and review.get("provider_ranking_forbidden") is True
        and review.get("provider_selection_performed") is False
        and review.get("production_rejection_cannot_be_overridden") is True
    )


def build_controlled_reveal(
    *,
    review: Mapping[str, Any],
    production_snapshot_sha256: str,
    production_statuses: Mapping[str, str],
    prepare_summary_sha256: str,
) -> dict[str, Any]:
    if not review_is_complete(review):
        raise ReconstructionValidationError("I4.3G4C blind assignment cannot be revealed before completed review.")
    scenario_id = str(review.get("scenario_id", ""))
    assignment = blind_assignment(production_snapshot_sha256, scenario_id=scenario_id)
    commitment = assignment_commitment(
        assignment,
        seed_sha256=production_snapshot_sha256,
        scenario_id=scenario_id,
    )
    if review.get("assignment_commitment_sha256") != commitment:
        raise ReconstructionValidationError("I4.3G4C blind assignment commitment mismatch.")
    if set(production_statuses) != set(I43G4C_PROVIDER_IMPLEMENTATIONS):
        raise ReconstructionValidationError("I4.3G4C controlled reveal requires all six frozen production statuses.")
    if any(str(value) not in I43G4C_ALLOWED_PRODUCTION_STATUSES for value in production_statuses.values()):
        raise ReconstructionValidationError("I4.3G4C controlled reveal contains invalid production status.")

    raw_reviews = _mapping(review.get("dimension_reviews"), label="manual dimension reviews")
    provider_preferences: dict[str, list[str]] = {}
    provider_not_assessable: dict[str, list[str]] = {}
    for dimension in I43G4C_REVIEW_DIMENSIONS:
        item = _mapping(raw_reviews[dimension], label=f"{dimension} review")
        provider_preferences[dimension] = [assignment[str(label)] for label in item["preferred_blind_labels"]]
        provider_not_assessable[dimension] = [assignment[str(label)] for label in item["not_assessable_blind_labels"]]

    return {
        "schema_version": I43G4C_SCHEMA_VERSION,
        "phase": I43G4C_PHASE,
        "status": "scenario_review_complete",
        "pass": True,
        "scenario_id": scenario_id,
        "comparison_mode": "matched_view_diagnostic",
        "production_native_evidence_primary": True,
        "matched_view_diagnostic_secondary": True,
        "prepare_summary_sha256": str(prepare_summary_sha256).lower(),
        "production_snapshot_sha256": str(production_snapshot_sha256).lower(),
        "blind_assignment": {label: assignment[label] for label in I43G4C_BLIND_LABELS},
        "blind_assignment_commitment_sha256": commitment,
        "raw_blinded_dimension_reviews": {
            dimension: dict(_mapping(raw_reviews[dimension], label=f"{dimension} review"))
            for dimension in I43G4C_REVIEW_DIMENSIONS
        },
        "provider_preferences": provider_preferences,
        "provider_not_assessable": provider_not_assessable,
        "production_statuses": {
            implementation: str(production_statuses[implementation])
            for implementation in I43G4C_PROVIDER_IMPLEMENTATIONS
        },
        "decision": {
            "scenario_manual_review_complete": True,
            "production_rejection_overridden_by_manual_review": False,
            "provider_selected": False,
            "provider_promoted": False,
            "provider_ranking_created": False,
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
    if payload.get("phase") != I43G4C_PHASE or payload.get("status") != "scenario_review_complete" or payload.get("pass") is not True:
        raise ReconstructionValidationError("I4.3G4C controlled reveal is not complete/passing.")
    if payload.get("scenario_id") not in I43G4C_SCENARIOS:
        raise ReconstructionValidationError("I4.3G4C controlled reveal scenario is invalid.")
    statuses = _mapping(payload.get("production_statuses"), label="controlled reveal production_statuses")
    if set(statuses) != set(I43G4C_PROVIDER_IMPLEMENTATIONS):
        raise ReconstructionValidationError("I4.3G4C controlled reveal provider coverage is incomplete.")
    if any(str(value) not in I43G4C_ALLOWED_PRODUCTION_STATUSES for value in statuses.values()):
        raise ReconstructionValidationError("I4.3G4C controlled reveal production status is invalid.")
    decision = _mapping(payload.get("decision"), label="controlled reveal decision")
    required_false = (
        "production_rejection_overridden_by_manual_review",
        "provider_selected",
        "provider_promoted",
        "provider_ranking_created",
        "universal_provider_winner_declared",
        "global_scalar_score_created",
    )
    if decision.get("scenario_manual_review_complete") is not True:
        raise ReconstructionValidationError("I4.3G4C controlled reveal manual-review completion drifted.")
    if any(decision.get(key) is not False for key in required_false):
        raise ReconstructionValidationError("I4.3G4C controlled reveal decision authority drifted.")
    authority = _mapping(payload.get("authority"), label="controlled reveal authority")
    if authority.get("camera_pose_authority_owner") != "pycolmap_sfm" or authority.get("permanent_global_view_cap") is not None:
        raise ReconstructionValidationError("I4.3G4C controlled reveal authority drifted.")
    drifted = [key for key in _REQUIRED_AUTHORITY_FALSE if authority.get(key) is not False]
    if drifted:
        raise ReconstructionValidationError(f"I4.3G4C controlled reveal forbidden authority drift: {drifted}")


def build_i4_3g4c_final_closure(reveals: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    payloads = tuple(reveals)
    if len(payloads) != len(I43G4C_SCENARIOS):
        raise ReconstructionValidationError("I4.3G4C final closure requires exactly three scenario reveals.")
    by_id: dict[str, Mapping[str, Any]] = {}
    for payload in payloads:
        validate_controlled_reveal(payload)
        scenario_id = str(payload.get("scenario_id"))
        if scenario_id in by_id:
            raise ReconstructionValidationError("I4.3G4C final closure received duplicate scenario evidence.")
        by_id[scenario_id] = payload
    if tuple(sid for sid in I43G4C_SCENARIOS if sid in by_id) != I43G4C_SCENARIOS:
        raise ReconstructionValidationError("I4.3G4C final closure scenario coverage is incomplete.")

    synthesis: dict[str, Any] = {}
    for scenario_id in I43G4C_SCENARIOS:
        payload = by_id[scenario_id]
        synthesis[scenario_id] = {
            "production_statuses": dict(_mapping(payload.get("production_statuses"), label=f"{scenario_id} production_statuses")),
            "provider_preferences": dict(_mapping(payload.get("provider_preferences"), label=f"{scenario_id} provider_preferences")),
            "provider_not_assessable": dict(_mapping(payload.get("provider_not_assessable"), label=f"{scenario_id} provider_not_assessable")),
            "manual_review_complete": True,
            "matched_view_diagnostic_secondary": True,
            "production_rejection_overridden": False,
            "provider_ranking_created": False,
            "global_provider_winner_declared": False,
        }

    return {
        "schema_version": I43G4C_SCHEMA_VERSION,
        "phase": I43G4C_PHASE,
        "status": "formally_closed",
        "pass": True,
        "scenario_count": len(I43G4C_SCENARIOS),
        "all_required_single_image_scenarios_accounted": True,
        "all_required_manual_reviews_complete": True,
        "all_six_provider_outcomes_accounted_per_scenario": True,
        "i4_3g4_closed": True,
        "i4_3g5_authorized": True,
        "scenario_synthesis": synthesis,
        "decision": {
            "production_native_evidence_primary": True,
            "matched_view_diagnostic_secondary": True,
            "scenario_specific_verdicts_preserved": True,
            "quality_rejections_preserved": True,
            "provider_selected": False,
            "provider_promoted": False,
            "provider_ranking_created": False,
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
        "next": "I4.3G5 cross-provider synthesis and G closure.",
    }


__all__ = [
    "I43G4C_SCHEMA_VERSION",
    "I43G4C_PHASE",
    "I43G4C_SCENARIOS",
    "I43G4C_PROVIDER_IMPLEMENTATIONS",
    "I43G4C_REVIEW_DIMENSIONS",
    "I43G4C_BLIND_LABELS",
    "SingleImageProviderReviewSource",
    "SingleImageMatchedViewSources",
    "sha256_file",
    "read_json_object",
    "verify_sha256",
    "verify_g1_single_image_review_requirement",
    "load_g4b_matched_view_sources",
    "blind_assignment",
    "assignment_commitment",
    "build_manual_review_template",
    "review_is_complete",
    "build_controlled_reveal",
    "validate_controlled_reveal",
    "build_i4_3g4c_final_closure",
]
