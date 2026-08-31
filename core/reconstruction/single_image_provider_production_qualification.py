"""I4.3G4B single-image production-native qualification contracts.

G4B consumes the closed G4A preflight, reuses its three frozen G2 source/SfM
contexts, executes each of the six authorized single-image providers through
its existing I4.3E production adapter and I4.3F adaptive view plan, and records
scenario-specific terminal evidence.

A provider-quality rejection is a valid terminal result only when it is the
narrow, pre-existing E-series alignment-acceptance gate. Runtime, integrity,
source, evidence, or authority failures remain hard scenario failures. G4B
never selects/promotes a provider, implements Auto/Hybrid, or claims metric or
metrology authority.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Any, Mapping, Sequence

from core.reconstruction.cross_provider_qualification import (
    ComparisonMode,
    qualification_metric_registry,
    validate_metric_value,
)
from core.reconstruction.errors import ReconstructionValidationError
from core.reconstruction.mvs_da3_qualification import build_source_identity
from core.reconstruction.single_image_provider_qualification import (
    I43G4A_G2_SCENARIO_MAP,
    I43G4A_INFRASTRUCTURE_STATUS_VALUES,
    I43G4A_MANUAL_REVIEW_DIMENSIONS,
    I43G4A_PROVIDER_IMPLEMENTATIONS,
    I43G4A_QUALITY_STATUS_VALUES,
    I43G4A_SCENARIOS,
    validate_runtime_probe,
)

I43G4B_SCHEMA_VERSION = 1
I43G4B_PHASE = "I4.3G4B"
I43G4B_SCENARIOS = I43G4A_SCENARIOS
I43G4B_PROVIDER_IMPLEMENTATIONS = I43G4A_PROVIDER_IMPLEMENTATIONS
I43G4B_QUALITY_STATUS_VALUES = I43G4A_QUALITY_STATUS_VALUES
I43G4B_INFRASTRUCTURE_STATUS_VALUES = I43G4A_INFRASTRUCTURE_STATUS_VALUES
I43G4B_MANUAL_REVIEW_DIMENSIONS = I43G4A_MANUAL_REVIEW_DIMENSIONS

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

_RUNTIME_IDENTITY_FIELDS: dict[str, tuple[str, ...]] = {
    "metric3d_v2": (
        "python_executable", "project_root", "source_directory", "ready_marker",
        "checkpoint_path", "source_revision",
    ),
    "unidepth_v2": (
        "python_executable", "project_root", "source_directory", "compatibility_marker",
        "checkpoint_path", "source_revision", "checkpoint_expected_revision", "variant", "repository",
    ),
    "moge_2": (
        "python_executable", "project_root", "distribution_name", "distribution_version",
        "distribution_root", "source_repository", "source_revision", "variant",
        "checkpoint_repository", "checkpoint_revision",
    ),
    "depth_pro": (
        "python_executable", "project_root", "distribution_name", "distribution_version",
        "distribution_root", "source_repository", "source_revision", "checkpoint_path", "checkpoint_sha256",
    ),
    "depth_anything_v2": (
        "python_executable", "project_root", "source_directory", "source_marker", "source_repository",
        "source_revision", "variant", "encoder", "checkpoint_repository", "checkpoint_declared_revision",
        "checkpoint_path", "checkpoint_sha256",
    ),
    "marigold": (
        "python_executable", "project_root", "distribution_name", "distribution_version", "distribution_root",
        "profile", "checkpoint_repository", "checkpoint_resolved_revision", "checkpoint_path",
        "checkpoint_marker_sha256", "model_index_sha256",
    ),
}

# These are the only E-series exceptions G4B may terminalize as provider-quality
# rejection. They are existing adapter quality gates, not new G4 thresholds.
_QUALITY_REJECTION_PREFIX: dict[str, str] = {
    "metric3d_v2": "I4.3E1 insufficient Metric3D/SfM alignment-accepted views:",
    "unidepth_v2": "I4.3E2 insufficient UniDepthV2/SfM alignment-accepted views:",
    "moge_2": "I4.3E3 insufficient MoGe-2/SfM alignment-accepted views:",
    "depth_pro": "I4.3E4 insufficient Depth Pro/SfM alignment-accepted views:",
    "depth_anything_v2": "I4.3E5 insufficient Depth Anything V2/SfM alignment-accepted views:",
    "marigold": "I4.3E6 insufficient Marigold/SfM alignment-accepted views:",
}


@dataclass(frozen=True, slots=True)
class G4BScenarioBinding:
    scenario_id: str
    predecessor_g2_scenario_id: str
    source_directory: Path
    source_image_count: int
    source_asset_set_sha256: str
    source_manifest_path: Path
    source_manifest_sha256: str
    authoritative_sfm_snapshot_path: Path
    authoritative_sfm_snapshot_sha256: str
    authoritative_sfm_frame_id: str
    registered_view_count: int
    landmark_count: int
    g2b_production_snapshot_path: Path
    g2b_production_snapshot_sha256: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "scenario_id": self.scenario_id,
            "predecessor_g2_scenario_id": self.predecessor_g2_scenario_id,
            "source_directory": str(self.source_directory),
            "source_image_count": self.source_image_count,
            "source_asset_set_sha256": self.source_asset_set_sha256,
            "source_manifest_path": str(self.source_manifest_path),
            "source_manifest_sha256": self.source_manifest_sha256,
            "authoritative_sfm_snapshot_path": str(self.authoritative_sfm_snapshot_path),
            "authoritative_sfm_snapshot_sha256": self.authoritative_sfm_snapshot_sha256,
            "authoritative_sfm_frame_id": self.authoritative_sfm_frame_id,
            "registered_view_count": self.registered_view_count,
            "landmark_count": self.landmark_count,
            "g2b_production_snapshot_path": str(self.g2b_production_snapshot_path),
            "g2b_production_snapshot_sha256": self.g2b_production_snapshot_sha256,
        }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).expanduser().resolve().open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ReconstructionValidationError(f"I4.3G4B requires mapping section: {label}")
    return value


def _sequence(value: Any, *, label: str) -> Sequence[Any]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise ReconstructionValidationError(f"I4.3G4B requires array section: {label}")
    return value


def _valid_sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(ch in "0123456789abcdef" for ch in value.lower())


def _verify_file(path: Path, expected_sha256: str, *, label: str) -> None:
    if not path.is_file():
        raise ReconstructionValidationError(f"I4.3G4B {label} is missing: {path}")
    if not _valid_sha256(expected_sha256):
        raise ReconstructionValidationError(f"I4.3G4B {label} expected SHA256 is invalid.")
    actual = sha256_file(path)
    if actual != expected_sha256.lower():
        raise ReconstructionValidationError(
            f"I4.3G4B {label} SHA256 drift: expected={expected_sha256} actual={actual} path={path}"
        )


def validate_i4_3g4b_predecessor(i4_3g4a_snapshot: Mapping[str, Any]) -> None:
    if (
        i4_3g4a_snapshot.get("phase") != "I4.3G4A"
        or i4_3g4a_snapshot.get("status") != "ready"
        or i4_3g4a_snapshot.get("pass") is not True
    ):
        raise ReconstructionValidationError("I4.3G4B requires a passing closed I4.3G4A preflight snapshot.")

    scope = _mapping(i4_3g4a_snapshot.get("scope"), label="G4A scope")
    if scope.get("execution_free_preflight") is not True:
        raise ReconstructionValidationError("I4.3G4B requires the execution-free G4A preflight contract.")
    if any(scope.get(key) is not False for key in ("sfm_started", "single_image_provider_inference_started", "matched_view_diagnostic_started")):
        raise ReconstructionValidationError("I4.3G4B refuses a G4A snapshot that already started execution/review work.")
    if scope.get("g2_source_sfm_context_reused") is not True or scope.get("new_sfm_variability_introduced") is not False:
        raise ReconstructionValidationError("I4.3G4B requires exact G2 source/SfM context reuse.")

    binding = _mapping(i4_3g4a_snapshot.get("binding_summary"), label="G4A binding_summary")
    if binding.get("bound_scenario_count") != 3 or binding.get("all_g4_sources_bound") is not True:
        raise ReconstructionValidationError("I4.3G4B requires all three G4 source/SfM bindings.")
    if dict(binding.get("g2_context_reuse_map", {})) != I43G4A_G2_SCENARIO_MAP:
        raise ReconstructionValidationError("I4.3G4B G2 context reuse map drifted.")

    runtime_summary = _mapping(i4_3g4a_snapshot.get("runtime_summary"), label="G4A runtime_summary")
    if runtime_summary.get("ready_provider_count") != 6 or runtime_summary.get("all_single_image_runtimes_ready") is not True:
        raise ReconstructionValidationError("I4.3G4B requires all six single-image runtimes ready in G4A.")
    runtime = _mapping(i4_3g4a_snapshot.get("runtime"), label="G4A runtime")
    if set(runtime) != set(I43G4B_PROVIDER_IMPLEMENTATIONS):
        raise ReconstructionValidationError("I4.3G4B G4A runtime coverage drifted.")
    for implementation in I43G4B_PROVIDER_IMPLEMENTATIONS:
        payload = _mapping(runtime[implementation], label=f"G4A runtime {implementation}")
        if payload.get("g4a_ready") is not True or not validate_runtime_probe(payload, implementation=implementation):
            raise ReconstructionValidationError(f"I4.3G4B predecessor runtime is not ready: {implementation}")

    status_contract = _mapping(i4_3g4a_snapshot.get("status_contract"), label="G4A status_contract")
    if tuple(status_contract.get("g4_quality_terminal_status_values", ())) != I43G4B_QUALITY_STATUS_VALUES:
        raise ReconstructionValidationError("I4.3G4B quality status contract drifted.")
    if tuple(status_contract.get("infrastructure_terminal_status_values", ())) != I43G4B_INFRASTRUCTURE_STATUS_VALUES:
        raise ReconstructionValidationError("I4.3G4B infrastructure status contract drifted.")
    if status_contract.get("rejected_is_terminal_provider_quality_evidence") is not True:
        raise ReconstructionValidationError("I4.3G4B requires G4A's terminal provider-quality rejection semantic.")
    if status_contract.get("threshold_weakening_to_avoid_rejection_forbidden") is not True:
        raise ReconstructionValidationError("I4.3G4B refuses threshold weakening to avoid provider rejection.")

    manual = _mapping(i4_3g4a_snapshot.get("manual_review_contract"), label="G4A manual review contract")
    if manual.get("required_for_all_three_scenarios") is not True or manual.get("deferred_to") != "I4.3G4C":
        raise ReconstructionValidationError("I4.3G4B requires G4C manual review to remain deferred and mandatory.")
    if tuple(manual.get("dimensions", ())) != I43G4B_MANUAL_REVIEW_DIMENSIONS:
        raise ReconstructionValidationError("I4.3G4B manual review dimensions drifted.")
    if manual.get("manual_review_cannot_override_production_rejection") is not True:
        raise ReconstructionValidationError("I4.3G4B requires production rejection to remain authoritative over manual review.")

    authority = _mapping(i4_3g4a_snapshot.get("authority"), label="G4A authority")
    if authority.get("camera_pose_authority_owner") != "pycolmap_sfm" or authority.get("permanent_global_view_cap") is not None:
        raise ReconstructionValidationError("I4.3G4B requires PyCOLMAP authority and no permanent global view cap.")
    drifted = [key for key in _REQUIRED_AUTHORITY_FALSE if authority.get(key) is not False]
    if drifted:
        raise ReconstructionValidationError(f"I4.3G4B predecessor authority drift: {drifted}")

    decision = _mapping(i4_3g4a_snapshot.get("decision"), label="G4A decision")
    if decision.get("i4_3g4a_preflight_closed") is not True or decision.get("i4_3g4b_production_native_execution_authorized") is not True:
        raise ReconstructionValidationError("I4.3G4B production-native execution is not authorized by G4A.")
    if decision.get("global_scalar_score_forbidden") is not True or decision.get("universal_provider_winner_assumed") is not False:
        raise ReconstructionValidationError("I4.3G4B refuses winner/global-score policy drift.")


def resolve_g4b_scenario_binding(
    *,
    i4_3g4a_snapshot: Mapping[str, Any],
    scenario_id: str,
    verify_files: bool = True,
) -> G4BScenarioBinding:
    validate_i4_3g4b_predecessor(i4_3g4a_snapshot)
    sid = str(scenario_id).strip()
    if sid not in I43G4B_SCENARIOS:
        raise ReconstructionValidationError(f"Unsupported I4.3G4B scenario: {sid}")
    raw_bindings = _sequence(i4_3g4a_snapshot.get("scenario_bindings"), label="G4A scenario_bindings")
    raw = next((item for item in raw_bindings if isinstance(item, Mapping) and item.get("scenario_id") == sid), None)
    if raw is None:
        raise ReconstructionValidationError(f"I4.3G4B scenario is absent from G4A: {sid}")

    expected_g2 = I43G4A_G2_SCENARIO_MAP[sid]
    if raw.get("predecessor_g2_scenario_id") != expected_g2:
        raise ReconstructionValidationError(f"I4.3G4B scenario/G2 predecessor drift for {sid}.")
    binding = G4BScenarioBinding(
        scenario_id=sid,
        predecessor_g2_scenario_id=expected_g2,
        source_directory=Path(str(raw.get("source_directory", ""))).expanduser().resolve(),
        source_image_count=int(raw.get("source_image_count", 0)),
        source_asset_set_sha256=str(raw.get("source_asset_set_sha256", "")).lower(),
        source_manifest_path=Path(str(raw.get("source_manifest_path", ""))).expanduser().resolve(),
        source_manifest_sha256=str(raw.get("source_manifest_sha256", "")).lower(),
        authoritative_sfm_snapshot_path=Path(str(raw.get("authoritative_sfm_snapshot_path", ""))).expanduser().resolve(),
        authoritative_sfm_snapshot_sha256=str(raw.get("authoritative_sfm_snapshot_sha256", "")).lower(),
        authoritative_sfm_frame_id=str(raw.get("authoritative_sfm_frame_id", "")).strip(),
        registered_view_count=int(raw.get("registered_view_count", 0)),
        landmark_count=int(raw.get("landmark_count", 0)),
        g2b_production_snapshot_path=Path(str(raw.get("g2b_production_snapshot_path", ""))).expanduser().resolve(),
        g2b_production_snapshot_sha256=str(raw.get("g2b_production_snapshot_sha256", "")).lower(),
    )
    if binding.source_image_count < 3 or binding.registered_view_count < 2 or binding.landmark_count <= 0:
        raise ReconstructionValidationError(f"I4.3G4B scenario binding is incomplete: {sid}")
    for value, label in (
        (binding.source_asset_set_sha256, "source asset set"),
        (binding.source_manifest_sha256, "source manifest"),
        (binding.authoritative_sfm_snapshot_sha256, "authoritative SfM snapshot"),
        (binding.g2b_production_snapshot_sha256, "G2B production snapshot"),
    ):
        if not _valid_sha256(value):
            raise ReconstructionValidationError(f"I4.3G4B {sid} {label} SHA256 is invalid.")
    if not binding.authoritative_sfm_frame_id:
        raise ReconstructionValidationError(f"I4.3G4B {sid} authoritative SfM frame ID is empty.")

    if verify_files:
        _verify_file(binding.source_manifest_path, binding.source_manifest_sha256, label=f"{sid} source manifest")
        _verify_file(binding.authoritative_sfm_snapshot_path, binding.authoritative_sfm_snapshot_sha256, label=f"{sid} SfM snapshot")
        _verify_file(binding.g2b_production_snapshot_path, binding.g2b_production_snapshot_sha256, label=f"{sid} G2B snapshot")
        current = build_source_identity(binding.source_directory, minimum_images=3)
        if current.image_count != binding.source_image_count or current.asset_set_sha256 != binding.source_asset_set_sha256:
            raise ReconstructionValidationError(f"I4.3G4B source identity drifted since G4A: {sid}")
    return binding


def validate_current_runtime_against_g4a(
    *,
    implementation: str,
    frozen_runtime: Mapping[str, Any],
    current_runtime: Mapping[str, Any],
) -> None:
    impl = str(implementation).strip()
    if impl not in I43G4B_PROVIDER_IMPLEMENTATIONS:
        raise ReconstructionValidationError(f"Unsupported I4.3G4B runtime: {impl}")
    if frozen_runtime.get("g4a_ready") is not True or not validate_runtime_probe(frozen_runtime, implementation=impl):
        raise ReconstructionValidationError(f"I4.3G4B frozen G4A runtime is not ready: {impl}")
    if not validate_runtime_probe(current_runtime, implementation=impl):
        raise ReconstructionValidationError(f"I4.3G4B current runtime is unavailable or drifted: {impl}")
    fields = _RUNTIME_IDENTITY_FIELDS[impl]
    drift = {
        key: {"g4a": frozen_runtime.get(key), "current": current_runtime.get(key)}
        for key in fields
        if frozen_runtime.get(key) != current_runtime.get(key)
    }
    if drift:
        raise ReconstructionValidationError(f"I4.3G4B runtime identity drift for {impl}: {drift}")


def validate_single_image_production_plan(plan: Mapping[str, Any], *, implementation: str) -> None:
    impl = str(implementation).strip()
    if impl not in I43G4B_PROVIDER_IMPLEMENTATIONS:
        raise ReconstructionValidationError(f"I4.3G4B unsupported provider plan: {impl}")
    if plan.get("implementation") != impl:
        raise ReconstructionValidationError(f"I4.3G4B adaptive plan implementation drift for {impl}.")
    if plan.get("adaptive_selection_evaluated") is not True:
        raise ReconstructionValidationError("I4.3G4B requires I4.3F adaptive selection to be evaluated.")
    if plan.get("permanent_global_view_cap") is not None:
        raise ReconstructionValidationError("I4.3G4B refuses a permanent global view cap in an I4.3F plan.")
    requirements = _mapping(plan.get("requirements"), label=f"{impl} plan requirements")
    if requirements.get("stateful_sequence") is not False:
        raise ReconstructionValidationError(f"I4.3G4B single-image plan cannot use stateful temporal semantics: {impl}")
    source_order = tuple(str(v) for v in plan.get("source_order_keyframe_ids", ()))
    selected = tuple(str(v) for v in plan.get("selected_keyframe_ids", ()))
    if not source_order or not selected or len(set(source_order)) != len(source_order) or len(set(selected)) != len(selected):
        raise ReconstructionValidationError(f"I4.3G4B adaptive plan has invalid view identity for {impl}.")
    if any(value not in source_order for value in selected):
        raise ReconstructionValidationError(f"I4.3G4B adaptive plan selected an unknown view for {impl}.")
    if int(plan.get("source_view_count", 0)) != len(source_order) or int(plan.get("selected_view_count", 0)) != len(selected):
        raise ReconstructionValidationError(f"I4.3G4B adaptive plan count drift for {impl}.")


def recognized_quality_rejection(*, implementation: str, error_message: str) -> bool:
    impl = str(implementation).strip()
    prefix = _QUALITY_REJECTION_PREFIX.get(impl)
    return bool(prefix and str(error_message).strip().startswith(prefix))


def validate_single_image_provider_run_evidence(manifest: Mapping[str, Any]) -> None:
    required_top = (
        "schema_version", "scenario_id", "provider_implementation", "comparison_mode", "attempt",
        "status", "source_identity", "authoritative_sfm_identity", "adaptive_plan_identity",
        "provider_provenance", "runtime_resources", "metrics", "artifacts", "authority",
    )
    missing = [key for key in required_top if key not in manifest]
    if missing:
        raise ReconstructionValidationError(f"I4.3G4B run evidence missing fields: {missing}")
    if manifest.get("scenario_id") not in I43G4B_SCENARIOS:
        raise ReconstructionValidationError("I4.3G4B run evidence scenario is not a frozen G4 scenario.")
    if manifest.get("provider_implementation") not in I43G4B_PROVIDER_IMPLEMENTATIONS:
        raise ReconstructionValidationError("I4.3G4B run evidence provider is not an authorized single-image provider.")
    if manifest.get("comparison_mode") != ComparisonMode.PRODUCTION_NATIVE.value:
        raise ReconstructionValidationError("I4.3G4B accepts production-native run evidence only.")
    status = str(manifest.get("status", "")).strip()
    if status not in set(I43G4B_QUALITY_STATUS_VALUES):
        raise ReconstructionValidationError("I4.3G4B provider quality status must be pass, degraded, or rejected.")

    metrics = _mapping(manifest.get("metrics"), label="run metrics")
    reasons = manifest.get("optional_metric_reasons", {})
    if not isinstance(reasons, Mapping):
        raise ReconstructionValidationError("I4.3G4B optional_metric_reasons must be a mapping.")
    for spec in qualification_metric_registry():
        if spec.metric_id in metrics:
            if status == "rejected" and spec.metric_id == "execution_completed":
                if metrics[spec.metric_id] is not False:
                    raise ReconstructionValidationError("Rejected I4.3G4B evidence must record execution_completed=False.")
            else:
                validate_metric_value(spec, metrics[spec.metric_id])
        elif spec.required:
            raise ReconstructionValidationError(f"Required G1 metric missing: {spec.metric_id}")
        elif not str(reasons.get(spec.metric_id, "")).strip():
            raise ReconstructionValidationError(
                f"Optional G1 metric {spec.metric_id} is missing without a persisted reason."
            )

    artifacts = _sequence(manifest.get("artifacts"), label="run artifacts")
    if not artifacts:
        raise ReconstructionValidationError("I4.3G4B run evidence requires at least one hashable artifact.")
    for item_raw in artifacts:
        item = _mapping(item_raw, label="run artifact")
        if not str(item.get("path", "")).strip() or not _valid_sha256(item.get("sha256")):
            raise ReconstructionValidationError("I4.3G4B run artifact identity is incomplete.")

    if status == "rejected":
        rejection = _mapping(manifest.get("rejection"), label="run rejection")
        if rejection.get("terminal") is not True or rejection.get("recognized_existing_quality_gate") is not True:
            raise ReconstructionValidationError("Rejected I4.3G4B evidence must bind a recognized terminal E-series quality gate.")
        reason = str(rejection.get("reason", "")).strip()
        if not reason:
            raise ReconstructionValidationError("Rejected I4.3G4B evidence requires a persisted reason.")
        if not recognized_quality_rejection(implementation=str(manifest.get("provider_implementation", "")), error_message=reason):
            raise ReconstructionValidationError("Rejected I4.3G4B evidence reason is not an authorized pre-existing E-series quality gate.")
        if rejection.get("canonical_geometry_generated") is not False:
            raise ReconstructionValidationError("Rejected I4.3G4B evidence cannot claim canonical geometry generation.")
        if rejection.get("threshold_changed") is not False:
            raise ReconstructionValidationError("Rejected I4.3G4B evidence must prove no threshold was changed to influence the outcome.")
        if rejection.get("provider_execution_reran") is not False:
            raise ReconstructionValidationError("Rejected I4.3G4B evidence must preserve the first terminal production attempt without retrying to force a pass.")
        if rejection.get("scenario_execution_continues_to_other_provider") is not True:
            raise ReconstructionValidationError("Rejected I4.3G4B evidence must allow the remaining provider matrix to continue.")

    authority = _mapping(manifest.get("authority"), label="run authority")
    if authority.get("camera_pose_authority_owner") != "pycolmap_sfm" or authority.get("permanent_global_view_cap") is not None:
        raise ReconstructionValidationError("I4.3G4B run evidence changed camera authority/view-cap policy.")
    drifted = [key for key in _REQUIRED_AUTHORITY_FALSE if authority.get(key) is not False]
    if drifted:
        raise ReconstructionValidationError(f"I4.3G4B run evidence authority drift: {drifted}")


def build_g4b_scenario_snapshot(
    *,
    binding: G4BScenarioBinding,
    g4a_snapshot_path: Path,
    g4a_snapshot_sha256: str,
    provider_evidence: Mapping[str, Mapping[str, Any]],
    provider_artifacts: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    if not isinstance(binding, G4BScenarioBinding):
        raise TypeError("binding must be G4BScenarioBinding.")
    if set(provider_evidence) != set(I43G4B_PROVIDER_IMPLEMENTATIONS):
        raise ReconstructionValidationError("I4.3G4B scenario closure requires terminal evidence from all six providers.")
    if set(provider_artifacts) != set(I43G4B_PROVIDER_IMPLEMENTATIONS):
        raise ReconstructionValidationError("I4.3G4B scenario closure requires all six provider evidence artifacts.")
    statuses: dict[str, str] = {}
    for implementation in I43G4B_PROVIDER_IMPLEMENTATIONS:
        evidence = provider_evidence[implementation]
        validate_single_image_provider_run_evidence(evidence)
        statuses[implementation] = str(evidence["status"])
    rejected = [name for name, status in statuses.items() if status == "rejected"]
    degraded = [name for name, status in statuses.items() if status == "degraded"]
    return {
        "schema_version": I43G4B_SCHEMA_VERSION,
        "phase": I43G4B_PHASE,
        "status": "production_native_complete_with_provider_rejection" if rejected else "production_native_complete",
        "pass": True,
        "scenario": binding.as_dict(),
        "predecessor_g4a": {
            "path": str(Path(g4a_snapshot_path).expanduser().resolve()),
            "sha256": str(g4a_snapshot_sha256).lower(),
        },
        "providers": {name: dict(provider_artifacts[name]) for name in I43G4B_PROVIDER_IMPLEMENTATIONS},
        "production_statuses": statuses,
        "accounting": {
            "required_provider_count": 6,
            "terminal_provider_count": 6,
            "all_required_providers_accounted": True,
            "provider_quality_rejection_present": bool(rejected),
            "rejected_providers": rejected,
            "degraded_providers": degraded,
        },
        "manual_review": {
            "required": True,
            "dimensions": list(I43G4B_MANUAL_REVIEW_DIMENSIONS),
            "matched_view_diagnostic_started": False,
            "verdict_recorded": False,
            "deferred_to": "I4.3G4C",
            "production_rejection_cannot_be_overridden": True,
        },
        "authority": {
            "camera_pose_authority_owner": "pycolmap_sfm",
            "same_authoritative_sfm_shared_by_all_six_providers": True,
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
        "decision": {
            "production_native_scenario_complete": True,
            "matched_view_diagnostic_authorized_for_g4c": True,
            "scenario_verdict_pending_g4c_manual_review": True,
            "provider_selected": False,
            "provider_promoted": False,
            "universal_provider_winner_assumed": False,
            "global_scalar_score_forbidden": True,
        },
    }


__all__ = [
    "I43G4B_SCHEMA_VERSION",
    "I43G4B_PHASE",
    "I43G4B_SCENARIOS",
    "I43G4B_PROVIDER_IMPLEMENTATIONS",
    "I43G4B_QUALITY_STATUS_VALUES",
    "I43G4B_INFRASTRUCTURE_STATUS_VALUES",
    "I43G4B_MANUAL_REVIEW_DIMENSIONS",
    "G4BScenarioBinding",
    "sha256_file",
    "validate_i4_3g4b_predecessor",
    "resolve_g4b_scenario_binding",
    "validate_current_runtime_against_g4a",
    "validate_single_image_production_plan",
    "recognized_quality_rejection",
    "validate_single_image_provider_run_evidence",
    "build_g4b_scenario_snapshot",
]
