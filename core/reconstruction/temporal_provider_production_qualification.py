"""I4.3G3B production-native temporal-provider qualification contracts.

G3B consumes the frozen G3A source/runtime binding and authorizes one real
production-native temporal scenario at a time.  The production runner must use
one authoritative PyCOLMAP SfM per scenario, derive independent I4.3F plans for
RollingDepth and Video Depth Anything from that same SfM, preserve temporal
source order, and persist G1-compatible run evidence.

Matched-view diagnostics and manual A/B review remain deferred to I4.3G3C.  No
provider is selected/promoted and no Auto, hybrid, physical metric-scale,
measurement, metrology, or mesh authority is introduced here.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from core.reconstruction.cross_provider_qualification import (
    ComparisonMode,
    qualification_metric_registry,
    validate_metric_value,
)
from core.reconstruction.errors import ReconstructionValidationError
from core.reconstruction.temporal_provider_qualification import (
    I43G3A_PROVIDER_IMPLEMENTATIONS,
    I43G3A_SCENARIOS,
    build_temporal_video_identity,
)

I43G3B_SCHEMA_VERSION = 1
I43G3B_PHASE = "I4.3G3B"
I43G3B_SCENARIOS = I43G3A_SCENARIOS
I43G3B_PROVIDER_IMPLEMENTATIONS = I43G3A_PROVIDER_IMPLEMENTATIONS

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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json_object(path: Path, *, label: str) -> dict[str, Any]:
    source = path.expanduser().resolve()
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ReconstructionValidationError(f"Unable to read {label}: {source}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ReconstructionValidationError(f"{label} must contain a JSON object: {source}")
    return payload


def verify_snapshot_sha256(path: Path, expected_sha256: str, *, label: str) -> str:
    expected = str(expected_sha256).strip().lower()
    if len(expected) != 64 or any(ch not in "0123456789abcdef" for ch in expected):
        raise ReconstructionValidationError(f"{label} expected SHA256 is invalid.")
    source = path.expanduser().resolve()
    if not source.is_file():
        raise ReconstructionValidationError(f"{label} is missing: {source}")
    actual = sha256_file(source)
    if actual != expected:
        raise ReconstructionValidationError(
            f"{label} SHA256 mismatch: expected={expected} actual={actual} path={source}"
        )
    return actual


def _mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ReconstructionValidationError(f"I4.3G3B requires mapping section: {label}")
    return value


def validate_g3b_predecessor(*, i4_3g3a_snapshot: Mapping[str, Any]) -> None:
    """Fail closed unless the final G3A snapshot explicitly authorizes G3B."""

    if (
        i4_3g3a_snapshot.get("phase") != "I4.3G3A"
        or i4_3g3a_snapshot.get("status") != "ready"
        or i4_3g3a_snapshot.get("pass") is not True
    ):
        raise ReconstructionValidationError("I4.3G3B requires a ready passing I4.3G3A snapshot.")

    summary = _mapping(i4_3g3a_snapshot.get("binding_summary"), label="G3A binding_summary")
    runtime = _mapping(i4_3g3a_snapshot.get("runtime"), label="G3A runtime")
    production = _mapping(i4_3g3a_snapshot.get("production_native_contract"), label="G3A production_native_contract")
    matched = _mapping(i4_3g3a_snapshot.get("matched_view_contract"), label="G3A matched_view_contract")
    scope = _mapping(i4_3g3a_snapshot.get("scope"), label="G3A scope")
    authority = _mapping(i4_3g3a_snapshot.get("authority"), label="G3A authority")
    decision = _mapping(i4_3g3a_snapshot.get("decision"), label="G3A decision")

    if summary.get("required_scenario_count") != 3 or summary.get("bound_scenario_count") != 3:
        raise ReconstructionValidationError("I4.3G3B requires all three frozen G3A scenario bindings.")
    if summary.get("all_g3_sources_bound") is not True or summary.get("semantic_binding_inferred") is not False:
        raise ReconstructionValidationError("I4.3G3B requires explicit, non-inferred G3A source bindings.")
    if runtime.get("video_depth_anything_ready") is not True or runtime.get("rolling_depth_ready") is not True:
        raise ReconstructionValidationError("I4.3G3B requires both G3A temporal runtimes to be ready.")
    if decision.get("i4_3g3b_production_native_execution_authorized") is not True:
        raise ReconstructionValidationError("I4.3G3B production-native execution is not authorized by G3A.")
    if decision.get("matched_view_diagnostic_deferred_until_production_native_evidence") is not True:
        raise ReconstructionValidationError("I4.3G3B requires matched-view work to remain deferred until G3C.")
    if decision.get("scenario_specific_only") is not True or decision.get("global_scalar_score_forbidden") is not True:
        raise ReconstructionValidationError("I4.3G3B refuses G3A scenario/score policy drift.")
    if decision.get("universal_provider_winner_assumed") is not False:
        raise ReconstructionValidationError("I4.3G3B refuses a preselected universal temporal-provider winner.")

    expected_production = {
        "comparison_mode": ComparisonMode.PRODUCTION_NATIVE.value,
        "primary_operating_envelope_evidence": True,
        "provider_specific_i4_3f_adaptive_plans_required": True,
        "source_order_preserved": True,
        "contiguous_subsequence_required": False,
        "same_source_identity_within_scenario": True,
        "same_authoritative_sfm_identity_within_scenario_required": True,
    }
    for key, expected in expected_production.items():
        if production.get(key) != expected:
            raise ReconstructionValidationError(f"I4.3G3B G3A production-native contract drift: {key}")

    if matched.get("comparison_mode") != ComparisonMode.MATCHED_VIEW_DIAGNOSTIC.value:
        raise ReconstructionValidationError("I4.3G3B matched-view comparison-mode contract drifted.")
    for key in (
        "secondary_causal_diagnostic_only",
        "common_contiguous_source_ordered_subsequence_required",
        "provider_specific_temporal_semantics_must_be_preserved",
        "same_source_identity_within_scenario",
        "same_authoritative_sfm_identity_within_scenario_required",
        "same_manual_review_camera_and_framing_required",
        "independent_provider_auto_fit_forbidden",
    ):
        if matched.get(key) is not True:
            raise ReconstructionValidationError(f"I4.3G3B G3A matched-view contract drift: {key}")

    for key in (
        "real_provider_execution_started",
        "authoritative_sfm_started",
        "video_depth_anything_started",
        "rolling_depth_started",
        "provider_quality_results_consumed",
        "matched_view_diagnostic_started",
        "manual_provider_ranking_performed",
    ):
        if scope.get(key) is not False:
            raise ReconstructionValidationError(f"I4.3G3B requires execution-free G3A scope: {key}=False")

    if authority.get("camera_pose_authority_owner") != "pycolmap_sfm":
        raise ReconstructionValidationError("I4.3G3B requires PyCOLMAP camera-pose authority.")
    if authority.get("permanent_global_view_cap") is not None:
        raise ReconstructionValidationError("I4.3G3B refuses a permanent global view cap.")
    drifted = [key for key in _REQUIRED_AUTHORITY_FALSE if authority.get(key) is not False]
    if drifted:
        raise ReconstructionValidationError(f"I4.3G3B G3A authority drift: {drifted}")

    bindings = i4_3g3a_snapshot.get("scenario_bindings")
    if not isinstance(bindings, Sequence) or isinstance(bindings, (str, bytes)) or len(bindings) != 3:
        raise ReconstructionValidationError("I4.3G3B requires exactly three G3A scenario bindings.")
    by_id = {str(item.get("scenario_id")): item for item in bindings if isinstance(item, Mapping)}
    if tuple(sid for sid in I43G3B_SCENARIOS if sid in by_id) != I43G3B_SCENARIOS or len(by_id) != 3:
        raise ReconstructionValidationError("I4.3G3B G3A scenario binding coverage drifted.")


@dataclass(frozen=True, slots=True)
class G3BScenarioBinding:
    scenario_id: str
    source_path: Path
    source_file_sha256: str
    source_manifest_path: Path
    source_manifest_sha256: str
    file_size_bytes: int
    reader_fingerprint: str
    expected_frame_rate_hz: float
    reported_frame_count: int
    semantic_binding_method: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "scenario_id": self.scenario_id,
            "source_path": str(self.source_path),
            "source_file_sha256": self.source_file_sha256,
            "asset_set_sha256": self.source_file_sha256,
            "source_manifest_path": str(self.source_manifest_path),
            "source_manifest_sha256": self.source_manifest_sha256,
            "file_size_bytes": self.file_size_bytes,
            "reader_fingerprint": self.reader_fingerprint,
            "expected_frame_rate_hz": self.expected_frame_rate_hz,
            "reported_frame_count": self.reported_frame_count,
            "semantic_binding_method": self.semantic_binding_method,
            "temporal_order_semantics": "recorded_video_decode_sequence_index",
        }


def resolve_g3b_scenario_binding(
    *,
    i4_3g3a_snapshot: Mapping[str, Any],
    scenario_id: str,
    verify_files: bool = True,
) -> G3BScenarioBinding:
    validate_g3b_predecessor(i4_3g3a_snapshot=i4_3g3a_snapshot)
    sid = str(scenario_id).strip()
    if sid not in I43G3B_SCENARIOS:
        raise ReconstructionValidationError(f"Unsupported I4.3G3B scenario: {sid}")

    bindings = i4_3g3a_snapshot.get("scenario_bindings")
    raw = next((item for item in bindings if isinstance(item, Mapping) and item.get("scenario_id") == sid), None)
    if raw is None or raw.get("status") != "bound" or raw.get("semantic_binding_inferred") is not False:
        raise ReconstructionValidationError(f"I4.3G3B scenario is not explicitly bound by G3A: {sid}")
    temporal = _mapping(raw.get("temporal_contract"), label=f"{sid} temporal_contract")
    for key in (
        "ordered_source_required",
        "strictly_increasing_canonical_timestamps_required_after_ingestion",
        "production_native_source_order_preserved",
        "matched_view_contiguous_source_order_required",
    ):
        if temporal.get(key) is not True:
            raise ReconstructionValidationError(f"I4.3G3B temporal binding drift for {sid}: {key}")

    source = _mapping(raw.get("source_identity"), label=f"{sid} source_identity")
    path = Path(str(source.get("source_path", ""))).expanduser().resolve()
    manifest_path = Path(str(source.get("manifest_path", ""))).expanduser().resolve()
    source_sha = str(source.get("source_file_sha256", "")).strip().lower()
    asset_sha = str(source.get("asset_set_sha256", "")).strip().lower()
    manifest_sha = str(source.get("manifest_sha256", "")).strip().lower()
    fingerprint = str(source.get("reader_fingerprint", "")).strip().lower()
    if source_sha != asset_sha or len(source_sha) != 64 or len(manifest_sha) != 64 or len(fingerprint) != 64:
        raise ReconstructionValidationError(f"I4.3G3B source identity is incomplete for {sid}.")

    binding = G3BScenarioBinding(
        scenario_id=sid,
        source_path=path,
        source_file_sha256=source_sha,
        source_manifest_path=manifest_path,
        source_manifest_sha256=manifest_sha,
        file_size_bytes=int(source.get("file_size_bytes", 0)),
        reader_fingerprint=fingerprint,
        expected_frame_rate_hz=float(source.get("expected_frame_rate_hz", 0.0)),
        reported_frame_count=int(source.get("reported_frame_count", 0)),
        semantic_binding_method=str(raw.get("semantic_binding_method", "")).strip(),
    )
    if binding.file_size_bytes <= 0 or binding.expected_frame_rate_hz <= 0.0:
        raise ReconstructionValidationError(f"I4.3G3B invalid source size/rate for {sid}.")
    if not binding.semantic_binding_method:
        raise ReconstructionValidationError(f"I4.3G3B missing semantic binding method for {sid}.")

    if verify_files:
        if not manifest_path.is_file() or sha256_file(manifest_path) != manifest_sha:
            raise ReconstructionValidationError(f"I4.3G3B source manifest changed or is missing for {sid}.")
        actual = build_temporal_video_identity(path)
        if (
            actual.source_file_sha256 != source_sha
            or actual.asset_set_sha256 != asset_sha
            or actual.file_size_bytes != binding.file_size_bytes
            or actual.reader_fingerprint != fingerprint
            or actual.reported_frame_count != binding.reported_frame_count
            or abs(actual.expected_frame_rate_hz - binding.expected_frame_rate_hz) > 1e-6
        ):
            raise ReconstructionValidationError(f"I4.3G3B source identity changed for {sid}.")
    return binding


def _vda_ready(runtime: Mapping[str, Any]) -> bool:
    return bool(
        runtime.get("available") is True
        and runtime.get("source_ready") is True
        and runtime.get("checkpoint_ready") is True
        and runtime.get("adapter_importable") is True
        and runtime.get("upstream_importable") is True
        and runtime.get("torch_available") is True
        and runtime.get("cuda_available") is True
        and str(runtime.get("source_revision", "")).strip()
    )


def _rolling_ready(runtime: Mapping[str, Any]) -> bool:
    return bool(
        runtime.get("available") is True
        and runtime.get("source_ready") is True
        and runtime.get("checkpoint_ready") is True
        and runtime.get("runtime_lock_valid") is True
        and runtime.get("upstream_importable") is True
        and runtime.get("modified_diffusers_ready") is True
        and runtime.get("torch_available") is True
        and runtime.get("ffmpeg_available") is True
        and runtime.get("cuda_available") is True
        and str(runtime.get("source_revision", "")).strip()
    )


def _sfm_ready(runtime: Mapping[str, Any]) -> bool:
    return bool(
        runtime.get("available") is True
        and runtime.get("backend_ready") is True
        and runtime.get("required_apis_available") is True
        and runtime.get("recommended_version") is True
        and runtime.get("cuda_ready") is True
        and int(runtime.get("cuda_devices", 0)) > 0
    )


def _runtime_identity_matches(bound: Mapping[str, Any], current: Mapping[str, Any], *, implementation: str) -> bool:
    if str(bound.get("source_revision", "")).strip().lower() != str(current.get("source_revision", "")).strip().lower():
        return False
    for key in ("checkpoint_path",):
        bound_value = str(bound.get(key, "")).strip()
        current_value = str(current.get(key, "")).strip()
        if bound_value and current_value and Path(bound_value).expanduser().resolve() != Path(current_value).expanduser().resolve():
            return False
    if implementation == "rolling_depth":
        for key in ("repository_root", "runtime_lock_path"):
            bound_value = str(bound.get(key, "")).strip()
            current_value = str(current.get(key, "")).strip()
            if bound_value and current_value and Path(bound_value).expanduser().resolve() != Path(current_value).expanduser().resolve():
                return False
    return True


def build_g3b_runtime_preflight_snapshot(
    *,
    i4_3g3a_snapshot: Mapping[str, Any],
    scenario_id: str,
    sfm_runtime: Mapping[str, Any],
    video_depth_anything_runtime: Mapping[str, Any],
    rolling_depth_runtime: Mapping[str, Any],
    verify_files: bool = True,
) -> dict[str, Any]:
    binding = resolve_g3b_scenario_binding(
        i4_3g3a_snapshot=i4_3g3a_snapshot,
        scenario_id=scenario_id,
        verify_files=verify_files,
    )
    bound_runtime = _mapping(i4_3g3a_snapshot.get("runtime"), label="G3A runtime")
    bound_vda = _mapping(bound_runtime.get("video_depth_anything"), label="G3A VDA runtime")
    bound_rolling = _mapping(bound_runtime.get("rolling_depth"), label="G3A RollingDepth runtime")
    sfm = dict(sfm_runtime)
    vda = dict(video_depth_anything_runtime)
    rolling = dict(rolling_depth_runtime)
    sfm_ok = _sfm_ready(sfm)
    vda_ok = _vda_ready(vda) and _runtime_identity_matches(bound_vda, vda, implementation="video_depth_anything")
    rolling_ok = _rolling_ready(rolling) and _runtime_identity_matches(bound_rolling, rolling, implementation="rolling_depth")
    ready = sfm_ok and vda_ok and rolling_ok

    return {
        "schema_version": I43G3B_SCHEMA_VERSION,
        "phase": I43G3B_PHASE,
        "status": "ready" if ready else "runtime_unavailable_or_drifted",
        "pass": ready,
        "scenario": binding.as_dict(),
        "comparison_mode": ComparisonMode.PRODUCTION_NATIVE.value,
        "runtime": {
            "pycolmap_sfm": sfm,
            "video_depth_anything": vda,
            "rolling_depth": rolling,
            "sfm_ready": sfm_ok,
            "video_depth_anything_ready_and_bound": vda_ok,
            "rolling_depth_ready_and_bound": rolling_ok,
        },
        "scope": {
            "runtime_preflight_only": True,
            "source_identity_verified": verify_files,
            "authoritative_sfm_started": False,
            "video_depth_anything_started": False,
            "rolling_depth_started": False,
            "matched_view_diagnostic_started": False,
            "manual_provider_ranking_performed": False,
        },
        "production_native_contract": {
            "one_authoritative_pycolmap_sfm_per_scenario": True,
            "same_authoritative_sfm_shared_by_both_providers": True,
            "provider_specific_i4_3f_plans": True,
            "temporal_source_order_preserved": True,
            "matched_view_contiguous_subsequence_deferred_to_g3c": True,
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
        "decision": {
            "production_native_real_execution_authorized": ready,
            "scenario_specific_only": True,
            "matched_view_diagnostic_started": False,
            "matched_view_diagnostic_deferred_to_g3c": True,
            "manual_verdict_deferred_to_g3c": True,
            "provider_selected": False,
            "provider_promoted": False,
            "universal_provider_winner_assumed": False,
            "global_scalar_score_forbidden": True,
        },
        "next": (
            f"Run production-native {binding.scenario_id} using one authoritative PyCOLMAP SfM and separate I4.3F temporal-provider plans."
            if ready
            else "Repair only the unavailable/drifted runtime or source binding; do not weaken G3A/F/authority gates."
        ),
    }


def validate_temporal_production_plan(plan: Mapping[str, Any], *, implementation: str) -> None:
    impl = str(implementation).strip()
    if impl not in I43G3B_PROVIDER_IMPLEMENTATIONS:
        raise ReconstructionValidationError(f"I4.3G3B unsupported temporal provider plan: {impl}")
    if plan.get("implementation") != impl:
        raise ReconstructionValidationError(f"I4.3G3B adaptive plan implementation drift for {impl}.")
    if plan.get("adaptive_selection_evaluated") is not True:
        raise ReconstructionValidationError("I4.3G3B requires I4.3F adaptive selection to be evaluated.")
    if plan.get("permanent_global_view_cap") is not None:
        raise ReconstructionValidationError("I4.3G3B refuses a permanent view cap in an I4.3F plan.")
    requirements = _mapping(plan.get("requirements"), label=f"{impl} plan requirements")
    if requirements.get("stateful_sequence") is not True or requirements.get("source_order_required") is not True:
        raise ReconstructionValidationError(f"I4.3G3B requires stateful ordered temporal plan semantics for {impl}.")
    source_order = tuple(str(v) for v in plan.get("source_order_keyframe_ids", ()))
    selected = tuple(str(v) for v in plan.get("selected_keyframe_ids", ()))
    if not source_order or not selected:
        raise ReconstructionValidationError(f"I4.3G3B temporal plan is empty for {impl}.")
    index = {key: i for i, key in enumerate(source_order)}
    if any(key not in index for key in selected) or tuple(sorted(selected, key=index.__getitem__)) != selected:
        raise ReconstructionValidationError(f"I4.3G3B temporal plan does not preserve source order for {impl}.")
    if int(plan.get("selected_view_count", 0)) != len(selected):
        raise ReconstructionValidationError(f"I4.3G3B selected-view count drift for {impl}.")



def validate_temporal_provider_summary_authority(summary: Mapping[str, Any], *, implementation: str) -> None:
    """Require the frozen D1/D2 summary to prove SfM/authority preservation.

    G3B does not infer these gates from a successful process exit. The existing
    provider summary must itself attest that authoritative SfM/registered views
    were unchanged and that no forbidden provider authority was promoted.
    """

    impl = str(implementation).strip()
    if impl not in I43G3B_PROVIDER_IMPLEMENTATIONS:
        raise ReconstructionValidationError(f"I4.3G3B unsupported temporal provider summary: {impl}")
    integrity = _mapping(summary.get("integrity"), label=f"{impl} summary integrity")
    if integrity.get("authoritative_sfm_unchanged") is not True:
        raise ReconstructionValidationError(f"I4.3G3B {impl} summary does not prove authoritative SfM preservation.")
    if integrity.get("registered_images_unchanged") is not True:
        raise ReconstructionValidationError(f"I4.3G3B {impl} summary does not prove registered-image preservation.")
    if not str(integrity.get("authoritative_sfm_digest", "")).strip():
        raise ReconstructionValidationError(f"I4.3G3B {impl} summary is missing its authoritative SfM digest.")
    if not str(integrity.get("registered_image_sha256", "")).strip():
        raise ReconstructionValidationError(f"I4.3G3B {impl} summary is missing its registered-image digest.")
    if integrity.get("canonical_pose_layer") != "offline_sfm_bundle_adjusted":
        raise ReconstructionValidationError(f"I4.3G3B {impl} canonical-pose layer drifted.")

    authority = _mapping(summary.get("authority"), label=f"{impl} summary authority")
    if authority.get("camera_pose_authority_owner") != "pycolmap_sfm":
        raise ReconstructionValidationError(f"I4.3G3B {impl} summary changed camera-pose authority.")
    for key in (
        "metric_scale_claimed",
        "measurement_authoritative",
        "certified_metrology",
        "surface_mesh_generated",
        "provider_promotion_performed",
    ):
        if authority.get(key) is not False:
            raise ReconstructionValidationError(f"I4.3G3B {impl} summary authority drift: {key}")
    if impl == "rolling_depth":
        for key in (
            "rolling_depth_pose_authority_promoted",
            "sequence_affine_establishes_physical_metric_scale",
            "per_frame_alignment_performed",
        ):
            if authority.get(key) is not False:
                raise ReconstructionValidationError(f"I4.3G3B RollingDepth summary authority drift: {key}")
    else:
        for key in (
            "estimated_model_intrinsics_consumed_for_canonical_projection",
            "model_metric_depth_promoted_to_canonical_metric_scale",
        ):
            if authority.get(key) is not False:
                raise ReconstructionValidationError(f"I4.3G3B VDA summary authority drift: {key}")

def validate_temporal_provider_run_evidence(manifest: Mapping[str, Any]) -> None:
    required_top = (
        "schema_version", "scenario_id", "provider_implementation", "comparison_mode", "attempt",
        "status", "source_identity", "authoritative_sfm_identity", "adaptive_plan_identity",
        "provider_provenance", "runtime_resources", "metrics", "artifacts", "authority",
    )
    missing = [key for key in required_top if key not in manifest]
    if missing:
        raise ReconstructionValidationError(f"I4.3G3B run evidence missing fields: {missing}")
    if manifest.get("scenario_id") not in I43G3B_SCENARIOS:
        raise ReconstructionValidationError("I4.3G3B run evidence scenario is not a frozen G3 scenario.")
    if manifest.get("provider_implementation") not in I43G3B_PROVIDER_IMPLEMENTATIONS:
        raise ReconstructionValidationError("I4.3G3B run evidence provider is not RollingDepth/VDA.")
    if manifest.get("comparison_mode") != ComparisonMode.PRODUCTION_NATIVE.value:
        raise ReconstructionValidationError("I4.3G3B accepts production-native run evidence only.")
    status = str(manifest.get("status", "")).strip()
    if status not in {"pass", "degraded", "rejected"}:
        raise ReconstructionValidationError("I4.3G3B run evidence must be pass, degraded, or rejected.")

    metrics = _mapping(manifest.get("metrics"), label="run metrics")
    reasons = manifest.get("optional_metric_reasons", {})
    if not isinstance(reasons, Mapping):
        raise ReconstructionValidationError("I4.3G3B optional_metric_reasons must be a mapping.")
    for spec in qualification_metric_registry():
        if spec.metric_id in metrics:
            if status == "rejected" and spec.metric_id == "execution_completed":
                if metrics[spec.metric_id] is not False:
                    raise ReconstructionValidationError(
                        "Rejected I4.3G3B provider evidence must record execution_completed=False."
                    )
            else:
                validate_metric_value(spec, metrics[spec.metric_id])
        elif spec.required:
            raise ReconstructionValidationError(f"Required G1 metric missing: {spec.metric_id}")
        elif not str(reasons.get(spec.metric_id, "")).strip():
            raise ReconstructionValidationError(
                f"Optional G1 metric {spec.metric_id} is missing without a persisted reason."
            )

    if status == "rejected":
        rejection = _mapping(manifest.get("rejection"), label="run rejection")
        if rejection.get("terminal") is not True:
            raise ReconstructionValidationError("Rejected I4.3G3B evidence must be a terminal provider outcome.")
        if not str(rejection.get("reason", "")).strip():
            raise ReconstructionValidationError("Rejected I4.3G3B evidence requires a persisted reason.")
        if rejection.get("canonical_geometry_generated") is not False:
            raise ReconstructionValidationError("Rejected I4.3G3B evidence cannot claim canonical geometry generation.")

    authority = _mapping(manifest.get("authority"), label="run authority")
    if authority.get("camera_pose_authority_owner") != "pycolmap_sfm":
        raise ReconstructionValidationError("I4.3G3B run evidence changed camera-pose authority.")
    if authority.get("permanent_global_view_cap") is not None:
        raise ReconstructionValidationError("I4.3G3B run evidence introduced a permanent global view cap.")
    drifted = [key for key in _REQUIRED_AUTHORITY_FALSE if authority.get(key) is not False]
    if drifted:
        raise ReconstructionValidationError(f"I4.3G3B run evidence authority drift: {drifted}")

    temporal = _mapping(manifest.get("temporal_sequence"), label="run temporal_sequence")
    if temporal.get("source_order_preserved") is not True or temporal.get("provider_specific_temporal_semantics_preserved") is not True:
        raise ReconstructionValidationError("I4.3G3B run evidence did not preserve temporal provider semantics.")
    if temporal.get("matched_view_contiguous_subsequence_used") is not False:
        raise ReconstructionValidationError("I4.3G3B production-native evidence must not masquerade as G3C matched-view evidence.")


__all__ = [
    "I43G3B_SCHEMA_VERSION",
    "I43G3B_PHASE",
    "I43G3B_SCENARIOS",
    "I43G3B_PROVIDER_IMPLEMENTATIONS",
    "G3BScenarioBinding",
    "sha256_file",
    "read_json_object",
    "verify_snapshot_sha256",
    "validate_g3b_predecessor",
    "resolve_g3b_scenario_binding",
    "build_g3b_runtime_preflight_snapshot",
    "validate_temporal_production_plan",
    "validate_temporal_provider_summary_authority",
    "validate_temporal_provider_run_evidence",
]
