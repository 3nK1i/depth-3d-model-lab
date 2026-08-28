"""I4.3G2B controlled production MVS-vs-DA3 comparison contracts.

G2B consumes the frozen G1 methodology and the closed G2A source-binding
snapshot.  It may authorize *explicit* production-native MVS/DA3 executions,
but it never selects a universal winner, implements Auto, authorizes hybrid
fusion, or changes PyCOLMAP camera/frame authority.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from math import isfinite
from pathlib import Path
from typing import Any, Mapping, Sequence

from core.reconstruction.cross_provider_qualification import (
    ComparisonMode,
    qualification_metric_registry,
    validate_metric_value,
)
from core.reconstruction.errors import ReconstructionValidationError
from core.reconstruction.mvs_da3_qualification import build_source_identity

I43G2B_SCHEMA_VERSION = 1
I43G2B_PHASE = "I4.3G2B"
I43G2B_FRESH_SCENARIOS = (
    "g2_complete_room_mvs_da3",
    "g2_temple_ring_mvs_da3",
    "g2_low_texture_mvs_da3",
    "g2_weak_overlap_mvs_da3",
)
I43G2B_HISTORICAL_SCENARIO = "g2_house_mvs_da3_historical"
_REQUIRED_PROVIDER_IMPLEMENTATIONS = ("colmap_mvs", "da3")


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
        raise ReconstructionValidationError(f"I4.3G2B requires mapping section: {label}")
    return value


def validate_g2b_predecessors(
    *,
    i4_3g1_snapshot: Mapping[str, Any],
    i4_3g2a_snapshot: Mapping[str, Any],
) -> None:
    """Fail closed unless G1 and the final G2A authorization permit G2B."""

    if (
        i4_3g1_snapshot.get("phase") != "I4.3G1"
        or i4_3g1_snapshot.get("status") != "pass"
        or i4_3g1_snapshot.get("pass") is not True
    ):
        raise ReconstructionValidationError("I4.3G2B requires a passing frozen I4.3G1 snapshot.")
    g1_decision = _mapping(i4_3g1_snapshot.get("decision"), label="G1 decision")
    g1_authority = _mapping(i4_3g1_snapshot.get("authority"), label="G1 authority")
    if g1_decision.get("i4_3g2_mvs_da3_qualification_authorized") is not True:
        raise ReconstructionValidationError("I4.3G2B is not authorized by G1.")
    if g1_decision.get("global_scalar_score_forbidden") is not True:
        raise ReconstructionValidationError("I4.3G2B refuses G1 global-score policy drift.")
    if g1_decision.get("universal_provider_winner_assumed") is not False:
        raise ReconstructionValidationError("I4.3G2B refuses a preselected universal provider winner.")

    scenarios = i4_3g1_snapshot.get("scenario_matrix")
    if not isinstance(scenarios, Sequence) or isinstance(scenarios, (str, bytes)):
        raise ReconstructionValidationError("I4.3G2B requires the frozen G1 scenario matrix.")
    g2 = [item for item in scenarios if isinstance(item, Mapping) and item.get("substep") == "I4.3G2"]
    if len(g2) != 5:
        raise ReconstructionValidationError("I4.3G2B requires exactly five frozen G2 scenarios.")
    for item in g2:
        providers = tuple(item.get("provider_implementations", ()))
        if providers != _REQUIRED_PROVIDER_IMPLEMENTATIONS:
            raise ReconstructionValidationError(
                f"I4.3G2B provider-set drift for {item.get('scenario_id')}: {providers}"
            )

    if (
        i4_3g2a_snapshot.get("phase") != "I4.3G2A"
        or i4_3g2a_snapshot.get("pass") is not True
    ):
        raise ReconstructionValidationError("I4.3G2B requires a passing final I4.3G2A snapshot.")
    summary = _mapping(i4_3g2a_snapshot.get("binding_summary"), label="G2A binding_summary")
    decision = _mapping(i4_3g2a_snapshot.get("decision"), label="G2A decision")
    authority = _mapping(i4_3g2a_snapshot.get("authority"), label="G2A authority")
    if summary.get("fresh_scenario_bindings_ready") != 4:
        raise ReconstructionValidationError("I4.3G2B requires all four fresh G2 source bindings.")
    if summary.get("historical_binding_verified") is not True:
        raise ReconstructionValidationError("I4.3G2B requires verified House historical evidence.")
    if summary.get("all_g2_sources_bound") is not True:
        raise ReconstructionValidationError("I4.3G2B requires all G2 sources to be bound.")
    if list(summary.get("missing_explicit_fresh_bindings", ())) != []:
        raise ReconstructionValidationError("I4.3G2B refuses missing fresh G2 source bindings.")
    if decision.get("i4_3g2b_real_execution_authorized") is not True:
        raise ReconstructionValidationError("I4.3G2B real execution is not authorized by G2A.")

    for label, section in (("G1", g1_authority), ("G2A", authority)):
        if section.get("camera_pose_authority_owner") != "pycolmap_sfm":
            raise ReconstructionValidationError(f"I4.3G2B requires PyCOLMAP camera authority in {label}.")
        if section.get("permanent_global_view_cap") is not None:
            raise ReconstructionValidationError(f"I4.3G2B refuses a permanent global view cap in {label}.")
        forbidden = (
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
        drifted = [key for key in forbidden if bool(section.get(key, False))]
        if drifted:
            raise ReconstructionValidationError(f"I4.3G2B authority drift in {label}: {drifted}")


@dataclass(frozen=True, slots=True)
class G2BScenarioBinding:
    scenario_id: str
    source_directory: Path
    source_asset_set_sha256: str
    source_image_count: int
    binding_method: str
    comparison_modes: tuple[str, ...]
    matching_strategy: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "scenario_id": self.scenario_id,
            "source_directory": str(self.source_directory),
            "source_asset_set_sha256": self.source_asset_set_sha256,
            "source_image_count": self.source_image_count,
            "binding_method": self.binding_method,
            "comparison_modes": list(self.comparison_modes),
            "matching_strategy": self.matching_strategy,
        }


def resolve_g2b_scenario_binding(
    *,
    i4_3g1_snapshot: Mapping[str, Any],
    i4_3g2a_snapshot: Mapping[str, Any],
    scenario_id: str,
    verify_files: bool = True,
) -> G2BScenarioBinding:
    validate_g2b_predecessors(
        i4_3g1_snapshot=i4_3g1_snapshot,
        i4_3g2a_snapshot=i4_3g2a_snapshot,
    )
    sid = str(scenario_id).strip()
    if sid == I43G2B_HISTORICAL_SCENARIO:
        raise ReconstructionValidationError(
            "House is historical_read_only in G2; G2B must not execute a new House provider run."
        )
    if sid not in I43G2B_FRESH_SCENARIOS:
        raise ReconstructionValidationError(f"Unsupported I4.3G2B fresh scenario: {sid}")

    bindings = i4_3g2a_snapshot.get("scenario_bindings")
    if not isinstance(bindings, Sequence) or isinstance(bindings, (str, bytes)):
        raise ReconstructionValidationError("I4.3G2B requires G2A scenario_bindings.")
    raw = next((item for item in bindings if isinstance(item, Mapping) and item.get("scenario_id") == sid), None)
    if raw is None or raw.get("status") != "bound" or raw.get("ready_for_real_execution") is not True:
        raise ReconstructionValidationError(f"I4.3G2B scenario is not bound/ready: {sid}")
    source = _mapping(raw.get("source_identity"), label=f"{sid} source_identity")
    directory = Path(str(source.get("directory", ""))).expanduser().resolve()
    image_count = int(source.get("image_count", 0))
    asset_sha = str(source.get("asset_set_sha256", "")).strip().lower()
    if image_count < 3 or len(asset_sha) != 64:
        raise ReconstructionValidationError(f"I4.3G2B source identity is incomplete for {sid}.")

    g1_scenarios = i4_3g1_snapshot.get("scenario_matrix")
    spec = next((item for item in g1_scenarios if isinstance(item, Mapping) and item.get("scenario_id") == sid), None)
    if spec is None:
        raise ReconstructionValidationError(f"I4.3G2B scenario is absent from G1: {sid}")
    modes = tuple(str(v) for v in spec.get("comparison_modes", ()))
    if ComparisonMode.PRODUCTION_NATIVE.value not in modes:
        raise ReconstructionValidationError(f"I4.3G2B scenario lacks production_native mode: {sid}")

    if verify_files:
        actual = build_source_identity(directory, minimum_images=3)
        if actual.image_count != image_count or actual.asset_set_sha256 != asset_sha:
            raise ReconstructionValidationError(
                f"I4.3G2B source identity changed for {sid}: "
                f"expected_count={image_count} actual_count={actual.image_count} "
                f"expected_sha={asset_sha} actual_sha={actual.asset_set_sha256}"
            )

    matching = "exhaustive" if sid == "g2_temple_ring_mvs_da3" else "sequential"
    return G2BScenarioBinding(
        scenario_id=sid,
        source_directory=directory,
        source_asset_set_sha256=asset_sha,
        source_image_count=image_count,
        binding_method=str(raw.get("binding_method", "frozen_g2a_binding")),
        comparison_modes=modes,
        matching_strategy=matching,
    )


def _runtime_ready_mvs(runtime: Mapping[str, Any]) -> bool:
    return bool(
        runtime.get("available") is True
        and runtime.get("mvs_api_ready") is True
        and runtime.get("patchmatch_cuda_ready") is True
        and runtime.get("recommended_version") is True
        and runtime.get("cuda_distribution") is True
        and int(runtime.get("cuda_devices", 0)) > 0
    )


def _runtime_ready_da3(runtime: Mapping[str, Any]) -> bool:
    return bool(
        runtime.get("available") is True
        and runtime.get("cuda_available") is True
        and runtime.get("depth_anything_3_importable") is True
        and runtime.get("turing_compatibility_patch_applied") is True
        and runtime.get("runtime_lock_valid") is True
        and bool(runtime.get("device_names"))
        and runtime.get("source_revision")
    )


def build_g2b_runtime_preflight_snapshot(
    *,
    i4_3g1_snapshot: Mapping[str, Any],
    i4_3g2a_snapshot: Mapping[str, Any],
    scenario_id: str,
    mvs_runtime: Mapping[str, Any],
    da3_runtime: Mapping[str, Any],
    verify_files: bool = True,
) -> dict[str, Any]:
    binding = resolve_g2b_scenario_binding(
        i4_3g1_snapshot=i4_3g1_snapshot,
        i4_3g2a_snapshot=i4_3g2a_snapshot,
        scenario_id=scenario_id,
        verify_files=verify_files,
    )
    mvs_ready = _runtime_ready_mvs(mvs_runtime)
    da3_ready = _runtime_ready_da3(da3_runtime)
    ready = mvs_ready and da3_ready
    return {
        "schema_version": I43G2B_SCHEMA_VERSION,
        "phase": I43G2B_PHASE,
        "status": "ready" if ready else "runtime_unavailable",
        "pass": ready,
        "scenario": binding.as_dict(),
        "comparison_mode": ComparisonMode.PRODUCTION_NATIVE.value,
        "runtime": {
            "colmap_mvs": dict(mvs_runtime),
            "da3": dict(da3_runtime),
            "mvs_ready": mvs_ready,
            "da3_ready": da3_ready,
        },
        "scope": {
            "runtime_preflight_only": True,
            "source_identity_verified": verify_files,
            "authoritative_sfm_started": False,
            "mvs_started": False,
            "da3_started": False,
            "provider_quality_results_consumed": False,
            "matched_view_diagnostic_started": False,
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
            "universal_provider_winner_assumed": False,
            "global_scalar_score_forbidden": True,
            "matched_view_diagnostic_deferred_until_production_native_evidence": True,
        },
        "next": (
            f"Run production-native {binding.scenario_id} with one authoritative PyCOLMAP SfM shared by MVS and DA3."
            if ready
            else "Repair only the unavailable runtime; do not weaken source/methodology/authority gates."
        ),
    }


def validate_provider_run_evidence(manifest: Mapping[str, Any]) -> None:
    """Validate a G1-compatible provider-run manifest produced by the G2B runner."""
    required_top = (
        "schema_version", "scenario_id", "provider_implementation", "comparison_mode", "attempt",
        "status", "source_identity", "authoritative_sfm_identity", "adaptive_plan_identity",
        "provider_provenance", "runtime_resources", "metrics", "artifacts", "authority",
    )
    missing = [key for key in required_top if key not in manifest]
    if missing:
        raise ReconstructionValidationError(f"I4.3G2B run evidence missing fields: {missing}")
    if manifest.get("scenario_id") not in I43G2B_FRESH_SCENARIOS:
        raise ReconstructionValidationError("I4.3G2B run evidence scenario is not a fresh G2 scenario.")
    if manifest.get("provider_implementation") not in _REQUIRED_PROVIDER_IMPLEMENTATIONS:
        raise ReconstructionValidationError("I4.3G2B run evidence provider is not MVS/DA3.")
    if manifest.get("comparison_mode") not in {
        ComparisonMode.PRODUCTION_NATIVE.value,
        ComparisonMode.MATCHED_VIEW_DIAGNOSTIC.value,
    }:
        raise ReconstructionValidationError("I4.3G2B run evidence comparison mode is invalid.")
    metrics = _mapping(manifest.get("metrics"), label="run metrics")
    reasons = manifest.get("optional_metric_reasons", {})
    if not isinstance(reasons, Mapping):
        raise ReconstructionValidationError("I4.3G2B optional_metric_reasons must be a mapping.")
    for spec in qualification_metric_registry():
        if spec.metric_id in metrics:
            validate_metric_value(spec, metrics[spec.metric_id])
        elif spec.required:
            raise ReconstructionValidationError(f"Required G1 metric missing: {spec.metric_id}")
        elif not str(reasons.get(spec.metric_id, "")).strip():
            raise ReconstructionValidationError(
                f"Optional G1 metric {spec.metric_id} is missing without a persisted reason."
            )
    authority = _mapping(manifest.get("authority"), label="run authority")
    if authority.get("camera_pose_authority_owner") != "pycolmap_sfm":
        raise ReconstructionValidationError("G2B run evidence changed camera-pose authority.")
    if authority.get("permanent_global_view_cap") is not None:
        raise ReconstructionValidationError("G2B run evidence introduced a permanent global view cap.")
    forbidden = (
        "authoritative_sfm_modified", "provider_promoted", "auto_policy_implemented", "hybrid_authorized",
        "metric_scale_claimed", "measurement_authoritative", "certified_metrology", "surface_mesh_generated",
    )
    drifted = [key for key in forbidden if bool(authority.get(key, False))]
    if drifted:
        raise ReconstructionValidationError(f"G2B run evidence authority drift: {drifted}")


def finite_nonnegative(value: Any, *, label: str) -> float:
    number = float(value)
    if not isfinite(number) or number < 0.0:
        raise ReconstructionValidationError(f"{label} must be finite and non-negative.")
    return number


__all__ = [
    "I43G2B_SCHEMA_VERSION",
    "I43G2B_PHASE",
    "I43G2B_FRESH_SCENARIOS",
    "I43G2B_HISTORICAL_SCENARIO",
    "G2BScenarioBinding",
    "sha256_file",
    "read_json_object",
    "verify_snapshot_sha256",
    "validate_g2b_predecessors",
    "resolve_g2b_scenario_binding",
    "build_g2b_runtime_preflight_snapshot",
    "validate_provider_run_evidence",
    "finite_nonnegative",
]
