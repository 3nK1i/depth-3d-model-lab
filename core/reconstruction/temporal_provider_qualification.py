"""I4.3G3A temporal-provider source/runtime binding preflight.

G3A is qualification orchestration only.  It binds the three frozen G1 temporal
scenarios to explicit content-addressed recorded-video sources, validates the G1
framework plus the formally closed G2 handoff, records the already-existing VDA
and RollingDepth runtime readiness, and freezes the temporal comparison
semantics needed by later G3 execution.

No SfM or dense backend is started here.  No provider is selected/promoted, no
Auto/hybrid policy is introduced, and no metric/measurement/metrology/mesh
authority is created.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Any, Mapping, Sequence

from core.reconstruction.cross_provider_qualification import ComparisonMode
from core.reconstruction.errors import ReconstructionValidationError
from core.reconstruction.sources import RecordedVideoReader

I43G3A_SCHEMA_VERSION = 1
I43G3A_PHASE = "I4.3G3A"
I43G3A_SCENARIOS = (
    "g3_temporal_static_sweep",
    "g3_temporal_motion",
    "g3_temporal_low_texture",
)
I43G3A_PROVIDER_IMPLEMENTATIONS = ("rolling_depth", "video_depth_anything")

_EXPECTED_SOURCE_KINDS = {
    "g3_temporal_static_sweep": "ordered_video_static_scene",
    "g3_temporal_motion": "ordered_video_dynamic_scene",
    "g3_temporal_low_texture": "ordered_video_low_texture",
}
_EXPECTED_TEXTURE_REGIMES = {
    "g3_temporal_static_sweep": "mixed",
    "g3_temporal_motion": "mixed",
    "g3_temporal_low_texture": "low",
}
_EXPECTED_MANUAL_REVIEW_DIMENSIONS = (
    "temporal_flicker",
    "temporal_drift",
    "static_geometry_consistency",
    "motion_boundary_stability",
    "transient_artifacts",
    "textureless_region_stability",
)
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


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _valid_sha256(value: Any) -> bool:
    text = str(value).strip().lower()
    return len(text) == 64 and all(ch in "0123456789abcdef" for ch in text)


def _mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ReconstructionValidationError(f"I4.3G3A requires mapping section: {label}")
    return value


@dataclass(frozen=True, slots=True)
class TemporalVideoIdentity:
    source_path: str
    source_file_sha256: str
    asset_set_sha256: str
    file_size_bytes: int
    reader_fingerprint: str
    expected_frame_rate_hz: float
    reported_frame_count: int

    def __post_init__(self) -> None:
        path = Path(self.source_path).expanduser()
        if not path.is_absolute():
            raise ReconstructionValidationError("I4.3G3A temporal source_path must be absolute.")
        object.__setattr__(self, "source_path", str(path.resolve()))
        source_sha = str(self.source_file_sha256).strip().lower()
        asset_sha = str(self.asset_set_sha256).strip().lower()
        fingerprint = str(self.reader_fingerprint).strip().lower()
        if not _valid_sha256(source_sha) or not _valid_sha256(asset_sha) or not _valid_sha256(fingerprint):
            raise ReconstructionValidationError("I4.3G3A temporal source hashes/fingerprint must be SHA256 values.")
        if source_sha != asset_sha:
            raise ReconstructionValidationError("I4.3G3A single-video asset identity must equal the full source-file SHA256.")
        size = int(self.file_size_bytes)
        if size <= 0:
            raise ReconstructionValidationError("I4.3G3A temporal source file must be non-empty.")
        fps = float(self.expected_frame_rate_hz)
        if not (fps > 0.0):
            raise ReconstructionValidationError("I4.3G3A temporal source frame rate must be positive.")
        count = int(self.reported_frame_count)
        if count < 0:
            raise ReconstructionValidationError("I4.3G3A reported_frame_count must be non-negative.")
        if 0 < count < 2:
            raise ReconstructionValidationError("I4.3G3A temporal source must contain at least two reported frames.")
        object.__setattr__(self, "source_file_sha256", source_sha)
        object.__setattr__(self, "asset_set_sha256", asset_sha)
        object.__setattr__(self, "reader_fingerprint", fingerprint)
        object.__setattr__(self, "file_size_bytes", size)
        object.__setattr__(self, "expected_frame_rate_hz", fps)
        object.__setattr__(self, "reported_frame_count", count)

    def as_dict(self) -> dict[str, Any]:
        return {
            "source_path": self.source_path,
            "source_file_sha256": self.source_file_sha256,
            "asset_set_sha256": self.asset_set_sha256,
            "file_size_bytes": self.file_size_bytes,
            "reader_fingerprint": self.reader_fingerprint,
            "expected_frame_rate_hz": self.expected_frame_rate_hz,
            "reported_frame_count": self.reported_frame_count,
        }


@dataclass(frozen=True, slots=True)
class TemporalScenarioBinding:
    scenario_id: str
    source_identity: TemporalVideoIdentity
    manifest_path: str
    manifest_sha256: str
    semantic_binding_method: str

    def __post_init__(self) -> None:
        scenario_id = str(self.scenario_id).strip()
        if scenario_id not in I43G3A_SCENARIOS:
            raise ReconstructionValidationError(f"Unsupported I4.3G3A scenario: {scenario_id}")
        if not isinstance(self.source_identity, TemporalVideoIdentity):
            raise TypeError("source_identity must be TemporalVideoIdentity.")
        manifest = Path(self.manifest_path).expanduser()
        if not manifest.is_absolute():
            raise ReconstructionValidationError("I4.3G3A source manifest_path must be absolute.")
        digest = str(self.manifest_sha256).strip().lower()
        if not _valid_sha256(digest):
            raise ReconstructionValidationError("I4.3G3A source manifest SHA256 is invalid.")
        method = str(self.semantic_binding_method).strip()
        if not method:
            raise ReconstructionValidationError("I4.3G3A requires an explicit semantic binding method.")
        object.__setattr__(self, "scenario_id", scenario_id)
        object.__setattr__(self, "manifest_path", str(manifest.resolve()))
        object.__setattr__(self, "manifest_sha256", digest)
        object.__setattr__(self, "semantic_binding_method", method)

    def as_dict(self) -> dict[str, Any]:
        identity = self.source_identity.as_dict()
        return {
            "scenario_id": self.scenario_id,
            "status": "bound",
            "semantic_binding_inferred": False,
            "semantic_binding_method": self.semantic_binding_method,
            "source_identity": {
                "manifest_path": self.manifest_path,
                "manifest_sha256": self.manifest_sha256,
                "asset_set_sha256": identity["asset_set_sha256"],
                **identity,
            },
            "temporal_contract": {
                "source_kind": _EXPECTED_SOURCE_KINDS[self.scenario_id],
                "ordered_source_required": True,
                "strictly_increasing_canonical_timestamps_required_after_ingestion": True,
                "production_native_source_order_preserved": True,
                "matched_view_contiguous_source_order_required": True,
            },
        }


def build_temporal_video_identity(path: str | Path) -> TemporalVideoIdentity:
    """Build a full-content qualification identity while reusing the production reader probe."""

    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise ReconstructionValidationError(f"I4.3G3A temporal source does not exist: {source}")
    reader = RecordedVideoReader(source)
    metadata = reader.metadata
    reported = int(dict(metadata.metadata).get("reported_frame_count", 0))
    full_sha = _sha256_file(source)
    return TemporalVideoIdentity(
        source_path=str(source),
        source_file_sha256=full_sha,
        asset_set_sha256=full_sha,
        file_size_bytes=source.stat().st_size,
        reader_fingerprint=metadata.fingerprint,
        expected_frame_rate_hz=float(reader.frame_rate_hz),
        reported_frame_count=reported,
    )


def validate_i4_3g3_predecessors(
    *,
    i4_3g1_snapshot: Mapping[str, Any],
    i4_3g2_closure: Mapping[str, Any],
) -> None:
    if i4_3g1_snapshot.get("phase") != "I4.3G1" or i4_3g1_snapshot.get("status") != "pass" or i4_3g1_snapshot.get("pass") is not True:
        raise ReconstructionValidationError("I4.3G3A requires the passing I4.3G1 framework snapshot.")
    coverage = _mapping(i4_3g1_snapshot.get("coverage"), label="G1 coverage")
    if coverage.get("g3_scenarios") != 3:
        raise ReconstructionValidationError("I4.3G3A requires exactly three frozen G3 scenarios.")
    scenarios = i4_3g1_snapshot.get("scenario_matrix")
    if not isinstance(scenarios, Sequence) or isinstance(scenarios, (str, bytes)):
        raise ReconstructionValidationError("I4.3G3A requires the frozen G1 scenario matrix.")
    g3 = [item for item in scenarios if isinstance(item, Mapping) and item.get("substep") == "I4.3G3"]
    if len(g3) != 3:
        raise ReconstructionValidationError("I4.3G3A G1 temporal scenario coverage drifted.")
    by_id = {str(item.get("scenario_id")): item for item in g3}
    if tuple(sid for sid in I43G3A_SCENARIOS if sid in by_id) != I43G3A_SCENARIOS or len(by_id) != 3:
        raise ReconstructionValidationError("I4.3G3A frozen temporal scenario IDs drifted.")
    expected_modes = (ComparisonMode.PRODUCTION_NATIVE.value, ComparisonMode.MATCHED_VIEW_DIAGNOSTIC.value)
    for sid in I43G3A_SCENARIOS:
        item = by_id[sid]
        if tuple(item.get("provider_implementations", ())) != I43G3A_PROVIDER_IMPLEMENTATIONS:
            raise ReconstructionValidationError(f"I4.3G3A provider-set/order drift for {sid}.")
        if tuple(item.get("comparison_modes", ())) != expected_modes:
            raise ReconstructionValidationError(f"I4.3G3A comparison-mode drift for {sid}.")
        if item.get("temporal_order_required") is not True or item.get("overlap_regime") != "temporal_contiguous":
            raise ReconstructionValidationError(f"I4.3G3A temporal-order contract drift for {sid}.")
        if item.get("source_kind") != _EXPECTED_SOURCE_KINDS[sid] or item.get("texture_regime") != _EXPECTED_TEXTURE_REGIMES[sid]:
            raise ReconstructionValidationError(f"I4.3G3A source/texture contract drift for {sid}.")
        if item.get("new_real_execution_required") is not True or item.get("manual_review_required") is not True:
            raise ReconstructionValidationError(f"I4.3G3A qualification obligations drift for {sid}.")
        if tuple(item.get("manual_review_dimensions", ())) != _EXPECTED_MANUAL_REVIEW_DIMENSIONS:
            raise ReconstructionValidationError(f"I4.3G3A manual-review dimensions drift for {sid}.")

    decision_rules = _mapping(i4_3g1_snapshot.get("decision_rules"), label="G1 decision rules")
    for key in (
        "production_native_plan_is_primary_operating_envelope",
        "matched_view_diagnostic_is_secondary_causal_diagnostic",
        "matched_view_mode_must_preserve_provider_semantics",
        "temporal_matched_view_mode_requires_contiguous_source_order",
        "same_source_asset_identity_required_within_scenario",
        "same_manual_review_camera_and_framing_required",
        "independent_visual_auto_fit_forbidden",
        "scenario_specific_verdicts_required",
        "global_scalar_score_forbidden",
    ):
        if decision_rules.get(key) is not True:
            raise ReconstructionValidationError(f"I4.3G3A G1 decision-rule drift: {key}")
    for key in (
        "auto_policy_implemented",
        "hybrid_experiment_authorized",
        "provider_promotion_performed",
        "universal_provider_winner_assumed",
    ):
        if decision_rules.get(key) is not False:
            raise ReconstructionValidationError(f"I4.3G3A G1 decision-rule authority drift: {key}")

    fairness = _mapping(i4_3g1_snapshot.get("fairness"), label="G1 fairness")
    for key in (
        "production_native_comparison_primary",
        "matched_view_diagnostic_secondary",
        "same_source_identity_within_scenario",
        "provider_specific_adaptive_plans_persisted",
        "temporal_contiguous_source_order_required",
        "manual_review_same_camera_and_framing",
        "independent_visual_auto_fit_forbidden",
    ):
        if fairness.get(key) is not True:
            raise ReconstructionValidationError(f"I4.3G3A G1 fairness drift: {key}")
    authority = _mapping(i4_3g1_snapshot.get("authority"), label="G1 authority")
    if authority.get("camera_pose_authority_owner") != "pycolmap_sfm" or authority.get("permanent_global_view_cap") is not None:
        raise ReconstructionValidationError("I4.3G3A requires unchanged PyCOLMAP authority and no permanent view cap.")
    drifted = [key for key in _REQUIRED_AUTHORITY_FALSE if authority.get(key) is not False]
    if drifted:
        raise ReconstructionValidationError(f"I4.3G3A G1 authority drift: {drifted}")

    if i4_3g2_closure.get("phase") != "I4.3G2" or i4_3g2_closure.get("status") != "formally_closed":
        raise ReconstructionValidationError("I4.3G3A requires the formally closed I4.3G2 closure.")
    if i4_3g2_closure.get("i4_3g2_closed") is not True or i4_3g2_closure.get("i4_3g3_authorized") is not True:
        raise ReconstructionValidationError("I4.3G3A requires explicit G2 closure and G3 authorization.")
    synthesis = _mapping(i4_3g2_closure.get("synthesis"), label="G2 synthesis")
    if synthesis.get("camera_pose_authority_owner") != "pycolmap_sfm":
        raise ReconstructionValidationError("I4.3G3A G2 closure changed camera-pose authority.")
    for key in (
        "auto_provider_selection_implemented",
        "hybrid_provider_authorized",
        "metric_scale_claimed",
        "measurement_authoritative",
        "certified_metrology",
        "provider_promoted",
        "provider_selected",
        "surface_mesh_generated",
        "universal_provider_winner_declared",
    ):
        if synthesis.get(key) is not False:
            raise ReconstructionValidationError(f"I4.3G3A refuses G2 synthesis drift: {key}")


def _runtime_ready(runtime: Mapping[str, Any], *, implementation: str) -> bool:
    if runtime.get("available") is not True:
        return False
    if not str(runtime.get("source_revision", "")).strip():
        return False
    if runtime.get("source_ready") is not True or runtime.get("checkpoint_ready") is not True:
        return False
    if implementation == "video_depth_anything":
        return bool(runtime.get("adapter_importable") is True and runtime.get("upstream_importable") is True and runtime.get("torch_available") is True)
    if implementation == "rolling_depth":
        return bool(
            runtime.get("runtime_lock_valid") is True
            and runtime.get("upstream_importable") is True
            and runtime.get("modified_diffusers_ready") is True
            and runtime.get("torch_available") is True
            and runtime.get("ffmpeg_available") is True
        )
    raise ReconstructionValidationError(f"Unsupported I4.3G3A runtime implementation: {implementation}")


def build_i4_3g3a_preflight_snapshot(
    *,
    i4_3g1_snapshot: Mapping[str, Any],
    i4_3g2_closure: Mapping[str, Any],
    scenario_bindings: Sequence[TemporalScenarioBinding],
    video_depth_anything_runtime: Mapping[str, Any],
    rolling_depth_runtime: Mapping[str, Any],
) -> dict[str, Any]:
    validate_i4_3g3_predecessors(i4_3g1_snapshot=i4_3g1_snapshot, i4_3g2_closure=i4_3g2_closure)
    bindings = tuple(scenario_bindings)
    if len(bindings) != 3 or any(not isinstance(item, TemporalScenarioBinding) for item in bindings):
        raise ReconstructionValidationError("I4.3G3A requires exactly three explicit TemporalScenarioBinding values.")
    by_id = {item.scenario_id: item for item in bindings}
    if len(by_id) != 3 or any(sid not in by_id for sid in I43G3A_SCENARIOS):
        raise ReconstructionValidationError("I4.3G3A scenario binding coverage is incomplete or duplicated.")
    ordered = tuple(by_id[sid] for sid in I43G3A_SCENARIOS)

    vda = dict(video_depth_anything_runtime)
    rolling = dict(rolling_depth_runtime)
    vda_ready = _runtime_ready(vda, implementation="video_depth_anything")
    rolling_ready = _runtime_ready(rolling, implementation="rolling_depth")
    ready = vda_ready and rolling_ready

    return {
        "schema_version": I43G3A_SCHEMA_VERSION,
        "phase": I43G3A_PHASE,
        "status": "ready" if ready else "runtime_unavailable",
        "pass": ready,
        "scenario_bindings": [item.as_dict() for item in ordered],
        "binding_summary": {
            "required_scenario_count": 3,
            "bound_scenario_count": len(ordered),
            "all_g3_sources_bound": len(ordered) == 3,
            "semantic_binding_inferred": False,
        },
        "runtime": {
            "video_depth_anything": vda,
            "rolling_depth": rolling,
            "video_depth_anything_ready": vda_ready,
            "rolling_depth_ready": rolling_ready,
        },
        "production_native_contract": {
            "comparison_mode": ComparisonMode.PRODUCTION_NATIVE.value,
            "primary_operating_envelope_evidence": True,
            "provider_specific_i4_3f_adaptive_plans_required": True,
            "source_order_preserved": True,
            "contiguous_subsequence_required": False,
            "same_source_identity_within_scenario": True,
            "same_authoritative_sfm_identity_within_scenario_required": True,
        },
        "matched_view_contract": {
            "comparison_mode": ComparisonMode.MATCHED_VIEW_DIAGNOSTIC.value,
            "secondary_causal_diagnostic_only": True,
            "common_contiguous_source_ordered_subsequence_required": True,
            "provider_specific_temporal_semantics_must_be_preserved": True,
            "same_source_identity_within_scenario": True,
            "same_authoritative_sfm_identity_within_scenario_required": True,
            "same_manual_review_camera_and_framing_required": True,
            "independent_provider_auto_fit_forbidden": True,
        },
        "scope": {
            "source_runtime_binding_preflight_only": True,
            "real_provider_execution_started": False,
            "authoritative_sfm_started": False,
            "video_depth_anything_started": False,
            "rolling_depth_started": False,
            "provider_quality_results_consumed": False,
            "matched_view_diagnostic_started": False,
            "manual_provider_ranking_performed": False,
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
            "i4_3g3b_production_native_execution_authorized": ready,
            "scenario_specific_only": True,
            "universal_provider_winner_assumed": False,
            "global_scalar_score_forbidden": True,
            "matched_view_diagnostic_deferred_until_production_native_evidence": True,
            "provider_selected": False,
            "provider_promoted": False,
        },
        "next": (
            "I4.3G3B production-native temporal execution, one frozen scenario at a time, using provider-specific I4.3F plans."
            if ready
            else "Repair only the unavailable temporal runtime; do not weaken source, temporal-order, methodology, or authority gates."
        ),
    }


__all__ = [
    "I43G3A_SCHEMA_VERSION",
    "I43G3A_PHASE",
    "I43G3A_SCENARIOS",
    "I43G3A_PROVIDER_IMPLEMENTATIONS",
    "TemporalVideoIdentity",
    "TemporalScenarioBinding",
    "build_temporal_video_identity",
    "validate_i4_3g3_predecessors",
    "build_i4_3g3a_preflight_snapshot",
]
