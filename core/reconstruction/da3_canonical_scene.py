"""Phase 9.3H DA3 supporting evidence -> canonical geometry integration.

This module reuses the qualified Phase 8 DA3 and canonical dense/surface
providers.  DA3 remains supporting relative geometry: it never establishes
metric scale, semantic authority, inspection authority, or certified metrology.

The helper intentionally accepts an already-materialized :class:`DA3ResultManifest`.
Launching/probing the isolated DA3 runtime belongs to the application worker
boundary, not this production-depth core module.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from math import isfinite
from typing import Mapping, Sequence

import numpy as np

from core.contracts.validation import JsonValue, require_nonempty
from core.reconstruction.da3 import DA3_PROVIDER_ID, DA3ResultManifest, validate_da3_npz_result
from core.reconstruction.dense_fusion import DenseFusionPolicy
from core.reconstruction.geometry_quality import DenseGeometryQualityPolicy
from core.reconstruction.orchestration import (
    ReconstructionArtifactAuthority,
    ReconstructionArtifactKind,
    ReconstructionPath,
    ReconstructionPipelineArtifact,
    ReconstructionPipelineExecutionScope,
    ReconstructionPipelineGoal,
    ReconstructionPipelineOrchestrator,
    ReconstructionPipelineRequest,
    ReconstructionPipelineRunResult,
    ReconstructionPipelineRunStatus,
    ReconstructionPipelineStage,
    ReconstructionProviderDescriptor,
    ReconstructionProviderExecution,
    ReconstructionProviderRole,
    ReconstructionStageExecutionContext,
    build_reconstruction_pipeline_plan,
)
from core.reconstruction.pipeline_geometry import (
    BoundEvidenceProviderExecutor,
    CanonicalDenseGeometryPayload,
    CanonicalDenseGeometryProviderExecutor,
    DenseSceneProfile,
    PipelineCameraTrajectoryEvidence,
    PipelineDepthEvidence,
    PipelineIngestionEvidence,
    PipelinePayloadStore,
    SurfaceMeshProviderExecutor,
    canonical_geometry_provider_descriptors,
    da3_camera_profile_from_result,
    da3_supporting_evidence_from_result,
)
from core.reconstruction.profiles import ReconstructionQualityProfile
from core.reconstruction.scale_reference import arbitrary_scale_reference
from core.reconstruction.surface_mesh import SurfaceMeshResult
from core.reconstruction.errors import ReconstructionValidationError

PHASE9_3H_SOURCE_PROVIDER_ID = "phase9-3h-source-evidence"
PHASE9_3H_SEMANTIC_UNAVAILABLE_PROVIDER_ID = "phase9-3h-semantic-unavailable"
PHASE9_3H_INSPECTION_UNAVAILABLE_PROVIDER_ID = "phase9-3h-inspection-unavailable"
PHASE9_3H_RECONSTRUCTION_FRAME_ID = "phase9-3h-da3-reconstruction"
PHASE9_3H_SCALE_PROFILE_ID = "phase9-3h-da3-arbitrary-scale"


@dataclass(frozen=True, slots=True)
class DA3CanonicalSceneIntegrationResult:
    """Typed result needed by the persistence/UI bridge after 9.3H integration."""

    pipeline_result: ReconstructionPipelineRunResult
    ingestion: PipelineIngestionEvidence
    trajectory: PipelineCameraTrajectoryEvidence
    depth: PipelineDepthEvidence
    canonical_dense: CanonicalDenseGeometryPayload
    surface: SurfaceMeshResult
    da3_manifest: DA3ResultManifest
    frame_ids: tuple[str, ...]
    reconstruction_frame_id: str
    semantic_context_available: bool = False
    inspection_qc_available: bool = False
    certified_metrology: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.pipeline_result, ReconstructionPipelineRunResult):
            raise TypeError("pipeline_result must be ReconstructionPipelineRunResult.")
        if not isinstance(self.ingestion, PipelineIngestionEvidence):
            raise TypeError("ingestion must be PipelineIngestionEvidence.")
        if not isinstance(self.trajectory, PipelineCameraTrajectoryEvidence):
            raise TypeError("trajectory must be PipelineCameraTrajectoryEvidence.")
        if not isinstance(self.depth, PipelineDepthEvidence):
            raise TypeError("depth must be PipelineDepthEvidence.")
        if not isinstance(self.canonical_dense, CanonicalDenseGeometryPayload):
            raise TypeError("canonical_dense must be CanonicalDenseGeometryPayload.")
        if not isinstance(self.surface, SurfaceMeshResult):
            raise TypeError("surface must be SurfaceMeshResult.")
        if not isinstance(self.da3_manifest, DA3ResultManifest):
            raise TypeError("da3_manifest must be DA3ResultManifest.")
        if self.pipeline_result.status is not ReconstructionPipelineRunStatus.DEGRADED:
            raise ReconstructionValidationError(
                "Phase 9.3H DA3 canonical scenes are degraded until semantics/QC are separately available."
            )
        if not self.pipeline_result.canonical_geometry_authoritative:
            raise ReconstructionValidationError("DA3 canonical integration must produce canonical geometry.")
        if self.pipeline_result.certified_metrology or self.certified_metrology:
            raise ReconstructionValidationError("Phase 9.3H cannot certify metrology.")
        if self.semantic_context_available or self.inspection_qc_available:
            raise ReconstructionValidationError(
                "Phase 9.3H does not fabricate semantic/inspection evidence for a new DA3 scene."
            )
        if self.depth.source_provider_id != DA3_PROVIDER_ID or self.trajectory.source_provider_id != DA3_PROVIDER_ID:
            raise ReconstructionValidationError("DA3 supporting evidence provider identity was lost.")
        if self.surface.metric_geometry_status.value != "arbitrary":
            raise ReconstructionValidationError(
                "Phase 9.3H DA3 integration must remain arbitrary-scale without separate qualification."
            )
        if self.surface.unit_to_metre is not None:
            raise ReconstructionValidationError("Phase 9.3H must not invent unit_to_metre.")

    @property
    def canonical_geometry_authoritative(self) -> bool:
        return True


class _UnavailableContextExecutor:
    """Persist an explicit unavailable semantic/QC stage without fabricating content."""

    def __init__(self, *, provider_id: str, reason: str) -> None:
        self.provider_id = require_nonempty(provider_id, field_name="provider_id")
        self.reason = require_nonempty(reason, field_name="reason")

    def execute(self, context: ReconstructionStageExecutionContext) -> ReconstructionPipelineArtifact:
        if context.provider.provider_id != self.provider_id:
            raise ReconstructionValidationError("Unavailable-context provider identity mismatch.")
        contract = context.planned_stage.contract
        if context.planned_stage.stage not in {
            ReconstructionPipelineStage.SEMANTIC_GEOMETRY,
            ReconstructionPipelineStage.INSPECTION_QC,
        }:
            raise ReconstructionValidationError("Unavailable-context executor serves only semantic/QC stages.")
        return ReconstructionPipelineArtifact(
            artifact_id=f"{context.request.request_id}:{context.planned_stage.stage.value}:{self.provider_id}",
            session_id=context.request.session_id,
            stage=context.planned_stage.stage,
            kind=contract.output_kind,
            authority=contract.output_authority,
            provider_id=self.provider_id,
            metadata={
                "available": False,
                "unavailable_reason": self.reason,
                "object_count": 0,
                "observations_total": 0,
                "observations_accepted": 0,
                "canonical_geometry_authoritative": False,
                "certified_metrology": False,
                "fabricated": False,
            },
        )


def _source_provider() -> ReconstructionProviderDescriptor:
    return ReconstructionProviderDescriptor(
        provider_id=PHASE9_3H_SOURCE_PROVIDER_ID,
        provider_version="phase9-3h",
        adapter_version="phase9-3h",
        roles=(ReconstructionProviderRole.INGESTION,),
        execution=ReconstructionProviderExecution.IN_PROCESS,
        priority=100,
        metadata={
            "source": "phase9_3h_selected_input",
            "binding_only": True,
            "development_mock": False,
            "certified_metrology": False,
        },
    )


def _da3_result_provider(manifest: DA3ResultManifest) -> ReconstructionProviderDescriptor:
    return ReconstructionProviderDescriptor(
        provider_id=DA3_PROVIDER_ID,
        provider_version=manifest.source_revision[:12],
        adapter_version="phase9-3h-existing-da3-result-binding",
        roles=(ReconstructionProviderRole.CAMERA_POSE, ReconstructionProviderRole.DEPTH),
        execution=ReconstructionProviderExecution.EXTERNAL_PROCESS,
        priority=150,
        metadata={
            "source_revision": manifest.source_revision,
            "model_id": manifest.model_id,
            "model_revision": manifest.model_revision,
            "geometry_scale": manifest.geometry_scale,
            "external_result_binding": True,
            "canonical_geometry_authoritative": False,
            "certified_metrology": False,
        },
    )


def _unavailable_provider(
    provider_id: str,
    role: ReconstructionProviderRole,
    *,
    reason: str,
) -> ReconstructionProviderDescriptor:
    return ReconstructionProviderDescriptor(
        provider_id=provider_id,
        provider_version="phase9-3h",
        adapter_version="phase9-3h",
        roles=(role,),
        execution=ReconstructionProviderExecution.IN_PROCESS,
        priority=900,
        metadata={
            "available_context": False,
            "unavailable_reason": reason,
            "fabricated": False,
            "canonical_geometry_authoritative": False,
            "certified_metrology": False,
        },
    )


def integrate_da3_result_to_canonical_scene(
    manifest: DA3ResultManifest,
    *,
    session_id: str,
    request_id: str,
    rgb_frames: Sequence[np.ndarray],
    frame_ids: Sequence[str] | None = None,
    timestamps_ns: Sequence[int] | None = None,
    source_label: str = "phase9-3h-da3-input",
    reconstruction_frame_id: str = PHASE9_3H_RECONSTRUCTION_FRAME_ID,
    pixel_stride: int = 4,
    relative_scale_uncertainty: float = 0.25,
) -> DA3CanonicalSceneIntegrationResult:
    """Canonicalize one already-qualified DA3 result without promoting its authority.

    Calibration is deliberately not required.  The generated scale profile is an
    arbitrary reconstruction gauge and ``unit_to_metre`` remains unavailable.
    """

    if not isinstance(manifest, DA3ResultManifest):
        raise TypeError("manifest must be DA3ResultManifest.")
    request_id = require_nonempty(request_id, field_name="request_id")
    reconstruction_frame_id = require_nonempty(
        reconstruction_frame_id, field_name="reconstruction_frame_id"
    )
    arrays = validate_da3_npz_result(manifest)
    frames = tuple(np.asarray(value, dtype=np.uint8) for value in rgb_frames)
    if len(frames) != manifest.frame_count:
        raise ReconstructionValidationError("RGB frame count must match the DA3 result.")
    if any(frame.ndim != 3 or frame.shape[2] != 3 for frame in frames):
        raise ReconstructionValidationError("rgb_frames must be HxWx3 arrays.")
    expected_shape = (manifest.height, manifest.width, 3)
    if any(frame.shape != expected_shape for frame in frames):
        raise ReconstructionValidationError(
            f"RGB frames must match DA3 output size {manifest.width}x{manifest.height}."
        )
    ids = tuple(frame_ids or (f"da3-9.3h-{index:04d}" for index in range(manifest.frame_count)))
    if len(ids) != manifest.frame_count or len(set(ids)) != len(ids) or any(not str(v).strip() for v in ids):
        raise ReconstructionValidationError("frame_ids must provide one unique non-empty ID per DA3 frame.")
    if timestamps_ns is None:
        timestamps = tuple(1_000_000_000 + index * 50_000_000 for index in range(manifest.frame_count))
    else:
        timestamps = tuple(int(value) for value in timestamps_ns)
        if len(timestamps) != manifest.frame_count or any(value < 0 for value in timestamps):
            raise ReconstructionValidationError("timestamps_ns must match the DA3 frame count and be non-negative.")
    stride = int(pixel_stride)
    if stride < 1 or stride > 32:
        raise ReconstructionValidationError("pixel_stride must be within [1, 32].")
    uncertainty = float(relative_scale_uncertainty)
    if not isfinite(uncertainty) or not 0.0 <= uncertainty < 1.0:
        raise ReconstructionValidationError("relative_scale_uncertainty must be finite in [0,1).")

    camera = da3_camera_profile_from_result(
        manifest,
        reconstruction_frame_id=reconstruction_frame_id,
    )
    scale = arbitrary_scale_reference(
        profile_id=PHASE9_3H_SCALE_PROFILE_ID,
        notes=(
            "Depth Anything 3 relative/arbitrary reconstruction gauge. Phase 9.3H does not "
            "require calibration and does not create metric scale."
        ),
    )
    ingestion = PipelineIngestionEvidence(
        session_id=session_id,
        frame_ids=ids,
        timestamps_ns=timestamps,
        rgb_frames=frames,
        camera_profile=camera,
        scale_profile=scale,
        reconstruction_frame_id=reconstruction_frame_id,
        scene_profile=DenseSceneProfile.SPATIAL_DIGITAL_TWIN,
        source_label=source_label,
        metadata={
            "da3_source_revision": manifest.source_revision,
            "da3_model_id": manifest.model_id,
            "da3_model_revision": manifest.model_revision,
            "calibration_required": False,
            "metric_scale_qualified": False,
            "certified_metrology": False,
        },
    )
    trajectory, depth = da3_supporting_evidence_from_result(
        manifest,
        session_id=session_id,
        frame_ids=ids,
        relative_scale_uncertainty=uncertainty,
    )

    store = PipelinePayloadStore()
    ingestion_ref = store.put(ingestion, payload_ref=f"memory://phase9-3h/{request_id}/ingestion")
    pose_ref = store.put(trajectory, payload_ref=f"memory://phase9-3h/{request_id}/poses")
    depth_ref = store.put(depth, payload_ref=f"memory://phase9-3h/{request_id}/depth")

    source_provider = _source_provider()
    da3_provider = _da3_result_provider(manifest)
    geometry = canonical_geometry_provider_descriptors(priority=500)
    semantic_provider = _unavailable_provider(
        PHASE9_3H_SEMANTIC_UNAVAILABLE_PROVIDER_ID,
        ReconstructionProviderRole.SEMANTICS,
        reason="semantic_context_not_run_for_new_da3_canonical_scene",
    )
    inspection_provider = _unavailable_provider(
        PHASE9_3H_INSPECTION_UNAVAILABLE_PROVIDER_ID,
        ReconstructionProviderRole.INSPECTION_QC,
        reason="inspection_qc_not_run_for_new_da3_canonical_scene",
    )
    providers = (
        source_provider,
        da3_provider,
        geometry[0],
        geometry[1],
        semantic_provider,
        inspection_provider,
    )

    median_depth = float(np.median(arrays["depth"]))
    voxel_size = max(median_depth * 0.02, 1e-4)
    dense_policy = DenseFusionPolicy(
        scene_profile=DenseSceneProfile.SPATIAL_DIGITAL_TWIN,
        voxel_size_reconstruction_units=voxel_size,
        pixel_stride=stride,
        minimum_view_observations=1,
        minimum_confidence=0.05,
        max_voxels=200_000,
    )
    quality_policy = DenseGeometryQualityPolicy(
        scene_profile=DenseSceneProfile.SPATIAL_DIGITAL_TWIN,
        target_view_observations=3,
        minimum_view_observations=1,
        minimum_confidence=0.05,
        desired_view_angle_span_degrees=12.0,
        maximum_depth_relative_residual=0.10,
        high_score_threshold=0.72,
        medium_score_threshold=0.40,
        minimum_input_depth_coverage=0.35,
    )
    source_executor = BoundEvidenceProviderExecutor(
        store=store,
        provider_id=source_provider.provider_id,
        payload_ref_by_stage={ReconstructionPipelineStage.SOURCE_INGESTION: ingestion_ref},
    )
    da3_executor = BoundEvidenceProviderExecutor(
        store=store,
        provider_id=DA3_PROVIDER_ID,
        payload_ref_by_stage={
            ReconstructionPipelineStage.CAMERA_POSE: pose_ref,
            ReconstructionPipelineStage.DEPTH_EVIDENCE: depth_ref,
        },
    )
    dense_executor = CanonicalDenseGeometryProviderExecutor(
        store=store,
        policy=dense_policy,
        quality_policy=quality_policy,
    )
    surface_executor = SurfaceMeshProviderExecutor(store=store)
    semantic_executor = _UnavailableContextExecutor(
        provider_id=PHASE9_3H_SEMANTIC_UNAVAILABLE_PROVIDER_ID,
        reason="semantic_context_not_run_for_new_da3_canonical_scene",
    )
    inspection_executor = _UnavailableContextExecutor(
        provider_id=PHASE9_3H_INSPECTION_UNAVAILABLE_PROVIDER_ID,
        reason="inspection_qc_not_run_for_new_da3_canonical_scene",
    )
    executors = {
        source_provider.provider_id: source_executor,
        DA3_PROVIDER_ID: da3_executor,
        geometry[0].provider_id: dense_executor,
        geometry[1].provider_id: surface_executor,
        semantic_provider.provider_id: semantic_executor,
        inspection_provider.provider_id: inspection_executor,
    }
    request = ReconstructionPipelineRequest(
        request_id=request_id,
        session_id=session_id,
        path=ReconstructionPath.OFFLINE,
        quality_profile=ReconstructionQualityProfile.EFFICIENT,
        goals=(ReconstructionPipelineGoal.SEMANTIC_INSPECTION,),
        allow_optional_degradation=True,
    )
    plan = build_reconstruction_pipeline_plan(request, providers)
    result = ReconstructionPipelineOrchestrator(
        plan=plan,
        providers=providers,
        executors=executors,
    ).run()
    if result.status is ReconstructionPipelineRunStatus.FAILED:
        reasons = [item.reason for item in result.stages if item.reason]
        raise ReconstructionValidationError(
            "DA3 canonical integration failed: " + " | ".join(reasons)
        )
    if not result.canonical_geometry_authoritative:
        raise ReconstructionValidationError("DA3 canonical integration produced no canonical geometry.")

    # Semantic/QC stages are explicit unavailable placeholders required by the
    # legacy persistent-scene contract.  Mark the integrated scene degraded so
    # callers cannot mistake those placeholders for completed semantic/QC work.
    result = replace(
        result,
        status=ReconstructionPipelineRunStatus.DEGRADED,
        execution_scope=ReconstructionPipelineExecutionScope.REAL_PROVIDERS,
    )
    canonical_dense = store.find_unique(CanonicalDenseGeometryPayload)
    surface = store.find_unique(SurfaceMeshResult)
    return DA3CanonicalSceneIntegrationResult(
        pipeline_result=result,
        ingestion=ingestion,
        trajectory=trajectory,
        depth=depth,
        canonical_dense=canonical_dense,
        surface=surface,
        da3_manifest=manifest,
        frame_ids=ids,
        reconstruction_frame_id=reconstruction_frame_id,
        semantic_context_available=False,
        inspection_qc_available=False,
        certified_metrology=False,
    )


__all__ = [
    "DA3CanonicalSceneIntegrationResult",
    "PHASE9_3H_RECONSTRUCTION_FRAME_ID",
    "integrate_da3_result_to_canonical_scene",
]
