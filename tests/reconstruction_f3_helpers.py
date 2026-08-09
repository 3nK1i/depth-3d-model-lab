from __future__ import annotations

from dataclasses import dataclass
from uuid import uuid4

import numpy as np

from core.contracts import FrameReference, ImageSize, ModelProvenance, PixelCoordinateSpace, PixelSpaceRole
from core.models.metadata import ConfidenceKind
from core.perception import DenseMaskPayload, InstanceMask, MaskSource, ObjectObservation
from core.reconstruction import (
    BoundEvidenceProviderExecutor,
    CanonicalDenseGeometryProviderExecutor,
    DenseDepthUnit,
    DenseFusionPolicy,
    DenseGeometryEvidence,
    DenseSceneProfile,
    FeatureTolerance,
    InspectionQCProviderExecutor,
    PipelineCameraTrajectoryEvidence,
    PipelineDepthEvidence,
    PipelineIngestionEvidence,
    PipelinePayloadStore,
    ReconstructionPath,
    ReconstructionPipelineGoal,
    ReconstructionPipelineOrchestrator,
    ReconstructionPipelineRequest,
    ReconstructionPipelineStage,
    ReconstructionProviderRole,
    ReconstructionQualityProfile,
    ReferenceGeometrySource,
    SemanticFrameObservation,
    SemanticGeometryProviderExecutor,
    SemanticGeometryRole,
    SurfaceMeshProviderExecutor,
    arbitrary_scale_reference,
    bound_evidence_provider_descriptor,
    build_reconstruction_pipeline_plan,
    build_reference_assembly,
    canonical_geometry_provider_descriptors,
    estimate_camera_geometry_from_horizontal_fov,
    project_surface_mesh_vertices,
    reconstruction_unit_depth_alignment,
)


def pose(tx: float) -> np.ndarray:
    matrix = np.eye(4, dtype=np.float64)
    matrix[0, 3] = tx
    return matrix


def perception_provenance() -> ModelProvenance:
    return ModelProvenance(
        model_id="phase8-7f3-semantic-fixture",
        display_name="Phase 8.7F3 semantic fixture",
        task="instance_segmentation",
        model_version="1",
        device="validation",
    )


def observation_for_region(
    surface,
    camera,
    *,
    frame_index: int,
    tx: float,
    class_id: int,
    class_name: str,
    track_id: str,
    target_vertex_indices: np.ndarray,
) -> SemanticFrameObservation:
    size = camera.camera.image_size
    frame_ref = FrameReference(
        session_id=873,
        frame_id=frame_index,
        source_timestamp=1.0 + frame_index * 0.05,
        source_size=size,
    )
    pixel_space = PixelCoordinateSpace(
        camera.camera.intrinsics.pixel_space_id,
        frame_ref,
        size,
        PixelSpaceRole.RGB,
    )
    T = pose(tx)
    projection = project_surface_mesh_vertices(surface, camera, T)
    wanted = np.isin(projection.visible_vertex_indices, target_vertex_indices)
    mask = np.zeros((size.height, size.width), dtype=bool)
    pixels = projection.pixel_xy[wanted]
    if len(pixels):
        x = np.clip(np.rint(pixels[:, 0]).astype(np.int64), 0, size.width - 1)
        y = np.clip(np.rint(pixels[:, 1]).astype(np.int64), 0, size.height - 1)
        mask[y, x] = True
    instance_mask = InstanceMask(
        mask_id=f"mask-{class_name}-{track_id}-{frame_index}",
        frame_reference=frame_ref,
        pixel_space=pixel_space,
        payload=DenseMaskPayload(mask),
        source=MaskSource.NATIVE_INSTANCE,
        model_provenance=perception_provenance(),
        confidence=0.96,
        confidence_kind=ConfidenceKind.CONFIDENCE,
    )
    observation = ObjectObservation(
        observation_id=f"obs-{class_name}-{track_id}-{frame_index}",
        frame_reference=frame_ref,
        class_id=class_id,
        class_name=class_name,
        masks=(instance_mask,),
        track_id=track_id,
    )
    return SemanticFrameObservation(
        frame_id=f"f3-{frame_index}",
        observation=observation,
        camera_profile=camera,
        T_reconstruction_from_camera=T,
        pose_layer="optimized_keyframe" if frame_index in {0, 4} else "optimized_propagated",
    )


@dataclass
class F3Fixture:
    session_id: str
    request: ReconstructionPipelineRequest
    store: PipelinePayloadStore
    providers: tuple
    executors: dict
    plan: object
    orchestrator: ReconstructionPipelineOrchestrator


def make_f3_fixture(*, session_id: str | None = None, request_id: str = "phase8-7f3-test") -> F3Fixture:
    session_id = session_id or str(uuid4())
    width, height = 96, 72
    camera = estimate_camera_geometry_from_horizontal_fov(
        profile_id="phase8-7f3-camera",
        camera_id="phase8-7f3-camera",
        image_size=ImageSize(width, height),
        horizontal_fov_degrees=72.0,
        focal_length_relative_uncertainty=0.15,
        camera_frame_id="phase8-7f3-camera-frame",
        reconstruction_frame_id="phase8-7f3-reconstruction",
        notes="Phase 8.7F3 integration fixture",
    )
    scale = arbitrary_scale_reference(profile_id="phase8-7f3-arbitrary")
    translations = (-0.12, -0.06, 0.0, 0.06, 0.12)
    frame_ids = tuple(f"f3-{i}" for i in range(len(translations)))
    timestamps = tuple(1_000_000_000 + i * 50_000_000 for i in range(len(translations)))
    rgb_frames = []
    depth_frames = []
    confidence_frames = []
    poses = []
    layers = []
    for i, tx in enumerate(translations):
        rgb = np.zeros((height, width, 3), dtype=np.uint8)
        rgb[..., 0] = np.arange(width, dtype=np.uint8)[None, :]
        rgb[..., 1] = 70 + i * 25
        rgb[..., 2] = 190
        rgb_frames.append(rgb)
        depth_frames.append(np.full((height, width), 2.0, dtype=np.float32))
        confidence_frames.append(np.full((height, width), 0.96, dtype=np.float32))
        poses.append(pose(tx))
        layers.append("optimized_keyframe" if i in {0, 4} else "optimized_propagated")
    alignment = reconstruction_unit_depth_alignment(provenance="Phase 8.7F3 fixture plane")
    ingestion = PipelineIngestionEvidence(
        session_id=session_id,
        frame_ids=frame_ids,
        timestamps_ns=timestamps,
        rgb_frames=tuple(rgb_frames),
        camera_profile=camera,
        scale_profile=scale,
        reconstruction_frame_id="phase8-7f3-reconstruction",
        scene_profile=DenseSceneProfile.OBJECT_INSPECTION,
        source_label="deterministic-f3-integration-fixture",
    )
    trajectory = PipelineCameraTrajectoryEvidence(
        session_id=session_id,
        frame_ids=frame_ids,
        T_reconstruction_from_camera=tuple(poses),
        pose_layers=tuple(layers),
        source_provider_id="qualified-pose-evidence",
    )
    depth = PipelineDepthEvidence(
        session_id=session_id,
        frame_ids=frame_ids,
        depth_frames=tuple(depth_frames),
        depth_unit=DenseDepthUnit.RECONSTRUCTION_UNITS,
        depth_alignments=tuple(alignment for _ in frame_ids),
        confidence_frames=tuple(confidence_frames),
        source_provider_id="qualified-depth-evidence",
        geometry_scale="arbitrary_reconstruction_scale",
    )
    store = PipelinePayloadStore()
    input_ref = store.put(ingestion, payload_ref=f"memory://{request_id}/ingestion")
    pose_ref = store.put(trajectory, payload_ref=f"memory://{request_id}/poses")
    depth_ref = store.put(depth, payload_ref=f"memory://{request_id}/depth")

    evidence_id = "phase8-f3-bound-evidence"
    evidence_provider = bound_evidence_provider_descriptor(provider_id=evidence_id)
    geometry_providers = canonical_geometry_provider_descriptors()
    providers = (evidence_provider,) + geometry_providers

    bound = BoundEvidenceProviderExecutor(
        store=store,
        provider_id=evidence_id,
        payload_ref_by_stage={
            ReconstructionPipelineStage.SOURCE_INGESTION: input_ref,
            ReconstructionPipelineStage.CAMERA_POSE: pose_ref,
            ReconstructionPipelineStage.DEPTH_EVIDENCE: depth_ref,
        },
    )

    def good_quality(dense):
        count = len(dense.point_cloud.points_reconstruction)
        return DenseGeometryEvidence(
            point_count=count,
            depth_relative_residual=np.full(count, 0.01, dtype=np.float32),
            view_angle_span_degrees=np.full(count, 28.0, dtype=np.float32),
            visibility_ratio=np.full(count, 0.97, dtype=np.float32),
            provenance="Phase 8.7F3 trusted integration evidence",
        )

    dense_executor = CanonicalDenseGeometryProviderExecutor(
        store=store,
        policy=DenseFusionPolicy(
            scene_profile=DenseSceneProfile.OBJECT_INSPECTION,
            voxel_size_reconstruction_units=0.075,
            pixel_stride=2,
            minimum_view_observations=2,
            max_voxels=100_000,
        ),
        quality_evidence_builder=good_quality,
    )
    surface_executor = SurfaceMeshProviderExecutor(store=store)

    def semantic_observations(_request, surface):
        vertices = surface.mesh.vertices_reconstruction
        body = np.flatnonzero(vertices[:, 0] < -0.25).astype(np.uint32)
        top = np.flatnonzero((vertices[:, 0] > 0.10) & (vertices[:, 1] < -0.16)).astype(np.uint32)
        bottom = np.flatnonzero((vertices[:, 0] > 0.10) & (vertices[:, 1] > 0.16)).astype(np.uint32)
        result = []
        for frame_index in (0, 2, 4):
            tx = translations[frame_index]
            for class_id, class_name, track_id, indices in (
                (0, "connector_body", "connector-main", body),
                (1, "pin", "pin-top", top),
                (1, "pin", "pin-bottom", bottom),
            ):
                result.append(
                    observation_for_region(
                        surface,
                        camera,
                        frame_index=frame_index,
                        tx=tx,
                        class_id=class_id,
                        class_name=class_name,
                        track_id=track_id,
                        target_vertex_indices=indices,
                    )
                )
        return tuple(result)

    semantic_executor = SemanticGeometryProviderExecutor(
        store=store,
        observation_resolver=semantic_observations,
        role_by_class_name={
            "connector_body": SemanticGeometryRole.CONNECTOR_BODY,
            "pin": SemanticGeometryRole.CONNECTOR_PIN,
        },
    )

    tolerance = FeatureTolerance(
        position_warn=0.08,
        position_reject=0.16,
        orientation_warn_degrees=5.0,
        orientation_reject_degrees=12.0,
        extent_relative_warn=0.08,
        extent_relative_reject=0.16,
        minimum_geometry_quality=0.60,
        maximum_uncertainty_fraction=0.30,
    )

    def reference_resolver(_request, surface, semantic):
        return build_reference_assembly(
            surface,
            semantic,
            reference_id="phase8-7f3-self-reference",
            source=ReferenceGeometrySource.GOLDEN_RECONSTRUCTION,
            default_tolerance=tolerance,
            registration_anchor_feature_ids=("connector-main",),
        )

    inspection_executor = InspectionQCProviderExecutor(
        store=store,
        reference_resolver=reference_resolver,
    )
    executors = {
        evidence_id: bound,
        geometry_providers[0].provider_id: dense_executor,
        geometry_providers[1].provider_id: surface_executor,
        geometry_providers[2].provider_id: semantic_executor,
        geometry_providers[3].provider_id: inspection_executor,
    }
    request = ReconstructionPipelineRequest(
        request_id=request_id,
        session_id=session_id,
        path=ReconstructionPath.OFFLINE,
        quality_profile=ReconstructionQualityProfile.EFFICIENT,
        goals=(ReconstructionPipelineGoal.SEMANTIC_INSPECTION,),
    )
    plan = build_reconstruction_pipeline_plan(request, providers)
    orchestrator = ReconstructionPipelineOrchestrator(plan=plan, providers=providers, executors=executors)
    return F3Fixture(session_id, request, store, providers, executors, plan, orchestrator)
