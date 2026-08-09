#!/usr/bin/env python3
"""Phase 8.7F3 end-to-end canonical geometry + semantics + QC smoke."""
from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import sys
from uuid import uuid4

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.contracts import ImageSize
from core.reconstruction import (
    BoundEvidenceProviderExecutor,
    CanonicalDenseGeometryPayload,
    CanonicalDenseGeometryProviderExecutor,
    DenseDepthUnit,
    DenseFusionPolicy,
    DenseGeometryEvidence,
    DenseSceneProfile,
    FeatureTolerance,
    InspectionDecision,
    InspectionQCProviderExecutor,
    InspectionReport,
    PipelineCameraTrajectoryEvidence,
    PipelineDepthEvidence,
    PipelineIngestionEvidence,
    PipelinePayloadStore,
    ReconstructionArtifactKind,
    ReconstructionPath,
    ReconstructionPipelineGoal,
    ReconstructionPipelineOrchestrator,
    ReconstructionPipelineRequest,
    ReconstructionPipelineRunStatus,
    ReconstructionPipelineStage,
    ReconstructionQualityProfile,
    ReferenceGeometrySource,
    SemanticGeometryProviderExecutor,
    SemanticGeometryResult,
    SemanticGeometryRole,
    SurfaceMeshProviderExecutor,
    SurfaceMeshResult,
    arbitrary_scale_reference,
    bound_evidence_provider_descriptor,
    build_reconstruction_pipeline_plan,
    build_reference_assembly,
    canonical_geometry_provider_descriptors,
    estimate_camera_geometry_from_horizontal_fov,
    reconstruction_unit_depth_alignment,
    summarize_f3_pipeline_result,
    write_inspection_report_json,
    write_reconstruction_pipeline_json,
    write_semantic_groups_obj,
    write_surface_mesh_obj,
)
from scripts.run_semantic_geometry_smoke import _observation_for_region


def _pose(tx: float) -> np.ndarray:
    matrix = np.eye(4, dtype=np.float64)
    matrix[0, 3] = tx
    return matrix


def main() -> int:
    out = Path("/tmp/phase8_7f3_integrated")
    out.mkdir(parents=True, exist_ok=True)
    session_id = str(uuid4())
    request_id = "phase8-7f3-integrated-smoke"
    width, height = 96, 72
    camera = estimate_camera_geometry_from_horizontal_fov(
        profile_id="phase8-7f3-smoke-camera",
        camera_id="phase8-7f3-smoke-camera",
        image_size=ImageSize(width, height),
        horizontal_fov_degrees=72.0,
        focal_length_relative_uncertainty=0.15,
        reconstruction_frame_id="phase8-7f3-reconstruction",
        notes="F3 integration smoke provisional camera",
    )
    scale = arbitrary_scale_reference(profile_id="phase8-7f3-smoke-arbitrary")
    translations = (-0.12, -0.06, 0.0, 0.06, 0.12)
    frame_ids = tuple(f"f3-{i}" for i in range(len(translations)))
    timestamps = tuple(1_000_000_000 + i * 50_000_000 for i in range(len(translations)))
    rgb_frames, depth_frames, confidence_frames, poses, layers = [], [], [], [], []
    for i, tx in enumerate(translations):
        rgb = np.zeros((height, width, 3), np.uint8)
        rgb[..., 0] = np.arange(width, dtype=np.uint8)[None, :]
        rgb[..., 1] = 70 + i * 25
        rgb[..., 2] = 190
        rgb_frames.append(rgb)
        depth_frames.append(np.full((height, width), 2.0, np.float32))
        confidence_frames.append(np.full((height, width), 0.96, np.float32))
        poses.append(_pose(tx))
        layers.append("optimized_keyframe" if i in {0, 4} else "optimized_propagated")
    alignment = reconstruction_unit_depth_alignment(provenance="Phase 8.7F3 integration smoke plane")
    ingestion = PipelineIngestionEvidence(
        session_id=session_id,
        frame_ids=frame_ids,
        timestamps_ns=timestamps,
        rgb_frames=tuple(rgb_frames),
        camera_profile=camera,
        scale_profile=scale,
        reconstruction_frame_id="phase8-7f3-reconstruction",
        scene_profile=DenseSceneProfile.OBJECT_INSPECTION,
        source_label="deterministic-f3-integration-smoke",
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
    ingestion_ref = store.put(ingestion, payload_ref="memory://phase8-7f3/ingestion")
    pose_ref = store.put(trajectory, payload_ref="memory://phase8-7f3/poses")
    depth_ref = store.put(depth, payload_ref="memory://phase8-7f3/depth")
    evidence_provider = bound_evidence_provider_descriptor()
    geometry_providers = canonical_geometry_provider_descriptors()
    providers = (evidence_provider,) + geometry_providers
    bound = BoundEvidenceProviderExecutor(
        store=store,
        provider_id=evidence_provider.provider_id,
        payload_ref_by_stage={
            ReconstructionPipelineStage.SOURCE_INGESTION: ingestion_ref,
            ReconstructionPipelineStage.CAMERA_POSE: pose_ref,
            ReconstructionPipelineStage.DEPTH_EVIDENCE: depth_ref,
        },
    )

    def quality_builder(dense):
        count = len(dense.point_cloud.points_reconstruction)
        return DenseGeometryEvidence(
            point_count=count,
            depth_relative_residual=np.full(count, 0.01, np.float32),
            view_angle_span_degrees=np.full(count, 28.0, np.float32),
            visibility_ratio=np.full(count, 0.97, np.float32),
            provenance="F3 deterministic integration evidence",
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
        quality_evidence_builder=quality_builder,
    )
    surface_executor = SurfaceMeshProviderExecutor(store=store)

    def observations(_request, surface):
        vertices = surface.mesh.vertices_reconstruction
        regions = (
            (0, "connector_body", "connector-main", np.flatnonzero(vertices[:, 0] < -0.25).astype(np.uint32)),
            (1, "pin", "pin-top", np.flatnonzero((vertices[:, 0] > 0.10) & (vertices[:, 1] < -0.16)).astype(np.uint32)),
            (1, "pin", "pin-bottom", np.flatnonzero((vertices[:, 0] > 0.10) & (vertices[:, 1] > 0.16)).astype(np.uint32)),
        )
        result = []
        for frame_index in (0, 2, 4):
            for class_id, class_name, track_id, indices in regions:
                item = _observation_for_region(
                    surface,
                    camera,
                    frame_index=frame_index,
                    tx=translations[frame_index],
                    class_id=class_id,
                    class_name=class_name,
                    track_id=track_id,
                    target_vertex_indices=indices,
                )
                result.append(replace(item, frame_id=f"f3-{frame_index}"))
        return tuple(result)

    semantic_executor = SemanticGeometryProviderExecutor(
        store=store,
        observation_resolver=observations,
        role_by_class_name={"connector_body": SemanticGeometryRole.CONNECTOR_BODY, "pin": SemanticGeometryRole.CONNECTOR_PIN},
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
        connector = next(
            (obj for obj in semantic.objects if obj.semantic_role is SemanticGeometryRole.CONNECTOR_BODY),
            None,
        )
        if connector is None:
            raise RuntimeError("F3 integration smoke requires one connector-body semantic object")
        return build_reference_assembly(
            surface,
            semantic,
            reference_id="phase8-7f3-self-reference",
            source=ReferenceGeometrySource.GOLDEN_RECONSTRUCTION,
            default_tolerance=tolerance,
            feature_id_by_object_id={connector.object_id: "connector-main"},
            registration_anchor_feature_ids=("connector-main",),
        )

    inspection_executor = InspectionQCProviderExecutor(store=store, reference_resolver=reference_resolver)
    executors = {
        evidence_provider.provider_id: bound,
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
    result = ReconstructionPipelineOrchestrator(plan=plan, providers=providers, executors=executors).run()
    if result.status is ReconstructionPipelineRunStatus.FAILED:
        for stage_result in result.stages:
            if stage_result.reason:
                print(f"{stage_result.stage.value}: {stage_result.reason}", file=sys.stderr)
        return 1
    summary = summarize_f3_pipeline_result(result, store=store)
    artifacts = {item.artifact.kind: item.artifact for item in result.stages if item.artifact is not None}
    dense_payload = store.get(artifacts[ReconstructionArtifactKind.DENSE_GEOMETRY].payload_ref, CanonicalDenseGeometryPayload)
    surface = store.get(artifacts[ReconstructionArtifactKind.SURFACE_MESH].payload_ref, SurfaceMeshResult)
    semantic = store.get(artifacts[ReconstructionArtifactKind.SEMANTIC_GEOMETRY].payload_ref, SemanticGeometryResult)
    inspection = store.get(artifacts[ReconstructionArtifactKind.INSPECTION_REPORT].payload_ref, InspectionReport)
    write_surface_mesh_obj(surface, out / "canonical_surface.obj")
    write_semantic_groups_obj(surface, semantic, out / "semantic_groups.obj")
    write_inspection_report_json(inspection, out / "inspection_report.json")
    write_reconstruction_pipeline_json(result, out / "pipeline_result.json")

    points = surface.mesh.vertices_reconstruction
    fig = plt.figure(figsize=(10, 4.5))
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.scatter(points[:, 0], points[:, 1], s=2)
    for obj in semantic.objects:
        ax1.scatter([obj.centroid_reconstruction[0]], [obj.centroid_reconstruction[1]], s=50)
        ax1.text(obj.centroid_reconstruction[0], obj.centroid_reconstruction[1], obj.semantic_role.value, fontsize=7)
    ax1.set_title("Canonical surface — XY")
    ax1.set_xlabel("X")
    ax1.set_ylabel("Y")
    ax1.set_aspect("equal", adjustable="box")
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.scatter(points[:, 0], points[:, 2], s=2)
    for obj in semantic.objects:
        ax2.scatter([obj.centroid_reconstruction[0]], [obj.centroid_reconstruction[2]], s=50)
    ax2.set_title(f"Canonical surface — XZ | QC={inspection.decision.value}")
    ax2.set_xlabel("X")
    ax2.set_ylabel("Z")
    fig.tight_layout()
    overview = out / "geometry_semantics_overview.png"
    fig.savefig(overview, dpi=160)
    plt.close(fig)

    payload = summary.as_dict()
    payload.update({
        "pass": bool(
            result.status is ReconstructionPipelineRunStatus.COMPLETED
            and summary.semantic_object_count == 3
            and inspection.decision is InspectionDecision.PASS
            and summary.canonical_geometry_authoritative
            and not summary.certified_metrology
        ),
        "dense_quality": dense_payload.quality.summary.as_dict(),
        "surface_obj": str((out / "canonical_surface.obj").resolve()),
        "semantic_obj": str((out / "semantic_groups.obj").resolve()),
        "inspection_report": str((out / "inspection_report.json").resolve()),
        "overview": str(overview.resolve()),
    })
    (out / "summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(f"Phase 8.7F3 integrated geometry/semantics/QC smoke: {'PASS' if payload['pass'] else 'FAIL'}")
    print(f"status={summary.pipeline_status} points={summary.dense_point_count} vertices={summary.mesh_vertex_count} faces={summary.mesh_face_count}")
    print(f"semantic_objects={summary.semantic_object_count} inspection={summary.inspection_decision}")
    print(f"canonical_geometry_artifact={summary.canonical_geometry_artifact_id}")
    print("canonical_geometry_authoritative=True certified_metrology=False")
    print("gaussian_nerf_deferred_to=8.7F4 persistent_scene_deferred_to=8.7F5 reconstruction_ui_enabled=False")
    print(f"overview={overview}")
    print(f"summary={out/'summary.json'}")
    return 0 if payload["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
