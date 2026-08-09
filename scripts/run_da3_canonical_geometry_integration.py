#!/usr/bin/env python3
"""Consume already-qualified F2 DA3 evidence and build F3 canonical geometry."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from uuid import uuid4

import cv2
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.reconstruction import (
    BoundEvidenceProviderExecutor,
    CanonicalDenseGeometryProviderExecutor,
    DenseFusionPolicy,
    DenseGeometryQualityPolicy,
    DenseSceneProfile,
    PipelineIngestionEvidence,
    PipelinePayloadStore,
    ReconstructionArtifactKind,
    ReconstructionPath,
    ReconstructionPipelineGoal,
    ReconstructionPipelineOrchestrator,
    ReconstructionPipelineRequest,
    ReconstructionPipelineRunStatus,
    ReconstructionPipelineStage,
    ReconstructionProviderRole,
    ReconstructionQualityProfile,
    SurfaceMeshProviderExecutor,
    SurfaceMeshResult,
    arbitrary_scale_reference,
    bound_evidence_provider_descriptor,
    build_reconstruction_pipeline_plan,
    canonical_geometry_provider_descriptors,
    da3_camera_profile_from_result,
    da3_supporting_evidence_from_result,
    read_da3_result_manifest,
    validate_da3_npz_result,
    write_reconstruction_pipeline_json,
    write_surface_mesh_obj,
)
from core.reconstruction.da3_provider import read_da3_input_manifest


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--da3-input-manifest", default="/tmp/phase8_7f2_da3_real/input_manifest.json")
    p.add_argument("--da3-result-manifest", default="/tmp/phase8_7f2_da3_real/worker-output/result_manifest.json")
    p.add_argument("--output-dir", default="/tmp/phase8_7f3_da3_canonical")
    p.add_argument("--pixel-stride", type=int, default=4)
    return p.parse_args()


def main() -> int:
    a = parse_args()
    output = Path(a.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    input_manifest = read_da3_input_manifest(a.da3_input_manifest)
    result_manifest = read_da3_result_manifest(a.da3_result_manifest)
    if input_manifest.request_id != result_manifest.request_id:
        raise SystemExit("F2 DA3 input/result request IDs differ")
    arrays = validate_da3_npz_result(result_manifest)
    session_id = str(uuid4())
    frame_ids = tuple(f"da3-f3-{i}" for i in range(result_manifest.frame_count))
    timestamps = tuple(1_000_000_000 + i * 50_000_000 for i in range(result_manifest.frame_count))
    camera = da3_camera_profile_from_result(
        result_manifest,
        reconstruction_frame_id="phase8-7f3-da3-reconstruction",
    )
    rgbs = []
    for path_text in input_manifest.image_paths:
        image = cv2.imread(path_text, cv2.IMREAD_COLOR)
        if image is None:
            raise SystemExit(f"Unable to read DA3 source image: {path_text}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (result_manifest.width, result_manifest.height), interpolation=cv2.INTER_AREA)
        rgbs.append(image)
    scale = arbitrary_scale_reference(profile_id="phase8-7f3-da3-arbitrary")
    ingestion = PipelineIngestionEvidence(
        session_id=session_id,
        frame_ids=frame_ids,
        timestamps_ns=timestamps,
        rgb_frames=tuple(rgbs),
        camera_profile=camera,
        scale_profile=scale,
        reconstruction_frame_id="phase8-7f3-da3-reconstruction",
        scene_profile=DenseSceneProfile.SPATIAL_DIGITAL_TWIN,
        source_label="qualified-f2-da3-real-scene-evidence",
        metadata={
            "da3_source_revision": result_manifest.source_revision,
            "da3_model_id": result_manifest.model_id,
            "da3_model_revision": result_manifest.model_revision,
        },
    )
    trajectory, depth = da3_supporting_evidence_from_result(
        result_manifest,
        session_id=session_id,
        frame_ids=frame_ids,
    )
    store = PipelinePayloadStore()
    ingestion_ref = store.put(ingestion, payload_ref="memory://phase8-7f3-da3/ingestion")
    pose_ref = store.put(trajectory, payload_ref="memory://phase8-7f3-da3/poses")
    depth_ref = store.put(depth, payload_ref="memory://phase8-7f3-da3/depth")

    source_provider = bound_evidence_provider_descriptor(
        provider_id="phase8-f3-real-input-evidence",
        roles=(ReconstructionProviderRole.INGESTION,),
        metadata={"source": "F2 real DA3 input manifest"},
    )
    da3_replay_provider = bound_evidence_provider_descriptor(
        provider_id="depth-anything-3-qualified-evidence",
        provider_version=result_manifest.source_revision[:12],
        roles=(ReconstructionProviderRole.CAMERA_POSE, ReconstructionProviderRole.DEPTH),
        metadata={
            "upstream_provider_id": "depth-anything-3-external",
            "source_revision": result_manifest.source_revision,
            "model_id": result_manifest.model_id,
            "model_revision": result_manifest.model_revision,
            "replayed_qualified_evidence": True,
            "canonical_geometry_authoritative": False,
        },
    )
    geometry = canonical_geometry_provider_descriptors()
    providers = (source_provider, da3_replay_provider, geometry[0], geometry[1])
    source_executor = BoundEvidenceProviderExecutor(
        store=store,
        provider_id=source_provider.provider_id,
        payload_ref_by_stage={ReconstructionPipelineStage.SOURCE_INGESTION: ingestion_ref},
    )
    da3_executor = BoundEvidenceProviderExecutor(
        store=store,
        provider_id=da3_replay_provider.provider_id,
        payload_ref_by_stage={
            ReconstructionPipelineStage.CAMERA_POSE: pose_ref,
            ReconstructionPipelineStage.DEPTH_EVIDENCE: depth_ref,
        },
    )
    median_depth = float(np.median(arrays["depth"]))
    voxel = max(median_depth * 0.02, 1e-4)
    dense_policy = DenseFusionPolicy(
        scene_profile=DenseSceneProfile.SPATIAL_DIGITAL_TWIN,
        voxel_size_reconstruction_units=voxel,
        pixel_stride=max(1, int(a.pixel_stride)),
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
    dense_executor = CanonicalDenseGeometryProviderExecutor(
        store=store,
        policy=dense_policy,
        quality_policy=quality_policy,
    )
    surface_executor = SurfaceMeshProviderExecutor(store=store)
    executors = {
        source_provider.provider_id: source_executor,
        da3_replay_provider.provider_id: da3_executor,
        geometry[0].provider_id: dense_executor,
        geometry[1].provider_id: surface_executor,
    }
    request = ReconstructionPipelineRequest(
        request_id="phase8-7f3-real-da3-canonical",
        session_id=session_id,
        path=ReconstructionPath.OFFLINE,
        quality_profile=ReconstructionQualityProfile.EFFICIENT,
        goals=(ReconstructionPipelineGoal.CANONICAL_GEOMETRY,),
    )
    plan = build_reconstruction_pipeline_plan(request, providers)
    result = ReconstructionPipelineOrchestrator(plan=plan, providers=providers, executors=executors).run()
    if result.status is ReconstructionPipelineRunStatus.FAILED:
        reasons = [item.reason for item in result.stages if item.reason]
        raise SystemExit("F3 DA3 canonical integration failed: " + " | ".join(reasons))
    artifacts = {item.artifact.kind: item.artifact for item in result.stages if item.artifact is not None}
    surface = store.get(artifacts[ReconstructionArtifactKind.SURFACE_MESH].payload_ref, SurfaceMeshResult)
    surface_path = write_surface_mesh_obj(surface, output / "da3_canonical_surface.obj")
    write_reconstruction_pipeline_json(result, output / "pipeline_result.json")
    points = surface.mesh.vertices_reconstruction
    fig = plt.figure(figsize=(10, 4.5))
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.scatter(points[:, 0], points[:, 1], s=1)
    ax1.set_title("DA3 → canonical surface XY")
    ax1.set_aspect("equal", adjustable="box")
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.scatter(points[:, 0], points[:, 2], s=1)
    ax2.set_title("DA3 → canonical surface XZ")
    fig.tight_layout()
    overview = output / "da3_canonical_geometry_overview.png"
    fig.savefig(overview, dpi=160)
    plt.close(fig)
    summary = {
        "phase": "8.7F3",
        "pass": True,
        "pipeline_status": result.status.value,
        "source": "qualified_phase8_7f2_da3_evidence",
        "da3_source_revision": result_manifest.source_revision,
        "da3_model_id": result_manifest.model_id,
        "da3_model_revision": result_manifest.model_revision,
        "frame_count": result_manifest.frame_count,
        "dense_voxel_size_reconstruction_units": voxel,
        "surface_vertex_count": surface.mesh.vertex_count,
        "surface_face_count": surface.mesh.face_count,
        "metric_geometry_status": surface.metric_geometry_status.value,
        "unit_to_metre": surface.unit_to_metre,
        "canonical_geometry_artifact_id": result.canonical_geometry_artifact_id,
        "canonical_geometry_authoritative": True,
        "certified_metrology": False,
        "da3_supporting_evidence_authoritative": False,
        "reconstruction_ui_enabled": False,
        "gaussian_nerf_deferred_to": "8.7F4",
        "surface_obj": str(surface_path),
        "overview": str(overview),
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print("Phase 8.7F3 qualified-DA3 → canonical geometry integration: PASS")
    print(f"frames={result_manifest.frame_count} vertices={surface.mesh.vertex_count} faces={surface.mesh.face_count}")
    print(f"metric_geometry_status={surface.metric_geometry_status.value} unit_to_metre={surface.unit_to_metre}")
    print("da3_supporting_evidence_authoritative=False canonical_geometry_authoritative=True certified_metrology=False")
    print(f"overview={overview}")
    print(f"summary={output/'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
