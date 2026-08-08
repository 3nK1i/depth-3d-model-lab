#!/usr/bin/env python3
"""Phase 8.7B deterministic semantic 2D-to-3D association smoke."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.contracts import FrameReference, ImageSize, ModelProvenance, PixelCoordinateSpace, PixelSpaceRole
from core.models.metadata import ConfidenceKind
from core.perception import DenseMaskPayload, InstanceMask, MaskSource, ObjectObservation
from core.reconstruction import (
    DenseDepthUnit,
    DenseFusionFrame,
    DenseFusionPolicy,
    DenseGeometryEvidence,
    DenseSceneProfile,
    SemanticFrameObservation,
    SemanticGeometryRole,
    arbitrary_scale_reference,
    assess_dense_geometry_quality,
    build_semantic_3d_geometry,
    estimate_camera_geometry_from_horizontal_fov,
    fuse_dense_frames,
    generate_quality_aware_surface_mesh,
    project_surface_mesh_vertices,
    reconstruction_unit_depth_alignment,
    write_semantic_groups_obj,
)


def _pose(tx: float) -> np.ndarray:
    matrix = np.eye(4, dtype=np.float64)
    matrix[0, 3] = tx
    return matrix


def _surface(width: int, height: int):
    camera = estimate_camera_geometry_from_horizontal_fov(
        profile_id="phase8-7b-estimated-camera",
        camera_id="phase8-7b-camera",
        image_size=ImageSize(width, height),
        horizontal_fov_degrees=72.0,
        focal_length_relative_uncertainty=0.15,
        camera_frame_id="phase8-7b-camera-frame",
        reconstruction_frame_id="phase8-7b-reconstruction",
        notes="Phase 8.7B validation estimated camera",
    )
    scale = arbitrary_scale_reference(profile_id="phase8-7b-arbitrary")
    alignment = reconstruction_unit_depth_alignment(provenance="Phase 8.7B synthetic plane")
    frames = []
    translations = (-0.12, -0.06, 0.0, 0.06, 0.12)
    for index, tx in enumerate(translations):
        rgb = np.zeros((height, width, 3), dtype=np.uint8)
        rgb[..., 0] = np.arange(width, dtype=np.uint8)[None, :]
        rgb[..., 1] = 80 + index * 20
        rgb[..., 2] = 180
        frames.append(
            DenseFusionFrame(
                frame_id=f"semantic-{index}",
                timestamp_ns=1_000_000_000 + index * 50_000_000,
                rgb=rgb,
                depth=np.full((height, width), 2.0, dtype=np.float32),
                T_reconstruction_from_camera=_pose(tx),
                depth_unit=DenseDepthUnit.RECONSTRUCTION_UNITS,
                depth_alignment=alignment,
                confidence=np.full((height, width), 0.96, dtype=np.float32),
                pose_layer="optimized_keyframe" if index in {0, 4} else "optimized_propagated",
            )
        )
    dense = fuse_dense_frames(
        frames,
        session_id="phase8-7b-semantic-smoke",
        camera_profile=camera,
        scale_profile=scale,
        policy=DenseFusionPolicy(
            scene_profile=DenseSceneProfile.OBJECT_INSPECTION,
            voxel_size_reconstruction_units=0.075,
            pixel_stride=2,
            minimum_view_observations=2,
            max_voxels=100_000,
        ),
        reconstruction_frame_id="phase8-7b-reconstruction",
    )
    count = len(dense.point_cloud.points_reconstruction)
    quality = assess_dense_geometry_quality(
        dense,
        camera_profile=camera,
        scale_profile=scale,
        evidence=DenseGeometryEvidence(
            point_count=count,
            depth_relative_residual=np.full(count, 0.01, dtype=np.float32),
            view_angle_span_degrees=np.full(count, 28.0, dtype=np.float32),
            visibility_ratio=np.full(count, 0.97, dtype=np.float32),
            provenance="Phase 8.7B trusted synthetic evidence",
        ),
    )
    surface = generate_quality_aware_surface_mesh(dense, quality)
    return surface, camera, scale, translations


def _perception_provenance() -> ModelProvenance:
    return ModelProvenance(
        model_id="phase8-7b-semantic-mock",
        display_name="Phase 8.7B semantic mock",
        task="instance_segmentation",
        model_version="1",
        device="validation",
    )


def _observation_for_region(
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
        session_id=87,
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
    pose = _pose(tx)
    projection = project_surface_mesh_vertices(surface, camera, pose)
    wanted = np.isin(projection.visible_vertex_indices, target_vertex_indices)
    mask = np.zeros((size.height, size.width), dtype=bool)
    pixels = projection.pixel_xy[wanted]
    if len(pixels):
        x = np.clip(np.rint(pixels[:, 0]).astype(np.int64), 0, size.width - 1)
        y = np.clip(np.rint(pixels[:, 1]).astype(np.int64), 0, size.height - 1)
        mask[y, x] = True
    instance_mask = InstanceMask(
        mask_id=f"mask-{class_name}-{track_id or 'untracked'}-{frame_index}",
        frame_reference=frame_ref,
        pixel_space=pixel_space,
        payload=DenseMaskPayload(mask),
        source=MaskSource.NATIVE_INSTANCE,
        model_provenance=_perception_provenance(),
        confidence=0.96,
        confidence_kind=ConfidenceKind.CONFIDENCE,
    )
    observation = ObjectObservation(
        observation_id=f"obs-{class_name}-{track_id or 'untracked'}-{frame_index}",
        frame_reference=frame_ref,
        class_id=class_id,
        class_name=class_name,
        masks=(instance_mask,),
        track_id=track_id,
    )
    return SemanticFrameObservation(
        frame_id=f"semantic-{frame_index}",
        observation=observation,
        camera_profile=camera,
        T_reconstruction_from_camera=pose,
        pose_layer="optimized_keyframe" if frame_index in {0, 4} else "optimized_propagated",
    )


def main() -> int:
    surface, camera, scale, translations = _surface(96, 72)
    vertices = surface.mesh.vertices_reconstruction
    body = np.flatnonzero(vertices[:, 0] < -0.25).astype(np.uint32)
    pin_top = np.flatnonzero((vertices[:, 0] > 0.10) & (vertices[:, 1] < -0.16)).astype(np.uint32)
    pin_bottom = np.flatnonzero((vertices[:, 0] > 0.10) & (vertices[:, 1] > 0.16)).astype(np.uint32)

    observations = []
    for frame_index in (0, 2, 4):
        tx = translations[frame_index]
        observations.append(
            _observation_for_region(
                surface,
                camera,
                frame_index=frame_index,
                tx=tx,
                class_id=0,
                class_name="connector_body",
                track_id="connector-main" if frame_index != 2 else "",
                target_vertex_indices=body,
            )
        )
        observations.append(
            _observation_for_region(
                surface,
                camera,
                frame_index=frame_index,
                tx=tx,
                class_id=1,
                class_name="pin",
                track_id="pin-top",
                target_vertex_indices=pin_top,
            )
        )
        observations.append(
            _observation_for_region(
                surface,
                camera,
                frame_index=frame_index,
                tx=tx,
                class_id=1,
                class_name="pin",
                track_id="pin-bottom",
                target_vertex_indices=pin_bottom,
            )
        )

    semantic = build_semantic_3d_geometry(
        surface,
        observations,
        role_by_class_name={
            "connector_body": SemanticGeometryRole.CONNECTOR_BODY,
            "pin": SemanticGeometryRole.CONNECTOR_PIN,
        },
    )
    by_track = {
        track_id: obj
        for obj in semantic.objects
        for track_id in obj.track_ids
    }
    connector = by_track.get("connector-main")
    top = by_track.get("pin-top")
    bottom = by_track.get("pin-bottom")

    output_dir = Path("/tmp/phase8_7b_semantic_geometry")
    output_dir.mkdir(parents=True, exist_ok=True)
    semantic_obj = write_semantic_groups_obj(surface, semantic, output_dir / "semantic_groups.obj")

    passed = (
        len(semantic.objects) == 3
        and semantic.statistics.observations_total == 9
        and semantic.statistics.observations_accepted == 9
        and semantic.statistics.observations_rejected == 0
        and connector is not None
        and top is not None
        and bottom is not None
        and len(connector.observation_ids) == 3
        and len(top.observation_ids) == 3
        and len(bottom.observation_ids) == 3
        and connector.semantic_role is SemanticGeometryRole.CONNECTOR_BODY
        and top.semantic_role is SemanticGeometryRole.CONNECTOR_PIN
        and bottom.semantic_role is SemanticGeometryRole.CONNECTOR_PIN
        and len(np.intersect1d(top.vertex_indices, bottom.vertex_indices)) < max(5, int(0.1 * min(len(top.vertex_indices), len(bottom.vertex_indices))))
        and semantic.metric_geometry_status.value == "arbitrary"
        and semantic.unit_to_metre is None
        and semantic.relative_scale_uncertainty is None
        and not semantic.certified_metrology
        and semantic.pose_convention == "T_reconstruction_from_camera"
        and semantic_obj.exists()
        and scale.gauge.unit_to_metre is None
    )

    summary = {
        "phase": "8.7B",
        "pass": passed,
        "semantic": semantic.as_dict(),
        "persistent_track_ids": sorted(by_track),
        "connector_observations": 0 if connector is None else len(connector.observation_ids),
        "pin_top_observations": 0 if top is None else len(top.observation_ids),
        "pin_bottom_observations": 0 if bottom is None else len(bottom.observation_ids),
        "semantic_obj": str(semantic_obj),
        "reconstruction_ui_enabled": False,
        "inspection_qc_enabled": False,
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(f"Phase 8.7B semantic object/connector geometry smoke: {'PASS' if passed else 'FAIL'}")
    print(
        f"observations={semantic.statistics.observations_total} "
        f"accepted={semantic.statistics.observations_accepted} "
        f"persistent_objects={semantic.statistics.persistent_object_count}"
    )
    print(
        f"connector_views={0 if connector is None else len(connector.observation_ids)} "
        f"pin_top_views={0 if top is None else len(top.observation_ids)} "
        f"pin_bottom_views={0 if bottom is None else len(bottom.observation_ids)}"
    )
    print(
        f"assigned_vertices={semantic.statistics.assigned_vertex_count} "
        f"ambiguous_vertices={semantic.statistics.ambiguous_vertex_count}"
    )
    print("association=track_id_plus_cross_view_mesh_overlap")
    print("pose_convention=T_reconstruction_from_camera")
    print("gauge=arbitrary unit_to_metre=None certified_metrology=False")
    print("inspection_qc_enabled=False reconstruction_ui_enabled=False")
    print(f"semantic_obj={semantic_obj}")
    print(f"summary={summary_path}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
