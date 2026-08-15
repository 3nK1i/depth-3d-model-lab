"""Phase 9.3H persistence bridge for DA3-derived canonical geometry.

The bridge persists the existing canonical dense/surface output together with
explicit DA3 pose/depth provenance.  Semantic and inspection stages are recorded
as unavailable, never fabricated.  Calibration is optional and no metric scale
is invented.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from app.reconstruction_canonical_cameras import build_canonical_camera_trajectory_snapshot
from core.reconstruction.da3 import DA3_PROVIDER_ID
from core.reconstruction.da3_canonical_scene import DA3CanonicalSceneIntegrationResult
from core.reconstruction.persistent_scene import (
    PersistentReconstructionScene,
    persistent_scene_from_pipeline_result,
    write_persistent_reconstruction_scene,
)
from core.reconstruction.surface_mesh import write_surface_mesh_obj


def materialize_da3_canonical_persistent_scene(
    integration: DA3CanonicalSceneIntegrationResult,
    output_dir: str | Path,
    *,
    input_manifest_path: str | Path = "",
    result_manifest_path: str | Path = "",
) -> tuple[PersistentReconstructionScene, Path, Path]:
    """Write the canonical OBJ + persistent scene without changing authority."""

    if not isinstance(integration, DA3CanonicalSceneIntegrationResult):
        raise TypeError("integration must be DA3CanonicalSceneIntegrationResult.")
    root = Path(output_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    surface_path = write_surface_mesh_obj(
        integration.surface,
        root / "da3_canonical_surface.obj",
    )
    manifest = integration.da3_manifest
    input_path = Path(input_manifest_path).expanduser().resolve() if str(input_manifest_path).strip() else None
    result_path = Path(result_manifest_path).expanduser().resolve() if str(result_manifest_path).strip() else None

    camera_snapshot = build_canonical_camera_trajectory_snapshot(
        matrices_reconstruction=integration.trajectory.T_reconstruction_from_camera,
        camera_ids=integration.frame_ids,
        provider_id=DA3_PROVIDER_ID,
        pose_layers=integration.trajectory.pose_layers,
        reconstruction_frame_id=integration.reconstruction_frame_id,
        source_frame_sha256=integration.source_frame_sha256,
    )
    snapshots: dict[str, dict[str, Any]] = {
        "ingestion_result": {
            "session_id": integration.ingestion.session_id,
            "frame_ids": list(integration.frame_ids),
            "frame_count": len(integration.frame_ids),
            "source_frame_sha256": list(integration.source_frame_sha256),
            "source_label": integration.ingestion.source_label,
            "reconstruction_frame_id": integration.reconstruction_frame_id,
            "calibration_required": False,
            "metric_scale_qualified": False,
            "certified_metrology": False,
        },
        "camera_trajectory": camera_snapshot,
        "depth_evidence": {
            "session_id": integration.depth.session_id,
            "frame_ids": list(integration.frame_ids),
            "frame_count": len(integration.frame_ids),
            "source_provider_id": integration.depth.source_provider_id,
            "geometry_scale": integration.depth.geometry_scale,
            "model_id": manifest.model_id,
            "model_revision": manifest.model_revision,
            "source_revision": manifest.source_revision,
            "da3_output_npz_path": manifest.output_npz_path,
            "da3_input_manifest_path": str(input_path) if input_path is not None else "",
            "da3_result_manifest_path": str(result_path) if result_path is not None else "",
            "canonical_geometry_authoritative": False,
            "metric_scale_qualified": False,
            "certified_metrology": False,
        },
        "dense_geometry": {
            "point_count": int(len(integration.canonical_dense.dense.point_cloud.points_reconstruction)),
            "metric_geometry_status": integration.canonical_dense.dense.metric_geometry_status.value,
            "scale_profile_id": integration.canonical_dense.dense.scale_profile_id,
            "source_depth_provider_id": integration.depth.source_provider_id,
            "da3_supporting_evidence": True,
            "canonical_geometry_authoritative": True,
            "certified_metrology": False,
        },
        "surface_mesh": {
            "surface_obj_path": str(surface_path),
            "surface_vertex_count": integration.surface.mesh.vertex_count,
            "surface_face_count": integration.surface.mesh.face_count,
            "metric_geometry_status": integration.surface.metric_geometry_status.value,
            "unit_to_metre": integration.surface.unit_to_metre,
            "source_depth_provider_id": integration.depth.source_provider_id,
            "canonicalized_from_da3_supporting_evidence": True,
            "canonical_geometry_authoritative": True,
            "certified_metrology": False,
        },
        "semantic_geometry": {
            "session_id": integration.ingestion.session_id,
            "available": False,
            "unavailable_reason": "Semantic context was not run for this new DA3 canonical scene.",
            "object_count": 0,
            "observations_total": 0,
            "observations_accepted": 0,
            "fabricated": False,
            "canonical_geometry_authoritative": False,
            "certified_metrology": False,
        },
        "inspection_report": {
            "session_id": integration.ingestion.session_id,
            "available": False,
            "unavailable_reason": "Inspection/QC was not run for this new DA3 canonical scene.",
            "decision": "not_evaluated",
            "fabricated": False,
            "canonical_geometry_authoritative": False,
            "certified_metrology": False,
        },
    }
    scene = persistent_scene_from_pipeline_result(
        integration.pipeline_result,
        payload_snapshots_by_kind=snapshots,
    )
    scene_manifest = write_persistent_reconstruction_scene(scene, root)
    return scene, scene_manifest, surface_path


__all__ = ["materialize_da3_canonical_persistent_scene"]
