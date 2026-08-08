from __future__ import annotations

import numpy as np

from core.reconstruction import SemanticGeometryRole, build_semantic_3d_geometry
from scripts.run_semantic_geometry_smoke import _observation_for_region, _surface


def test_surface_perception_to_persistent_semantic_geometry_pipeline() -> None:
    surface, camera, _, translations = _surface(96, 72)
    vertices = surface.mesh.vertices_reconstruction
    body = np.flatnonzero(vertices[:, 0] < -0.25).astype(np.uint32)
    feature = np.flatnonzero((vertices[:, 0] > 0.15) & (np.abs(vertices[:, 1]) < 0.35)).astype(np.uint32)
    observations = []
    for frame_index in (0, 2, 4):
        observations.append(
            _observation_for_region(
                surface,
                camera,
                frame_index=frame_index,
                tx=translations[frame_index],
                class_id=0,
                class_name="connector_body",
                track_id="connector-track" if frame_index != 2 else "",
                target_vertex_indices=body,
            )
        )
        observations.append(
            _observation_for_region(
                surface,
                camera,
                frame_index=frame_index,
                tx=translations[frame_index],
                class_id=1,
                class_name="pin_region",
                track_id="pin-region",
                target_vertex_indices=feature,
            )
        )

    result = build_semantic_3d_geometry(
        surface,
        observations,
        role_by_class_name={
            "connector_body": SemanticGeometryRole.CONNECTOR_BODY,
            "pin_region": SemanticGeometryRole.CONNECTOR_PIN_ROW,
        },
    )

    assert result.session_id == surface.session_id
    assert result.pose_convention == "T_reconstruction_from_camera"
    assert result.statistics.observations_total == 6
    assert result.statistics.observations_accepted == 6
    assert result.statistics.persistent_object_count == 2
    assert result.statistics.assigned_vertex_count > 100
    assert result.statistics.assigned_face_count > 100
    assert result.certified_metrology is False
    objects = {obj.class_name: obj for obj in result.objects}
    assert len(objects["connector_body"].observation_ids) == 3
    assert len(objects["pin_region"].observation_ids) == 3
    assert objects["connector_body"].semantic_role is SemanticGeometryRole.CONNECTOR_BODY
    assert objects["pin_region"].semantic_role is SemanticGeometryRole.CONNECTOR_PIN_ROW
