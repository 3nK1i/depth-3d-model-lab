from __future__ import annotations

import numpy as np

from core.reconstruction import SemanticAssociationPolicy, build_semantic_3d_geometry
from scripts.run_semantic_geometry_smoke import _observation_for_region, _surface


def test_semantic_projection_and_association_is_bounded_and_deterministic() -> None:
    surface, camera, _, translations = _surface(128, 96)
    vertices = surface.mesh.vertices_reconstruction
    x_min = float(np.min(vertices[:, 0]))
    x_max = float(np.max(vertices[:, 0]))
    edges = np.linspace(x_min, x_max, 13)
    observations = []
    expected_objects = 0
    for band in range(len(edges) - 1):
        target = np.flatnonzero(
            (vertices[:, 0] >= edges[band]) & (vertices[:, 0] < edges[band + 1])
        ).astype(np.uint32)
        if len(target) < 12:
            continue
        expected_objects += 1
        for frame_index in (0, 1, 2, 3, 4):
            observations.append(
                _observation_for_region(
                    surface,
                    camera,
                    frame_index=frame_index,
                    tx=translations[frame_index],
                    class_id=20,
                    class_name="scene_entity",
                    track_id=f"entity-{band:02d}",
                    target_vertex_indices=target,
                )
            )

    policy = SemanticAssociationPolicy(
        minimum_projected_vertices=5,
        minimum_shared_vertices=3,
        minimum_cross_view_vertex_iou=0.08,
        maximum_observations=500,
        maximum_objects=64,
    )
    first = build_semantic_3d_geometry(surface, observations, policy=policy)
    second = build_semantic_3d_geometry(surface, observations, policy=policy)

    assert first.statistics.observations_total == len(observations)
    assert first.statistics.observations_rejected == 0
    assert first.statistics.persistent_object_count == expected_objects
    assert first.statistics.persistent_object_count <= policy.maximum_objects
    assert first.statistics.assigned_vertex_count <= surface.mesh.vertex_count
    assert tuple(obj.object_id for obj in first.objects) == tuple(obj.object_id for obj in second.objects)
    for left, right in zip(first.objects, second.objects, strict=True):
        assert np.array_equal(left.vertex_indices, right.vertex_indices)
        assert np.array_equal(left.face_indices, right.face_indices)
        assert np.allclose(left.centroid_reconstruction, right.centroid_reconstruction)
        assert left.certified_metrology is False
