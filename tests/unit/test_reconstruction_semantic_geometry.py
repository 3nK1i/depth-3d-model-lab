from __future__ import annotations

from dataclasses import replace
from functools import lru_cache

import numpy as np
import pytest

from core.contracts import FrameReference, ImageSize, ModelProvenance, PixelCoordinateSpace, PixelSpaceRole
from core.models.metadata import ConfidenceKind
from core.perception import (
    DenseMaskPayload,
    InstanceMask,
    MaskSource,
    ObjectObservation,
    RLEMaskPayload,
)
from core.reconstruction import (
    DenseSceneProfile,
    MetricGeometryStatus,
    SemanticAssociationPolicy,
    SemanticFrameObservation,
    SemanticGeometryRole,
    SemanticProjectionRejection,
    build_semantic_3d_geometry,
    generate_quality_aware_surface_mesh,
    project_surface_mesh_vertices,
)
from core.reconstruction.errors import ReconstructionValidationError
from scripts.run_semantic_geometry_smoke import _observation_for_region, _pose, _surface
from scripts.run_surface_mesh_smoke import _metric_surface


@lru_cache(maxsize=1)
def _fixture():
    return _surface(96, 72)


def _regions(surface):
    vertices = surface.mesh.vertices_reconstruction
    body = np.flatnonzero(vertices[:, 0] < -0.25).astype(np.uint32)
    pin_top = np.flatnonzero((vertices[:, 0] > 0.10) & (vertices[:, 1] < -0.16)).astype(np.uint32)
    pin_bottom = np.flatnonzero((vertices[:, 0] > 0.10) & (vertices[:, 1] > 0.16)).astype(np.uint32)
    return body, pin_top, pin_bottom


def _observations():
    surface, camera, _, translations = _fixture()
    body, top, bottom = _regions(surface)
    values = []
    for frame_index in (0, 2, 4):
        tx = translations[frame_index]
        values.append(
            _observation_for_region(
                surface,
                camera,
                frame_index=frame_index,
                tx=tx,
                class_id=0,
                class_name="connector_body",
                track_id="body" if frame_index != 2 else "",
                target_vertex_indices=body,
            )
        )
        values.append(
            _observation_for_region(
                surface,
                camera,
                frame_index=frame_index,
                tx=tx,
                class_id=1,
                class_name="pin",
                track_id="top",
                target_vertex_indices=top,
            )
        )
        values.append(
            _observation_for_region(
                surface,
                camera,
                frame_index=frame_index,
                tx=tx,
                class_id=1,
                class_name="pin",
                track_id="bottom",
                target_vertex_indices=bottom,
            )
        )
    return tuple(values)


def _roles():
    return {
        "connector_body": SemanticGeometryRole.CONNECTOR_BODY,
        "pin": SemanticGeometryRole.CONNECTOR_PIN,
    }


def _result():
    surface, _, _, _ = _fixture()
    return build_semantic_3d_geometry(surface, _observations(), role_by_class_name=_roles())


def _empty_observation(*, frame_id: int = 0, class_id: int = 9, class_name: str = "empty"):
    surface, camera, _, translations = _fixture()
    ref = FrameReference(87, frame_id, 1.0, camera.camera.image_size)
    observation = ObjectObservation(
        observation_id=f"empty-{frame_id}-{class_id}",
        frame_reference=ref,
        class_id=class_id,
        class_name=class_name,
        masks=(),
    )
    return SemanticFrameObservation(
        frame_id="semantic-0",
        observation=observation,
        camera_profile=camera,
        T_reconstruction_from_camera=_pose(translations[0]),
        pose_layer="optimized_keyframe",
    )


def test_default_semantic_policy_is_bounded() -> None:
    policy = SemanticAssociationPolicy()
    assert policy.minimum_projected_vertices >= 1
    assert policy.maximum_observations > policy.maximum_objects
    assert 0.0 <= policy.minimum_cross_view_vertex_iou <= 1.0


def test_semantic_policy_rejects_invalid_values() -> None:
    with pytest.raises(ReconstructionValidationError):
        SemanticAssociationPolicy(minimum_projected_vertices=0)
    with pytest.raises(ReconstructionValidationError):
        SemanticAssociationPolicy(minimum_cross_view_vertex_iou=1.1)
    with pytest.raises(ReconstructionValidationError):
        SemanticAssociationPolicy(face_minimum_assigned_vertices=4)


def test_mesh_projection_returns_visible_mesh_indices() -> None:
    surface, camera, _, translations = _fixture()
    projection = project_surface_mesh_vertices(surface, camera, _pose(translations[2]))
    assert len(projection.pixel_xy) > 100
    assert len(projection.visible_vertex_indices) == len(projection.pixel_xy)
    assert int(np.max(projection.visible_vertex_indices)) < surface.mesh.vertex_count


def test_semantic_pipeline_builds_three_persistent_objects() -> None:
    result = _result()
    assert result.statistics.persistent_object_count == 3
    assert result.statistics.observations_total == 9
    assert result.statistics.observations_accepted == 9
    assert result.statistics.observations_rejected == 0


def test_untracked_middle_observation_merges_by_cross_view_geometry() -> None:
    result = _result()
    body = next(obj for obj in result.objects if obj.class_name == "connector_body")
    assert len(body.observation_ids) == 3
    assert body.track_ids == ("body",)
    assert body.semantic_role is SemanticGeometryRole.CONNECTOR_BODY


def test_distinct_explicit_track_ids_remain_distinct_objects() -> None:
    result = _result()
    pins = [obj for obj in result.objects if obj.class_name == "pin"]
    assert len(pins) == 2
    assert {obj.track_ids[0] for obj in pins} == {"top", "bottom"}


def test_conflicting_class_for_same_track_is_rejected() -> None:
    values = list(_observations())
    original = values[0]
    conflicting = ObjectObservation(
        observation_id="conflicting-track-class",
        frame_reference=original.observation.frame_reference,
        class_id=8,
        class_name="other",
        masks=original.observation.masks,
        track_id="body",
    )
    values[0] = replace(original, observation=conflicting)
    surface, _, _, _ = _fixture()
    with pytest.raises(ReconstructionValidationError):
        build_semantic_3d_geometry(surface, values, role_by_class_name=_roles())


def test_duplicate_track_in_same_frame_is_rejected() -> None:
    values = list(_observations())
    first = values[1]
    duplicate_obs = ObjectObservation(
        observation_id="same-frame-same-track",
        frame_reference=first.observation.frame_reference,
        class_id=first.observation.class_id,
        class_name=first.observation.class_name,
        masks=first.observation.masks,
        track_id=first.observation.track_id,
    )
    values.append(replace(first, observation=duplicate_obs))
    surface, _, _, _ = _fixture()
    with pytest.raises(ReconstructionValidationError):
        build_semantic_3d_geometry(surface, values, role_by_class_name=_roles())


def test_observation_without_authoritative_mask_is_rejected_not_invented() -> None:
    surface, _, _, _ = _fixture()
    result = build_semantic_3d_geometry(surface, (_empty_observation(),))
    assert result.statistics.observations_rejected == 1
    assert result.statistics.persistent_object_count == 0
    assert result.projections[0].rejection is SemanticProjectionRejection.NO_AUTHORITATIVE_MASK


def test_frame_outside_surface_provenance_is_rejected() -> None:
    surface, camera, _, translations = _fixture()
    body, _, _ = _regions(surface)
    item = _observation_for_region(
        surface,
        camera,
        frame_index=0,
        tx=translations[0],
        class_id=0,
        class_name="connector_body",
        track_id="body",
        target_vertex_indices=body,
    )
    item = replace(item, frame_id="not-a-surface-frame")
    result = build_semantic_3d_geometry(surface, (item,), role_by_class_name=_roles())
    assert result.projections[0].rejection is SemanticProjectionRejection.FRAME_NOT_IN_SURFACE_PROVENANCE


def test_mask_pixel_space_mismatch_is_rejected() -> None:
    surface, camera, _, translations = _fixture()
    body, _, _ = _regions(surface)
    item = _observation_for_region(
        surface,
        camera,
        frame_index=0,
        tx=translations[0],
        class_id=0,
        class_name="connector_body",
        track_id="body",
        target_vertex_indices=body,
    )
    mask = item.observation.selected_mask
    assert mask is not None
    wrong_space = PixelCoordinateSpace(
        "wrong-space",
        mask.frame_reference,
        mask.pixel_space.size,
        PixelSpaceRole.RGB,
    )
    wrong_mask = replace(mask, pixel_space=wrong_space)
    wrong_obs = replace(item.observation, masks=(wrong_mask,), selected_mask_id="")
    result = build_semantic_3d_geometry(surface, (replace(item, observation=wrong_obs),), role_by_class_name=_roles())
    assert result.projections[0].rejection is SemanticProjectionRejection.MASK_PIXEL_SPACE_MISMATCH


def test_unsupported_mask_representation_is_rejected() -> None:
    surface, camera, _, translations = _fixture()
    ref = FrameReference(87, 0, 1.0, camera.camera.image_size)
    space = PixelCoordinateSpace(
        camera.camera.intrinsics.pixel_space_id,
        ref,
        camera.camera.image_size,
        PixelSpaceRole.RGB,
    )
    mask = InstanceMask(
        mask_id="rle-mask",
        frame_reference=ref,
        pixel_space=space,
        payload=RLEMaskPayload(space.size.width, space.size.height, (space.size.width * space.size.height,)),
        source=MaskSource.NATIVE_INSTANCE,
        model_provenance=ModelProvenance("rle", "rle", "instance_segmentation"),
        foreground_valid=True,
    )
    observation = ObjectObservation("rle-obs", ref, 2, "rle", masks=(mask,))
    item = SemanticFrameObservation(
        "semantic-0", observation, camera, _pose(translations[0]), "optimized_keyframe"
    )
    result = build_semantic_3d_geometry(surface, (item,))
    assert result.projections[0].rejection is SemanticProjectionRejection.UNSUPPORTED_MASK_REPRESENTATION


def test_projection_assigns_faces_only_from_masked_vertices() -> None:
    result = _result()
    surface, _, _, _ = _fixture()
    projection = next(value for value in result.projections if value.accepted)
    membership = set(int(value) for value in projection.vertex_indices)
    for face_index in projection.face_indices[:100]:
        face = surface.mesh.faces[int(face_index)]
        assert all(int(vertex) in membership for vertex in face)


def test_semantic_objects_reference_existing_mesh_vertices_and_faces() -> None:
    result = _result()
    surface, _, _, _ = _fixture()
    for obj in result.objects:
        assert int(np.max(obj.vertex_indices)) < surface.mesh.vertex_count
        if len(obj.face_indices):
            assert int(np.max(obj.face_indices)) < surface.mesh.face_count


def test_semantic_geometry_does_not_modify_surface_positions() -> None:
    surface, _, _, _ = _fixture()
    before = surface.mesh.vertices_reconstruction.copy()
    _ = _result()
    assert np.array_equal(surface.mesh.vertices_reconstruction, before)


def test_semantic_object_centroid_and_bounds_are_consistent() -> None:
    result = _result()
    surface, _, _, _ = _fixture()
    for obj in result.objects:
        points = surface.mesh.vertices_reconstruction[obj.vertex_indices.astype(np.int64)]
        assert np.allclose(obj.centroid_reconstruction, np.mean(points, axis=0))
        assert np.allclose(obj.bounds_min_reconstruction, np.min(points, axis=0))
        assert np.allclose(obj.bounds_max_reconstruction, np.max(points, axis=0))


def test_principal_axes_are_right_handed_orthonormal() -> None:
    result = _result()
    for obj in result.objects:
        axes = obj.principal_axes_reconstruction
        assert np.allclose(axes.T @ axes, np.eye(3), atol=1e-5)
        assert np.linalg.det(axes) == pytest.approx(1.0, abs=1e-5)


def test_object_confidence_and_quality_are_bounded() -> None:
    result = _result()
    for obj in result.objects:
        assert 0.0 <= obj.confidence <= 1.0
        assert 0.0 <= obj.mean_vertex_quality <= 1.0
        assert obj.median_vertex_uncertainty_reconstruction_units > 0.0
        assert obj.maximum_vertex_uncertainty_reconstruction_units >= obj.median_vertex_uncertainty_reconstruction_units


def test_arbitrary_semantic_geometry_refuses_metric_conversion() -> None:
    result = _result()
    _, _, scale, _ = _fixture()
    obj = result.objects[0]
    assert obj.metric_geometry_status is MetricGeometryStatus.ARBITRARY
    assert obj.unit_to_metre is None
    with pytest.raises(ReconstructionValidationError):
        obj.centroid_metres(scale)


def test_metric_semantic_object_preserves_provisional_metric_semantics() -> None:
    surface_camera, _, _, _ = _fixture()
    camera = surface_camera  # silence naming confusion; use the same image size below
    del camera
    camera_profile = _surface(96, 72)[1]
    dense, quality, scale = _metric_surface(96, 72, camera_profile)
    surface = generate_quality_aware_surface_mesh(dense, quality)
    vertices = surface.mesh.vertices_reconstruction
    target = np.arange(surface.mesh.vertex_count, dtype=np.uint32)
    item = _observation_for_region(
        surface,
        camera_profile,
        frame_index=0,
        tx=-3.0,
        class_id=4,
        class_name="metric_part",
        track_id="metric-part",
        target_vertex_indices=target,
    )
    # _observation_for_region uses semantic-0, while the metric surface provenance uses metric-0.
    item = replace(item, frame_id="metric-0")
    result = build_semantic_3d_geometry(surface, (item,))
    assert result.metric_geometry_status is MetricGeometryStatus.PROVISIONAL_METRIC
    assert result.unit_to_metre == pytest.approx(0.01)
    obj = result.objects[0]
    assert np.isfinite(obj.centroid_metres(scale)).all()
    assert obj.certified_metrology is False


def test_default_role_does_not_infer_connector_semantics_from_class_name() -> None:
    surface, _, _, _ = _fixture()
    result = build_semantic_3d_geometry(surface, (_observations()[0],))
    assert result.objects[0].semantic_role is SemanticGeometryRole.GENERIC_OBJECT


def test_explicit_role_mapping_accepts_string_values() -> None:
    surface, _, _, _ = _fixture()
    result = build_semantic_3d_geometry(
        surface,
        (_observations()[0],),
        role_by_class_name={"connector_body": "connector_body"},
    )
    assert result.objects[0].semantic_role is SemanticGeometryRole.CONNECTOR_BODY


def test_semantic_observation_capacity_is_enforced() -> None:
    surface, _, _, _ = _fixture()
    values = _observations()[:2]
    with pytest.raises(ReconstructionValidationError):
        build_semantic_3d_geometry(
            surface,
            values,
            policy=SemanticAssociationPolicy(maximum_observations=1),
            role_by_class_name=_roles(),
        )


def test_semantic_object_capacity_is_enforced() -> None:
    surface, _, _, _ = _fixture()
    values = (_observations()[1], _observations()[2])
    with pytest.raises(ReconstructionValidationError):
        build_semantic_3d_geometry(
            surface,
            values,
            policy=SemanticAssociationPolicy(maximum_objects=1),
            role_by_class_name=_roles(),
        )


def test_semantic_statistics_cover_assigned_ambiguous_and_unassigned_vertices() -> None:
    result = _result()
    surface, _, _, _ = _fixture()
    assert result.statistics.assigned_vertex_count + result.statistics.unassigned_vertex_count == surface.mesh.vertex_count
    assert 0 <= result.statistics.ambiguous_vertex_count <= result.statistics.assigned_vertex_count
    assert result.statistics.assigned_face_count > 0


def test_semantic_result_preserves_surface_pose_and_scale_provenance() -> None:
    result = _result()
    surface, _, _, _ = _fixture()
    assert result.session_id == surface.session_id
    assert result.pose_convention == surface.pose_convention == "T_reconstruction_from_camera"
    assert result.reconstruction_frame_id == surface.reconstruction_frame_id
    assert result.scale_profile_id == surface.scale_profile_id
    assert result.metric_geometry_status is surface.metric_geometry_status
    assert result.certified_metrology is False
