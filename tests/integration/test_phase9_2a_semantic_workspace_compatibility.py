from __future__ import annotations

from dataclasses import replace

import numpy as np

from core.contracts.common import ModelProvenance
from core.perception import (
    PerceptionFrameResult,
    SemanticReviewStatus,
    SemanticWorkspace,
    accepted_observations,
    assign_workspace_class,
    set_workspace_review_status,
    workspace_from_perception_result,
)
from core.reconstruction import SemanticGeometryRole, build_semantic_3d_geometry
from scripts.run_semantic_geometry_smoke import _observation_for_region, _surface


def test_2a_reviewed_custom_class_flows_into_existing_semantic_3d_pipeline() -> None:
    surface, camera, _, translations = _surface(96, 72)
    vertices = surface.mesh.vertices_reconstruction
    target = np.flatnonzero(vertices[:, 0] < -0.25).astype(np.uint32)
    reviewed_frames = []

    provenance = ModelProvenance("ultralytics-yolo-seg", "YOLO Seg", "instance_segmentation")
    for frame_index in (0, 2, 4):
        source = _observation_for_region(
            surface,
            camera,
            frame_index=frame_index,
            tx=translations[frame_index],
            class_id=0,
            class_name="person",
            track_id="machine-1",
            target_vertex_indices=target,
        )
        result = PerceptionFrameResult(
            source.observation.frame_reference,
            (source.observation,),
            (provenance,),
        )
        workspace = workspace_from_perception_result(result, workspace_id=f"ws-{frame_index}")
        catalog, machine = workspace.class_catalog.with_custom_class("robot_arm")
        workspace = SemanticWorkspace(
            workspace.workspace_id,
            workspace.frame_reference,
            catalog,
            workspace.objects,
            workspace.revision,
        )
        workspace = assign_workspace_class(
            workspace,
            source.observation.observation_id,
            machine,
            instance_name="robot_arm_001",
        )
        workspace = set_workspace_review_status(
            workspace,
            source.observation.observation_id,
            SemanticReviewStatus.ACCEPTED,
        )
        reviewed = accepted_observations(workspace)
        assert len(reviewed) == 1
        reviewed_frames.append(replace(source, observation=reviewed[0]))

    semantic = build_semantic_3d_geometry(
        surface,
        reviewed_frames,
        role_by_class_name={"robot_arm": SemanticGeometryRole.DIGITAL_TWIN_ENTITY},
    )
    assert semantic.statistics.observations_accepted == 3
    assert semantic.statistics.persistent_object_count == 1
    assert semantic.objects[0].class_name == "robot_arm"
    assert semantic.objects[0].semantic_role is SemanticGeometryRole.DIGITAL_TWIN_ENTITY
    assert semantic.certified_metrology is False
