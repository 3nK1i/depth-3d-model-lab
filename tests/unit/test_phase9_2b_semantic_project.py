from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from core.contracts import FrameReference, ImageSize, ModelProvenance, PixelCoordinateSpace, PixelSpaceRole
from core.contracts.validation import ContractValidationError
from core.models.metadata import ConfidenceKind
from core.perception import (
    DenseMaskPayload,
    InstanceMask,
    MaskSource,
    SegmentationProject,
    SegmentationProjectStore,
    SemanticClassOrigin,
    SemanticInstanceNamingPolicy,
    add_project_class,
    create_segmentation_project,
    create_segmentation_project_from_model_names,
    load_segmentation_project,
    merge_project_catalog_into_workspace,
    rename_project_class,
    save_segmentation_project,
    workspace_from_mask_proposals,
)


def _mask() -> InstanceMask:
    frame = FrameReference(7, 1, 0.0, ImageSize(20, 12))
    space = PixelCoordinateSpace("rgb", frame, frame.source_size, PixelSpaceRole.RGB)
    data = np.zeros((12, 20), dtype=bool)
    data[2:9, 3:12] = True
    return InstanceMask(
        mask_id="proposal-1",
        frame_reference=frame,
        pixel_space=space,
        payload=DenseMaskPayload(data),
        source=MaskSource.NATIVE_INSTANCE,
        model_provenance=ModelProvenance("auto-seg", "Auto Segment", "segmentation"),
        confidence=0.8,
        confidence_kind=ConfidenceKind.CONFIDENCE,
    )


def test_2b_project_from_yolo_names_preserves_ids_and_allocates_custom_after_model_range() -> None:
    project = create_segmentation_project_from_model_names(
        "Factory",
        {0: "person", 2: "car"},
        source_model_id="ultralytics-yolo-seg",
        project_id="factory",
        created_at_utc="2026-08-14T09:00:00+00:00",
    )
    assert [(item.class_id, item.class_name, item.origin) for item in project.class_catalog.classes] == [
        (0, "person", SemanticClassOrigin.MODEL),
        (2, "car", SemanticClassOrigin.MODEL),
    ]
    project, robot = add_project_class(project, "robot_arm", updated_at_utc="2026-08-14T10:00:00+00:00")
    assert robot.class_id == 3
    assert robot.origin is SemanticClassOrigin.CUSTOM
    assert project.class_by_name("ROBOT_ARM") == robot


def test_2b_instance_naming_is_human_friendly_and_collision_safe() -> None:
    project = create_segmentation_project("Factory", project_id="factory")
    project, robot = add_project_class(project, "Robot Arm")
    assert project.suggest_instance_name(robot.class_id) == "robot_arm_001"
    assert project.suggest_instance_name(
        robot.class_id,
        existing_names=("robot_arm_001", "ROBOT_ARM_002", "other_001"),
    ) == "robot_arm_003"


def test_2b_custom_class_rename_preserves_stable_id() -> None:
    project = create_segmentation_project("Factory", project_id="factory")
    project, item = add_project_class(project, "cabinet")
    renamed = rename_project_class(project, item.class_id, "control_cabinet")
    definition = renamed.class_by_id(item.class_id)
    assert definition is not None
    assert definition.class_id == item.class_id
    assert definition.class_name == "control_cabinet"
    assert definition.origin is SemanticClassOrigin.CUSTOM


def test_2b_model_class_cannot_be_renamed_in_place() -> None:
    project = create_segmentation_project_from_model_names(
        "Factory", {0: "person"}, source_model_id="yolo", project_id="factory"
    )
    with pytest.raises(ContractValidationError, match="Model-defined classes are immutable"):
        rename_project_class(project, 0, "robot_arm")


def test_2b_project_round_trip_is_deterministic(tmp_path: Path) -> None:
    project = create_segmentation_project_from_model_names(
        "Factory",
        {0: "person"},
        source_model_id="yolo",
        project_id="factory",
        description="Factory semantic catalog",
        created_at_utc="2026-08-14T09:00:00+00:00",
    )
    project, _ = add_project_class(project, "pallet", updated_at_utc="2026-08-14T10:30:00+00:00")
    path = save_segmentation_project(project, tmp_path / "factory.segmentation-project.json")
    loaded = load_segmentation_project(path)
    assert loaded == project
    assert loaded.as_dict() == project.as_dict()


def test_2b_store_uses_safe_hashed_names_and_lists_projects(tmp_path: Path) -> None:
    store = SegmentationProjectStore(tmp_path / "projects")
    unsafe = create_segmentation_project("Unsafe", project_id="../../outside")
    path = store.save(unsafe)
    assert path.parent == store.root
    assert ".." not in path.name
    assert store.contains("../../outside")
    assert store.load("../../outside") == unsafe

    other = create_segmentation_project("Alpha", project_id="alpha")
    store.save(other)
    assert [item.project_name for item in store.list_projects()] == ["Alpha", "Unsafe"]


def test_2b_merge_project_catalog_into_unlabeled_workspace_preserves_mask_and_authority() -> None:
    mask = _mask()
    workspace = workspace_from_mask_proposals((mask,), workspace_id="auto")
    project = create_segmentation_project("Factory", project_id="factory")
    project, robot = add_project_class(project, "robot_arm")
    merged = merge_project_catalog_into_workspace(workspace, project)
    assert merged.class_catalog.by_id(robot.class_id) == robot
    assert merged.objects[0].selected_mask_id == mask.mask_id
    assert merged.objects[0].class_id is None
    assert merged.objects[0].is_geometry_authority is False
    assert merged.objects[0].is_metric_authority is False
    assert merged.objects[0].is_measurement_authority is False


def test_2b_project_never_becomes_geometry_metric_or_measurement_authority() -> None:
    project = create_segmentation_project("Factory", project_id="factory")
    assert project.is_geometry_authority is False
    assert project.is_metric_authority is False
    assert project.is_measurement_authority is False


def test_2b_naming_policy_validation() -> None:
    with pytest.raises(ContractValidationError):
        SemanticInstanceNamingPolicy(separator=" ")
    with pytest.raises(ContractValidationError):
        SemanticInstanceNamingPolicy(zero_pad=0)


def test_2b_catalog_merge_rejects_ambiguous_id_collision() -> None:
    mask = _mask()
    workspace = workspace_from_mask_proposals((mask,), workspace_id="auto")
    model_project = create_segmentation_project_from_model_names(
        "Model", {0: "person"}, source_model_id="yolo", project_id="model",
        created_at_utc="2026-08-14T09:00:00+00:00",
    )
    workspace = merge_project_catalog_into_workspace(workspace, model_project)

    independent = create_segmentation_project(
        "Independent", project_id="independent", created_at_utc="2026-08-14T09:00:00+00:00"
    )
    independent, robot = add_project_class(
        independent, "robot_arm", updated_at_utc="2026-08-14T10:00:00+00:00"
    )
    assert robot.class_id == 0
    with pytest.raises(ContractValidationError, match="already assigned"):
        merge_project_catalog_into_workspace(workspace, independent)
