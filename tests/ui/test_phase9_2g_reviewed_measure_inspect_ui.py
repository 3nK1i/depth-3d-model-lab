from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from app.dialogs.auto_segment_workspace_dialog import AutoSegmentWorkspaceDialog
from app.main_window import MainWindow
from app.measurement_inspection_controller import MeasurementInspectionController
from app.widgets.inspection_page import InspectionPage
from core.contracts import FrameReference, ImageSize, ModelProvenance, PixelCoordinateSpace, PixelSpaceRole
from core.perception import (
    DenseMaskPayload,
    InstanceMask,
    MaskSource,
    SemanticClassCatalog,
    SemanticReviewStatus,
    SemanticWorkspace,
    SemanticWorkspaceObject,
    assign_workspace_class,
    set_workspace_review_status,
)
from core.perception.reviewed_analysis import reviewed_selection_from_workspace
from core.perception.semantic_project import SegmentationProjectStore, add_project_class, create_segmentation_project


def _reviewed(image: np.ndarray):
    h, w = image.shape[:2]
    frame = FrameReference(31, 9, 0.0, ImageSize(w, h))
    space = PixelCoordinateSpace("rgb", frame, frame.source_size, PixelSpaceRole.RGB)
    catalog, plant = SemanticClassCatalog().with_custom_class("plant")
    dense = np.zeros((h, w), dtype=bool)
    dense[8 : h - 8, 8 : w // 2] = True
    mask = InstanceMask(
        "plant-mask",
        frame,
        space,
        DenseMaskPayload(dense),
        MaskSource.NATIVE_INSTANCE,
        model_provenance=ModelProvenance("sam2", "SAM2", "automatic_segmentation"),
    )
    workspace = SemanticWorkspace(
        "ui-reviewed",
        frame,
        catalog,
        (SemanticWorkspaceObject("proposal:plant", proposal_mask=mask),),
    )
    workspace = assign_workspace_class(workspace, "proposal:plant", plant, instance_name="plant_001")
    workspace = set_workspace_review_status(workspace, "proposal:plant", SemanticReviewStatus.ACCEPTED)
    return workspace, reviewed_selection_from_workspace(workspace, image)


def test_9_2g_inspection_page_defaults_to_reviewed_masks_and_disables_perception_combo(qtbot) -> None:
    page = InspectionPage()
    qtbot.addWidget(page)
    image = np.zeros((72, 96, 3), dtype=np.uint8)
    _workspace, selection = _reviewed(image)
    page.set_context(source_available=True, depth_model_id="depth-pro", depth_model_label="Depth Pro")
    page.set_captured_frame(image)
    page.set_reviewed_semantic_context(selection, source_label="Auto Segment Project")

    assert page.use_reviewed_semantics_for_analysis is True
    assert page._use_reviewed_semantics.isChecked()
    assert not page._perception_combo.isEnabled()
    assert "1 accepted reviewed object" in page._reviewed_semantic_status.text()
    assert "skip YOLO/perception inference" in page._reviewed_semantic_status.text()

    emitted = []
    page.analysis_requested.connect(emitted.append)
    page._analyze_btn.click()
    assert emitted and emitted[0]["use_reviewed_semantics"] is True

    page._use_reviewed_semantics.setChecked(False)
    assert page._perception_combo.isEnabled()


def test_9_2g_measurement_controller_serializes_reviewed_semantics(qtbot, monkeypatch) -> None:
    controller = MeasurementInspectionController(Path.cwd())
    image = np.zeros((72, 96, 3), dtype=np.uint8)
    _workspace, selection = _reviewed(image)
    captured = {}
    monkeypatch.setattr(controller, "_start_process", lambda args, state: captured.update(args=args, state=state))
    controller.start_analysis(
        image_bgr=image,
        depth_model_id="development-mock",
        depth_config={},
        perception_model_id="ultralytics-yolo-seg",
        depth_profile="efficient",
        perception_profile="efficient",
        max_objects=2,
        reviewed_semantics=selection,
    )
    args = captured["args"]
    assert "--reviewed-semantics" in args
    metadata = Path(args[args.index("--reviewed-semantics") + 1])
    assert metadata.is_file()
    assert "ui-reviewed" in metadata.read_text(encoding="utf-8")
    controller.shutdown()


@pytest.fixture
def window(qtbot, monkeypatch):
    monkeypatch.setattr("app.application_controller.CameraEnumerator.start", lambda self: None)
    win = MainWindow()
    qtbot.addWidget(win)
    yield win
    win.close()


def test_9_2g_main_window_adopts_dialog_review_as_exact_inspection_frame(window, qtbot, tmp_path: Path) -> None:
    image = np.zeros((80, 120, 3), dtype=np.uint8)
    image[15:65, 20:55] = (10, 120, 200)
    store = SegmentationProjectStore(tmp_path / "projects")
    project = create_segmentation_project("Factory", project_id="factory")
    project, plant = add_project_class(project, "plant")
    store.save(project)
    dialog = AutoSegmentWorkspaceDialog(
        window,
        project_store=store,
        initial_bgr=image,
        initial_source_label="Imported reviewed frame",
    )
    qtbot.addWidget(dialog)
    workspace, _selection = _reviewed(image)
    # Rebind the helper workspace to the dialog project catalog while preserving the accepted mask.
    accepted = workspace.accepted_objects[0]
    dialog._project = project
    dialog._workspace = SemanticWorkspace(
        workspace.workspace_id,
        workspace.frame_reference,
        project.class_catalog,
        (accepted,),
        revision=workspace.revision,
    )
    dialog._source_label = "Imported reviewed frame"

    window._adopt_reviewed_semantic_context(dialog)
    assert window._reviewed_semantic_selection is not None
    assert np.array_equal(window._inspection_frame, image)
    assert window._inspection_page.use_reviewed_semantics_for_analysis is True
    assert "Imported reviewed frame" in window._inspection_page._reviewed_semantic_status.text()
