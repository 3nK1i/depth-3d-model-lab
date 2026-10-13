from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("PyQt6")

from PyQt6.QtWidgets import QSizePolicy

from app.reconstruction_controller import ReconstructionSourceSelection, ReconstructionWorkflow
from app.reconstruction_processing_jobs import (
    DigitalTwinProcessingJobSpec,
    DigitalTwinProcessingKind,
    begin_digital_twin_processing_snapshot,
)
from app.widgets.reconstruction_page import ReconstructionPage
from uuid import uuid4


def _recorded_video(tmp_path: Path) -> ReconstructionSourceSelection:
    video = tmp_path / "office-tour.mp4"
    video.write_bytes(b"video")
    return ReconstructionSourceSelection(
        workflow=ReconstructionWorkflow.RECORDED_VIDEO,
        ready=True,
        display_name=video.name,
        detail="Recorded video ready",
        path=video.resolve(),
        item_count=1,
    )


def test_e2_build_action_is_progressively_disclosed_and_recorded_video_only(qtbot, tmp_path: Path) -> None:
    page = ReconstructionPage()
    qtbot.addWidget(page)
    assert page._digital_twin_processing_group.expanded is False
    assert page._da3_full_scene_build_button.text() == "Build Full Scene"
    assert page._da3_full_scene_build_button.isEnabled() is False

    class Controller:
        is_shutdown = False
        is_active = False
        source_selection = _recorded_video(tmp_path)

    page._controller = Controller()
    page._refresh_da3_full_scene_controls()
    assert page._da3_full_scene_build_button.isEnabled() is True
    assert "Full Scene" in page._da3_full_scene_value.text()


def test_e2_active_e1_job_disables_full_scene_and_existing_runtime_controls(qtbot, tmp_path: Path) -> None:
    page = ReconstructionPage()
    qtbot.addWidget(page)

    class Controller:
        is_shutdown = False
        is_active = False
        source_selection = _recorded_video(tmp_path)

    page._controller = Controller()
    spec = DigitalTwinProcessingJobSpec(
        job_id=str(uuid4()),
        kind=DigitalTwinProcessingKind.DENSE_GEOMETRY,
        title="Build Full-Scene DA3",
        backend_label="Depth Anything 3",
        runtime_label="external eia-da3",
        output_dir=tmp_path / "out",
        expected_protocol="depth3d.digital-twin.da3-full-scene.v1",
    )
    page._digital_twin_processing_snapshot = begin_digital_twin_processing_snapshot(spec)
    page._refresh_da3_full_scene_controls()
    assert page._da3_full_scene_build_button.isEnabled() is False
    assert "Preparing" in page._da3_full_scene_value.text() or "Active" in page._da3_full_scene_value.text()


def test_e2_full_scene_actions_remain_visible_in_1366_workspace(qtbot, tmp_path: Path) -> None:
    page = ReconstructionPage()
    qtbot.addWidget(page)
    page.resize(1366, 768)

    class Controller:
        is_shutdown = False
        is_active = False
        source_selection = _recorded_video(tmp_path)

    page._controller = Controller()
    page._digital_twin_processing_group.set_expanded(True)
    page._refresh_da3_full_scene_controls()
    page.show()
    qtbot.wait(20)

    assert page._da3_full_scene_build_button.sizePolicy().horizontalPolicy() is QSizePolicy.Policy.Expanding
    assert page._digital_twin_processing_cancel_button.sizePolicy().horizontalPolicy() is QSizePolicy.Policy.Expanding
    assert page._da3_full_scene_build_button.isVisible() is True
    assert page._digital_twin_processing_cancel_button.isVisible() is True
    assert page._da3_full_scene_build_button.width() >= 70
    assert page._digital_twin_processing_cancel_button.width() >= 70
    assert "click Build Full Scene" in page._da3_full_scene_value.text()
