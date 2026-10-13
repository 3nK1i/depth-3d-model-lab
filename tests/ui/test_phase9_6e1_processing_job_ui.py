from __future__ import annotations

from pathlib import Path
import sys
from types import SimpleNamespace
from uuid import uuid4

import pytest

pytest.importorskip("PyQt6")

from app.reconstruction_processing_jobs import (
    DigitalTwinProcessingCommand,
    DigitalTwinProcessingJobSpec,
    DigitalTwinProcessingKind,
    DigitalTwinProcessingLifecycle,
    begin_digital_twin_processing_snapshot,
    update_digital_twin_processing_snapshot,
)
from app.widgets.reconstruction_page import ReconstructionPage


def _spec(tmp_path: Path) -> DigitalTwinProcessingJobSpec:
    return DigitalTwinProcessingJobSpec(
        job_id=str(uuid4()),
        kind=DigitalTwinProcessingKind.NERF_APPEARANCE,
        title="Build NeRF Appearance",
        backend_label="Nerfstudio Nerfacto",
        runtime_label="eia-nerf-nerfstudio",
        output_dir=tmp_path / "nerf-build",
        expected_protocol="depth3d.digital-twin.processing.test.v1",
        scene_id="scene-e1-ui",
        source_label="persistent scene",
        profile_label="Balanced",
    )


def test_e1_build_processing_group_is_compact_idle_and_non_authoritative(qtbot) -> None:
    page = ReconstructionPage()
    qtbot.addWidget(page)
    assert page._digital_twin_processing_group.expanded is False
    assert page._digital_twin_processing_job_value.text() == "No Digital Twin build job"
    assert page._digital_twin_processing_state_value.text() == "Idle"
    assert page._digital_twin_processing_stage_value.text() == "Waiting for build request"
    assert page._digital_twin_processing_backend_value.text() == "External runtime not active"
    assert page._digital_twin_processing_authority_value.text() == "Canonical geometry remains authoritative"
    assert page._digital_twin_processing_cancel_button.isEnabled() is False
    assert page._digital_twin_processing_progress.value() == 0


def test_e1_running_snapshot_surfaces_progress_runtime_output_and_cancel(qtbot, tmp_path: Path) -> None:
    page = ReconstructionPage()
    qtbot.addWidget(page)
    snapshot = begin_digital_twin_processing_snapshot(_spec(tmp_path))
    snapshot = update_digital_twin_processing_snapshot(
        snapshot,
        lifecycle=DigitalTwinProcessingLifecycle.RUNNING,
        stage="optimization",
        message="Optimizing Nerfacto externally.",
        progress_fraction=0.68,
        elapsed_seconds=1042.0,
        completed_units=6800,
        total_units=10000,
    )
    page._on_digital_twin_processing_snapshot_changed(snapshot)

    assert page._digital_twin_processing_job_value.text() == "Build NeRF Appearance"
    assert page._digital_twin_processing_state_value.text() == "Running"
    assert page._digital_twin_processing_stage_value.text() == "Optimization"
    assert "Nerfstudio Nerfacto" in page._digital_twin_processing_backend_value.text()
    assert "eia-nerf-nerfstudio" in page._digital_twin_processing_backend_value.text()
    assert page._digital_twin_processing_progress.value() == 68
    assert "6800 / 10000" in page._digital_twin_processing_progress.format()
    assert page._digital_twin_processing_elapsed_value.text() == "17:22"
    assert page._digital_twin_processing_cancel_button.isEnabled() is True
    assert "canonical authority unchanged" in page._digital_twin_processing_authority_value.text()


def test_e1_completed_or_failed_job_never_relabels_canonical_authority(qtbot, tmp_path: Path) -> None:
    page = ReconstructionPage()
    qtbot.addWidget(page)
    running = update_digital_twin_processing_snapshot(
        begin_digital_twin_processing_snapshot(_spec(tmp_path)),
        lifecycle=DigitalTwinProcessingLifecycle.RUNNING,
        stage="persist",
        progress_fraction=0.9,
        elapsed_seconds=5.0,
    )
    completed = update_digital_twin_processing_snapshot(
        running,
        lifecycle=DigitalTwinProcessingLifecycle.COMPLETED,
        stage="completed",
        message="Persistent appearance artifact ready.",
        progress_fraction=1.0,
        elapsed_seconds=6.0,
        result_path=str(tmp_path / "nerf-build" / "representation.json"),
    )
    page._on_digital_twin_processing_snapshot_changed(completed)
    assert page._digital_twin_processing_state_value.text() == "Completed"
    assert page._digital_twin_processing_progress.value() == 100
    assert page._digital_twin_processing_cancel_button.isEnabled() is False
    assert "Appearance build only" in page._digital_twin_processing_authority_value.text()
    assert "canonical authority unchanged" in page._digital_twin_processing_authority_value.text()


def test_e1_page_rejects_scene_mismatched_future_adapter_before_process_start(qtbot, tmp_path: Path) -> None:
    page = ReconstructionPage()
    qtbot.addWidget(page)
    page._persistent_scene = SimpleNamespace(scene_id="active-scene")
    spec = _spec(tmp_path)
    command = DigitalTwinProcessingCommand(
        program=Path(sys.executable),
        arguments=("-c", "print('must not run')"),
        working_directory=tmp_path,
    )
    with pytest.raises(ValueError, match="scene_id does not match"):
        page.start_digital_twin_processing_job(spec, command)
    assert page._digital_twin_processing_controller.is_running is False
    assert page.digital_twin_processing_snapshot.lifecycle is DigitalTwinProcessingLifecycle.IDLE
