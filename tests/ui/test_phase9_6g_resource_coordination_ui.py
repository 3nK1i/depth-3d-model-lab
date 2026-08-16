from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import pytest

from app.main_window import MainWindow
from app.reconstruction_digital_twin_resources import build_digital_twin_resource_coordination
from app.reconstruction_processing_jobs import (
    DigitalTwinProcessingJobSpec,
    DigitalTwinProcessingKind,
    begin_digital_twin_processing_snapshot,
    idle_digital_twin_processing_snapshot,
)
from app.widgets.mode_tabs import AppMode
from app.widgets.reconstruction_page import ReconstructionPage
from tests.reconstruction_helpers import make_hardware


def _spec(tmp_path: Path) -> DigitalTwinProcessingJobSpec:
    return DigitalTwinProcessingJobSpec(
        job_id=str(uuid4()),
        kind=DigitalTwinProcessingKind.NERF_APPEARANCE,
        title="Build & Train NeRF",
        backend_label="Nerfstudio Nerfacto",
        runtime_label="external eia-nerf-nerfstudio",
        output_dir=tmp_path / "job",
        expected_protocol="eia.phase9.6e3.nerf-build.v1",
    )


def test_96g_processing_card_exposes_resource_owner_and_gpu_snapshot(qtbot, tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(
        "app.reconstruction_digital_twin_resources.detect_hardware",
        lambda: make_hardware(total_gib=6.0, free_gib=4.0, name="NVIDIA GeForce RTX 2060"),
    )
    page = ReconstructionPage()
    qtbot.addWidget(page)
    assert page._digital_twin_processing_group.expanded is False
    assert "Available" in page._digital_twin_processing_resource_value.text()
    assert "startup 4.0/6.0 GB" in page._digital_twin_processing_gpu_value.text()

    seen: list[bool] = []
    page.digital_twin_resource_activity_changed.connect(seen.append)
    page._on_digital_twin_processing_snapshot_changed(begin_digital_twin_processing_snapshot(_spec(tmp_path)))
    assert page.digital_twin_resource_active is True
    assert "Build job owns external runtime" in page._digital_twin_processing_resource_value.text()
    assert seen and seen[-1] is True
    page._on_digital_twin_processing_snapshot_changed(idle_digital_twin_processing_snapshot())
    assert page.digital_twin_resource_active is False
    assert seen[-1] is False


def test_96g_active_build_blocks_interactive_appearance_without_launching_worker(qtbot, tmp_path: Path) -> None:
    page = ReconstructionPage()
    qtbot.addWidget(page)
    page._on_digital_twin_processing_snapshot_changed(begin_digital_twin_processing_snapshot(_spec(tmp_path)))
    assert "Cancel the active Digital Twin build" in page._interactive_resource_admission_failure("interactive NeRF")
    page._show_interactive_nerf()
    assert page._nerf_worker_client.is_running is False


def test_96g_main_window_treats_dynamic_digital_twin_resource_as_application_owner(qtbot, monkeypatch) -> None:
    monkeypatch.setattr("app.application_controller.CameraEnumerator.start", lambda self: None)
    win = MainWindow()
    qtbot.addWidget(win)
    coordination = build_digital_twin_resource_coordination(
        processing_active=False,
        reconstruction_active=False,
        interactive_nerf_active=True,
        interactive_gaussian_resident=False,
        hardware=make_hardware(total_gib=6.0),
    )
    win._reconstruction_page._digital_twin_resource_coordination = coordination
    win._reconstruction_page._digital_twin_resource_activity_active = True
    win._mode_tabs.set_active_mode(AppMode.RECONSTRUCTION)
    win._on_reconstruction_activity_changed()
    qtbot.wait(10)
    assert "Digital Twin resources are active" in win._reconstruction_conflict_for_application_job("Benchmark")
    assert not win._mode_tabs._buttons[AppMode.BENCHMARK].isEnabled()
    win.close()
