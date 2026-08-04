"""Interactive benchmark page and command-controller tests."""

from __future__ import annotations

from pathlib import Path

import pytest
from PyQt6.QtCore import Qt

from app.benchmark_controller import (
    BenchmarkRunConfiguration,
    BenchmarkRunState,
    build_benchmark_command,
)
from app.widgets.benchmark_page import BenchmarkPage, _MODEL_ID_ROLE
from core.hardware.profiles import ComputeProfile
from core.models.catalog import MOCK_MODEL_ID


@pytest.fixture
def page(qtbot) -> BenchmarkPage:
    widget = BenchmarkPage()
    qtbot.addWidget(widget)
    widget.show()
    return widget


def _item_for_model(page: BenchmarkPage, model_id: str):
    for index in range(page._model_list.count()):
        item = page._model_list.item(index)
        if item.data(_MODEL_ID_ROLE) == model_id:
            return item
    raise AssertionError(f"Model not found: {model_id}")


def test_page_lists_registry_models_and_selects_one_production_model(page) -> None:
    assert page._model_list.count() == 8
    selected = page.selected_model_ids()
    assert len(selected) == 1
    assert selected[0] != MOCK_MODEL_ID


def test_page_emits_complete_configuration_for_mock(page, qtbot, tmp_path: Path) -> None:
    page._clear_selection()
    mock = _item_for_model(page, MOCK_MODEL_ID)
    mock.setCheckState(Qt.CheckState.Checked)
    page._profile_combo.setCurrentIndex(page._profile_combo.findData(ComputeProfile.AUTO))
    page._point_budget.setValue(12_345)
    page._iterations.setValue(2)
    page._sequence_frames.setValue(4)
    page._image_asset.setCurrentIndex(page._image_asset.findData("landscape"))
    page._output_path.setText(str(tmp_path))
    page._smoke_test.setChecked(True)

    with qtbot.waitSignal(page.run_requested, timeout=1000) as signal:
        qtbot.mouseClick(page._run_btn, Qt.MouseButton.LeftButton)

    configuration = signal.args[0]
    assert configuration.model_ids == (MOCK_MODEL_ID,)
    assert configuration.point_budget == 12_345
    assert configuration.iterations == 2
    assert configuration.sequence_frames == 4
    assert configuration.image_asset == "landscape"
    assert configuration.smoke_test is True
    assert configuration.output_dir == tmp_path


def test_runtime_safety_blocks_run_until_live_runtime_is_disconnected(page, qtbot) -> None:
    page.set_runtime_safe(False, "Disconnect Live 3D first.")

    assert not page._run_btn.isEnabled()
    assert page._safety_message.isVisible()
    assert "Disconnect" in page._safety_message.text()

    page.set_runtime_safe(True)
    assert page._run_btn.isEnabled()
    assert not page._safety_message.isVisible()


def test_progress_events_and_results_are_rendered(page, tmp_path: Path) -> None:
    page.handle_event({"event": "benchmark_started", "total_models": 2})
    page.handle_event(
        {
            "event": "model_started",
            "index": 1,
            "total_models": 2,
            "display_name": "Development Mock",
        }
    )
    assert page._progress.maximum() == 2
    assert "Development Mock" in page._progress_label.text()

    page.handle_event(
        {
            "event": "model_completed",
            "index": 1,
            "total_models": 2,
            "display_name": "Development Mock",
            "status": "passed",
        }
    )
    assert page._progress.value() == 1

    payload = {
        "passed": True,
        "results": [
            {
                "model_id": MOCK_MODEL_ID,
                "display_name": "Development Mock",
                "status": "passed",
                "classification": "development",
                "runtime": {"effective_profile": "auto"},
                "timing": {
                    "load_ms": 5.0,
                    "warmup_ms": 2.0,
                    "model_latency_ms": {"mean": 12.5},
                    "fps": 80.0,
                },
                "memory": {"reserved_gib": None, "peak_reserved_gib": None},
                "output": {
                    "resolution": [640, 384],
                    "point_count": 12_345,
                    "depth_scale": "synthetic",
                    "intrinsics_source": "estimated",
                },
            }
        ],
    }
    page.set_results(payload, str(tmp_path))

    assert page._results.rowCount() == 1
    assert page._results.item(0, 0).text() == "Development Mock"
    assert page._results.item(0, 1).text() == "PASSED"
    assert page._results.item(0, 6).text() == "12.5 ms"
    assert page._results.item(0, 10).text() == "640×384"
    assert page._open_output_btn.isEnabled()


def test_running_state_locks_configuration_and_enables_cancel(page) -> None:
    page.apply_state(BenchmarkRunState.RUNNING)

    assert not page._model_list.isEnabled()
    assert not page._run_btn.isEnabled()
    assert page._cancel_btn.isEnabled()

    page.apply_state(BenchmarkRunState.COMPLETED)
    assert page._model_list.isEnabled()
    assert page._run_btn.isEnabled()
    assert not page._cancel_btn.isEnabled()


def test_command_builder_uses_isolated_cli_and_timestamped_run_directory(tmp_path: Path) -> None:
    configuration = BenchmarkRunConfiguration(
        model_ids=(MOCK_MODEL_ID,),
        profile=ComputeProfile.AUTO,
        point_budget=10_000,
        iterations=1,
        sequence_frames=2,
        smoke_test=True,
        output_dir=tmp_path,
    )

    program, arguments, output = build_benchmark_command(configuration)

    assert Path(program).name.startswith("python")
    assert "--progress-jsonl" in arguments
    assert "--include-mock" in arguments
    assert "--smoke-test" in arguments
    assert output.parent == tmp_path
    assert output.name.startswith("run_")


def test_mode_navigation_lock_keeps_only_active_enabled(qtbot) -> None:
    from app.widgets.mode_tabs import AppMode, ModeTabs

    tabs = ModeTabs()
    qtbot.addWidget(tabs)
    tabs.set_active_mode(AppMode.BENCHMARK)
    tabs.set_navigation_locked(True)

    assert tabs._buttons[AppMode.BENCHMARK].isEnabled()
    assert not tabs._buttons[AppMode.LIVE_3D].isEnabled()

    tabs.set_navigation_locked(False)
    assert tabs._buttons[AppMode.BENCHMARK].isEnabled()
    assert tabs._buttons[AppMode.LIVE_3D].isEnabled()
