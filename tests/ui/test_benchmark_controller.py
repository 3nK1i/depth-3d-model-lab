"""QProcess lifecycle tests for the interactive benchmark controller."""

from __future__ import annotations

from pathlib import Path

from app.benchmark_controller import (
    BenchmarkController,
    BenchmarkRunConfiguration,
    BenchmarkRunState,
)
from core.hardware.profiles import ComputeProfile
from core.models.catalog import MOCK_MODEL_ID


def test_controller_runs_mock_benchmark_without_blocking_qt(qtbot, tmp_path: Path) -> None:
    controller = BenchmarkController()
    configuration = BenchmarkRunConfiguration(
        model_ids=(MOCK_MODEL_ID,),
        profile=ComputeProfile.AUTO,
        point_budget=10_000,
        iterations=1,
        sequence_frames=2,
        smoke_test=True,
        output_dir=tmp_path,
    )

    with qtbot.waitSignal(controller.run_finished, timeout=30_000) as signal:
        controller.start(configuration)

    result = signal.args[0]
    assert result["state"] is BenchmarkRunState.COMPLETED
    assert result["payload"]["passed"] is True
    assert result["payload"]["results"][0]["model_id"] == MOCK_MODEL_ID
    assert (Path(result["output_dir"]) / "benchmark_results.json").is_file()


def test_controller_cancel_terminates_external_process(qtbot, tmp_path: Path, monkeypatch) -> None:
    script = tmp_path / "slow_benchmark.py"
    script.write_text(
        "import time\nprint('slow benchmark started', flush=True)\ntime.sleep(30)\n",
        encoding="utf-8",
    )
    monkeypatch.setattr("app.benchmark_controller.BENCHMARK_SCRIPT", script)
    controller = BenchmarkController()
    configuration = BenchmarkRunConfiguration(
        model_ids=(MOCK_MODEL_ID,),
        smoke_test=True,
        output_dir=tmp_path,
    )

    with qtbot.waitSignal(controller.run_finished, timeout=10_000) as signal:
        controller.start(configuration)
        qtbot.waitUntil(lambda: controller.state is BenchmarkRunState.RUNNING, timeout=3000)
        controller.cancel()

    assert signal.args[0]["state"] is BenchmarkRunState.CANCELLED
    assert not controller.is_running
