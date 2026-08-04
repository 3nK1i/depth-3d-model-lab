"""Non-blocking Qt controller for the registry-driven benchmark CLI."""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

from PyQt6.QtCore import QObject, QProcess, QProcessEnvironment, QTimer, pyqtSignal

from core.benchmark.events import parse_event_line
from core.hardware.profiles import ComputeProfile
from core.models.registry import get_model_descriptor

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_SCRIPT = PROJECT_ROOT / "scripts" / "run_phase6_benchmark.py"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "phase7" / "benchmark"


class BenchmarkRunState(Enum):
    IDLE = "idle"
    STARTING = "starting"
    RUNNING = "running"
    CANCELLING = "cancelling"
    COMPLETED = "completed"
    COMPLETED_WITH_ERRORS = "completed_with_errors"
    CANCELLED = "cancelled"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class BenchmarkRunConfiguration:
    model_ids: tuple[str, ...]
    profile: ComputeProfile = ComputeProfile.AUTO
    point_budget: int = 100_000
    iterations: int = 3
    sequence_frames: int = 12
    image_asset: str = "registry"
    sequence_asset: str = "registry"
    require_all: bool = False
    smoke_test: bool = False
    output_dir: Path | None = None

    def resolved_output_dir(self) -> Path:
        base = (
            Path(self.output_dir).expanduser().resolve()
            if self.output_dir is not None
            else DEFAULT_OUTPUT_ROOT.resolve()
        )
        stamp = datetime.now().strftime("run_%Y%m%d_%H%M%S_%f")
        return base / stamp

    def validate(self) -> None:
        if not self.model_ids:
            raise ValueError("Select at least one model to benchmark.")
        for model_id in self.model_ids:
            get_model_descriptor(model_id)
        if self.point_budget < 1:
            raise ValueError("Point budget must be at least 1.")
        if self.iterations < 1:
            raise ValueError("Iterations must be at least 1.")
        if self.sequence_frames < 1:
            raise ValueError("Sequence frames must be at least 1.")
        if self.image_asset not in {"registry", "landscape", "portrait"}:
            raise ValueError(f"Unsupported image benchmark asset: {self.image_asset!r}")
        if self.sequence_asset not in {"registry", "temporal"}:
            raise ValueError(f"Unsupported sequence benchmark asset: {self.sequence_asset!r}")


def build_benchmark_command(
    configuration: BenchmarkRunConfiguration,
) -> tuple[str, list[str], Path]:
    """Return the Python executable, arguments, and resolved output path."""
    configuration.validate()
    output_dir = configuration.resolved_output_dir()
    include_mock = any(
        get_model_descriptor(model_id).development_only
        for model_id in configuration.model_ids
    )
    arguments = [
        str(BENCHMARK_SCRIPT),
        "--models",
        ",".join(configuration.model_ids),
        "--profile",
        configuration.profile.value,
        "--point-budget",
        str(int(configuration.point_budget)),
        "--iterations",
        str(int(configuration.iterations)),
        "--sequence-frames",
        str(int(configuration.sequence_frames)),
        "--image-asset",
        configuration.image_asset,
        "--sequence-asset",
        configuration.sequence_asset,
        "--output-dir",
        str(output_dir),
        "--progress-jsonl",
    ]
    if include_mock:
        arguments.append("--include-mock")
    if configuration.require_all:
        arguments.append("--require-all")
    if configuration.smoke_test:
        arguments.append("--smoke-test")
    return sys.executable, arguments, output_dir


class BenchmarkController(QObject):
    """Run the benchmark CLI in a QProcess and expose structured progress."""

    state_changed = pyqtSignal(object)
    event_received = pyqtSignal(object)
    log_received = pyqtSignal(str)
    results_ready = pyqtSignal(object, str)
    run_finished = pyqtSignal(object)

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._process = QProcess(self)
        self._process.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        self._process.readyReadStandardOutput.connect(self._on_ready_read)
        self._process.finished.connect(self._on_finished)
        self._process.errorOccurred.connect(self._on_process_error)
        self._state = BenchmarkRunState.IDLE
        self._buffer = ""
        self._output_dir: Path | None = None
        self._cancel_requested = False
        self._kill_timer = QTimer(self)
        self._kill_timer.setSingleShot(True)
        self._kill_timer.setInterval(8000)
        self._kill_timer.timeout.connect(self._kill_if_running)

    @property
    def state(self) -> BenchmarkRunState:
        return self._state

    @property
    def is_running(self) -> bool:
        return self._process.state() != QProcess.ProcessState.NotRunning

    @property
    def output_dir(self) -> Path | None:
        return self._output_dir

    def start(self, configuration: BenchmarkRunConfiguration) -> None:
        if self.is_running:
            raise RuntimeError("A benchmark is already running.")
        program, arguments, output_dir = build_benchmark_command(configuration)
        output_dir.mkdir(parents=True, exist_ok=True)
        self._buffer = ""
        self._output_dir = output_dir
        self._cancel_requested = False
        self._set_state(BenchmarkRunState.STARTING)

        environment = QProcessEnvironment.systemEnvironment()
        environment.insert("PYTHONUNBUFFERED", "1")
        self._process.setProcessEnvironment(environment)
        self._process.setWorkingDirectory(str(PROJECT_ROOT))
        self._process.setProgram(program)
        self._process.setArguments(arguments)
        self._process.start()
        if not self._process.waitForStarted(3000):
            self._set_state(BenchmarkRunState.FAILED)
            raise RuntimeError(
                f"Failed to start benchmark process: {self._process.errorString()}"
            )
        self._set_state(BenchmarkRunState.RUNNING)

    def cancel(self) -> None:
        if not self.is_running or self._cancel_requested:
            return
        self._cancel_requested = True
        self._set_state(BenchmarkRunState.CANCELLING)
        self.log_received.emit("Cancellation requested; stopping the active model safely…")
        self._process.terminate()
        self._kill_timer.start()

    def shutdown(self) -> None:
        if not self.is_running:
            return
        self._cancel_requested = True
        self._process.terminate()
        if not self._process.waitForFinished(8000):
            self._process.kill()
            self._process.waitForFinished(3000)

    def _on_ready_read(self) -> None:
        raw = bytes(self._process.readAllStandardOutput()).decode("utf-8", errors="replace")
        if not raw:
            return
        self._buffer += raw
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            self._consume_line(line.rstrip("\r"))

    def _consume_line(self, line: str) -> None:
        if not line:
            return
        try:
            event = parse_event_line(line)
        except (ValueError, json.JSONDecodeError) as exc:
            self.log_received.emit(f"Malformed benchmark event: {exc}")
            return
        if event is None:
            self.log_received.emit(line)
            return
        self.event_received.emit(event)

    def _on_finished(self, exit_code: int, _exit_status: QProcess.ExitStatus) -> None:
        self._kill_timer.stop()
        if self._buffer.strip():
            self._consume_line(self._buffer.strip())
        self._buffer = ""

        payload = self._load_results()
        if payload is not None and self._output_dir is not None:
            self.results_ready.emit(payload, str(self._output_dir))

        cancelled = self._cancel_requested or bool((payload or {}).get("cancelled"))
        if cancelled:
            state = BenchmarkRunState.CANCELLED
        elif payload is None:
            state = BenchmarkRunState.FAILED
        elif bool(payload.get("passed")) and exit_code == 0:
            state = BenchmarkRunState.COMPLETED
        else:
            state = BenchmarkRunState.COMPLETED_WITH_ERRORS
        self._set_state(state)
        self.run_finished.emit(
            {
                "state": state,
                "exit_code": int(exit_code),
                "output_dir": str(self._output_dir or ""),
                "payload": payload,
            }
        )

    def _on_process_error(self, _error: QProcess.ProcessError) -> None:
        if self._process.state() == QProcess.ProcessState.NotRunning and not self._cancel_requested:
            self.log_received.emit(f"Benchmark process error: {self._process.errorString()}")

    def _load_results(self) -> dict[str, Any] | None:
        if self._output_dir is None:
            return None
        path = self._output_dir / "benchmark_results.json"
        if not path.is_file():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            self.log_received.emit(f"Could not read benchmark report: {exc}")
            return None
        return payload if isinstance(payload, dict) else None

    def _kill_if_running(self) -> None:
        if self.is_running:
            self.log_received.emit("Benchmark did not stop in time; forcing process termination.")
            self._process.kill()

    def _set_state(self, state: BenchmarkRunState) -> None:
        if state is self._state:
            return
        self._state = state
        self.state_changed.emit(state)
