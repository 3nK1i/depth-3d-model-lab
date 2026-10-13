"""Non-blocking application bridge for Phase 9.3H DA3 canonical integration."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import json
import os
from pathlib import Path
import signal
import sys
from uuid import UUID

from PyQt6.QtCore import QObject, QProcess, QProcessEnvironment, QTimer, pyqtSignal

from core.reconstruction.da3 import DA3_EFFICIENT_MODEL_ID
from core.reconstruction.errors import ReconstructionValidationError

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DA3_CANONICAL_WORKER = PROJECT_ROOT / "scripts" / "run_phase9_3h_da3_canonical_job.py"
DA3_CANONICAL_EVENT_PROTOCOL = "eia.reconstruction.da3-canonical.v1"
DEFAULT_DA3_PYTHON = Path.home() / "anaconda3/envs/eia-da3/bin/python"
DEFAULT_DA3_REPOSITORY = Path.home() / "Desktop/eia-da3-backends/depth-anything-3"


class DA3CanonicalSourceKind(str, Enum):
    IMAGE_DIRECTORY = "image_directory"
    RECORDED_VIDEO = "recorded_video"


class DA3CanonicalRunState(str, Enum):
    IDLE = "idle"
    STARTING = "starting"
    RUNNING = "running"
    CANCELLING = "cancelling"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass(frozen=True, slots=True)
class DA3CanonicalIntegrationConfiguration:
    source_kind: DA3CanonicalSourceKind
    source_path: Path
    output_dir: Path
    da3_python: Path
    da3_repository: Path
    model_id: str = DA3_EFFICIENT_MODEL_ID
    process_res: int = 392
    pixel_stride: int = 4
    max_frames: int = 3
    timeout_seconds: float = 900.0

    def __post_init__(self) -> None:
        if not isinstance(self.source_kind, DA3CanonicalSourceKind):
            raise TypeError("source_kind must be DA3CanonicalSourceKind.")
        source = Path(self.source_path).expanduser().resolve()
        if self.source_kind is DA3CanonicalSourceKind.IMAGE_DIRECTORY:
            if not source.is_dir():
                raise ReconstructionValidationError(f"DA3 image directory does not exist: {source}")
        elif not source.is_file():
            raise ReconstructionValidationError(f"DA3 recorded video does not exist: {source}")
        output = Path(self.output_dir).expanduser().resolve()
        if output.exists() and (not output.is_dir() or any(output.iterdir())):
            raise ReconstructionValidationError("DA3 canonical output directory must be empty.")
        python = Path(self.da3_python).expanduser().resolve()
        repo = Path(self.da3_repository).expanduser().resolve()
        if not python.is_file():
            raise ReconstructionValidationError(f"Qualified DA3 Python is missing: {python}")
        if not repo.is_dir():
            raise ReconstructionValidationError(f"Qualified DA3 checkout is missing: {repo}")
        process_res = int(self.process_res)
        pixel_stride = int(self.pixel_stride)
        max_frames = int(self.max_frames)
        timeout = float(self.timeout_seconds)
        if not 196 <= process_res <= 1008:
            raise ReconstructionValidationError("DA3 process_res must be within [196, 1008].")
        if not 1 <= pixel_stride <= 32:
            raise ReconstructionValidationError("DA3 canonical pixel_stride must be within [1, 32].")
        if not 2 <= max_frames <= 8:
            raise ReconstructionValidationError("DA3 canonical integration supports 2-8 frames per run.")
        if timeout <= 0:
            raise ReconstructionValidationError("DA3 canonical timeout_seconds must be positive.")
        object.__setattr__(self, "source_path", source)
        object.__setattr__(self, "output_dir", output)
        object.__setattr__(self, "da3_python", python)
        object.__setattr__(self, "da3_repository", repo)
        object.__setattr__(self, "process_res", process_res)
        object.__setattr__(self, "pixel_stride", pixel_stride)
        object.__setattr__(self, "max_frames", max_frames)
        object.__setattr__(self, "timeout_seconds", timeout)


def default_da3_python_path() -> Path:
    raw = os.environ.get("DEPTH3D_RECONSTRUCTION_DA3_PYTHON", "").strip()
    if not raw:
        raw = os.environ.get("EIA_RECONSTRUCTION_DA3_PYTHON", "").strip()
    return Path(raw).expanduser().resolve() if raw else DEFAULT_DA3_PYTHON.expanduser().resolve()


def default_da3_repository_path() -> Path:
    raw = os.environ.get("DEPTH3D_RECONSTRUCTION_DA3_REPOSITORY", "").strip()
    if not raw:
        raw = os.environ.get("EIA_RECONSTRUCTION_DA3_REPOSITORY", "").strip()
    return Path(raw).expanduser().resolve() if raw else DEFAULT_DA3_REPOSITORY.expanduser().resolve()


def build_da3_canonical_integration_command(
    configuration: DA3CanonicalIntegrationConfiguration,
    *,
    session_id: str,
    worker_script: Path = DA3_CANONICAL_WORKER,
) -> tuple[str, list[str], Path]:
    if not isinstance(configuration, DA3CanonicalIntegrationConfiguration):
        raise TypeError("configuration must be DA3CanonicalIntegrationConfiguration.")
    worker = Path(worker_script).expanduser().resolve()
    if not worker.is_file():
        raise ReconstructionValidationError(f"DA3 canonical worker is missing: {worker}")
    arguments = [
        str(worker),
        "--session-id", str(session_id),
        "--output-dir", str(configuration.output_dir),
        "--da3-python", str(configuration.da3_python),
        "--da3-repo", str(configuration.da3_repository),
        "--model", configuration.model_id,
        "--process-res", str(configuration.process_res),
        "--pixel-stride", str(configuration.pixel_stride),
        "--max-frames", str(configuration.max_frames),
        "--timeout-seconds", str(configuration.timeout_seconds),
    ]
    if configuration.source_kind is DA3CanonicalSourceKind.IMAGE_DIRECTORY:
        arguments.extend(["--images-dir", str(configuration.source_path)])
    else:
        arguments.extend(["--video", str(configuration.source_path)])
    return sys.executable, arguments, configuration.output_dir


class DA3CanonicalIntegrationController(QObject):
    state_changed = pyqtSignal(object)
    event_received = pyqtSignal(object)
    log_received = pyqtSignal(str)
    result_ready = pyqtSignal(object, str)
    run_finished = pyqtSignal(object)
    error_occurred = pyqtSignal(str)

    def __init__(self, parent: QObject | None = None, *, worker_script: Path = DA3_CANONICAL_WORKER) -> None:
        super().__init__(parent)
        self._worker_script = Path(worker_script).expanduser().resolve()
        self._process = QProcess(self)
        self._process.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        # On Unix, give the depth-side orchestration worker its own process group.
        # The external eia-da3 subprocess inherits that group, so cancellation can
        # terminate the whole job without touching the application process.
        if os.name == "posix" and hasattr(self._process, "setChildProcessModifier"):
            self._process.setChildProcessModifier(os.setsid)
        self._process.readyReadStandardOutput.connect(self._read_output)
        self._process.finished.connect(self._finished)
        self._process.errorOccurred.connect(self._process_error)
        self._kill_timer = QTimer(self)
        self._kill_timer.setSingleShot(True)
        self._kill_timer.setInterval(8000)
        self._kill_timer.timeout.connect(self._kill_if_running)
        self._state = DA3CanonicalRunState.IDLE
        self._buffer = ""
        self._output_dir: Path | None = None
        self._session_id: str | None = None
        self._cancel_requested = False
        self._shutdown_started = False
        self._run_finalized = True

    @property
    def state(self) -> DA3CanonicalRunState:
        return self._state

    @property
    def session_id(self) -> str | None:
        return self._session_id

    @property
    def output_dir(self) -> Path | None:
        return self._output_dir

    @property
    def is_running(self) -> bool:
        return self._process.state() != QProcess.ProcessState.NotRunning

    @property
    def is_active(self) -> bool:
        return self.is_running or not self._run_finalized

    def start(self, configuration: DA3CanonicalIntegrationConfiguration, *, session_id: str) -> str:
        if self._shutdown_started:
            raise ReconstructionValidationError("DA3 canonical integration controller is shut down.")
        if self.is_active:
            raise ReconstructionValidationError("A DA3 canonical integration job is already active.")
        sid = self._canonical_session_id(session_id)
        program, arguments, output_dir = build_da3_canonical_integration_command(
            configuration, session_id=sid, worker_script=self._worker_script
        )
        if output_dir.exists() and any(output_dir.iterdir()):
            raise ReconstructionValidationError("DA3 canonical output directory became non-empty before start.")
        output_dir.mkdir(parents=True, exist_ok=True)
        self._buffer = ""
        self._output_dir = output_dir
        self._session_id = sid
        self._cancel_requested = False
        self._run_finalized = False
        self._set_state(DA3CanonicalRunState.STARTING)
        environment = QProcessEnvironment.systemEnvironment()
        environment.insert("PYTHONUNBUFFERED", "1")
        self._process.setProcessEnvironment(environment)
        self._process.setWorkingDirectory(str(PROJECT_ROOT))
        self._process.setProgram(program)
        self._process.setArguments(arguments)
        self._process.start()
        if not self._process.waitForStarted(3000):
            self._run_finalized = True
            self._set_state(DA3CanonicalRunState.FAILED)
            raise ReconstructionValidationError(
                f"Failed to start DA3 canonical integration worker: {self._process.errorString()}"
            )
        self._set_state(DA3CanonicalRunState.RUNNING)
        return sid

    def cancel(self) -> bool:
        if self._shutdown_started or not self.is_running or self._cancel_requested:
            return False
        self._cancel_requested = True
        self._set_state(DA3CanonicalRunState.CANCELLING)
        self._terminate_process_group()
        self._kill_timer.start()
        return True

    def shutdown(self) -> None:
        if self._shutdown_started:
            return
        self._shutdown_started = True
        if self.is_running:
            self._cancel_requested = True
            self._set_state(DA3CanonicalRunState.CANCELLING)
            self._terminate_process_group()
            if not self._process.waitForFinished(8000):
                self._kill_if_running()
                self._process.waitForFinished(3000)
        self._kill_timer.stop()

    def _set_state(self, state: DA3CanonicalRunState) -> None:
        if state is self._state:
            return
        self._state = state
        self.state_changed.emit(state)

    def _read_output(self) -> None:
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
            payload = json.loads(line)
        except json.JSONDecodeError:
            self.log_received.emit(line)
            return
        if isinstance(payload, dict) and payload.get("protocol") == DA3_CANONICAL_EVENT_PROTOCOL:
            self.event_received.emit(payload)
        else:
            self.log_received.emit(line)

    def _load_result(self) -> dict[str, object] | None:
        if self._output_dir is None:
            return None
        path = self._output_dir / "offline_result.json"
        if not path.is_file():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            self.log_received.emit(f"Could not read DA3 canonical result: {exc}")
            return None
        return payload if isinstance(payload, dict) else None

    def _finished(self, exit_code: int, _status: QProcess.ExitStatus) -> None:
        self._kill_timer.stop()
        if self._buffer.strip():
            self._consume_line(self._buffer.strip())
        self._buffer = ""
        result = self._load_result()
        rejection = self._result_rejection_reason(result)
        if self._cancel_requested:
            state = DA3CanonicalRunState.CANCELLED
        elif exit_code == 0 and result is not None and not rejection and bool(result.get("pass")):
            state = DA3CanonicalRunState.COMPLETED
        else:
            state = DA3CanonicalRunState.FAILED
        if rejection and not self._cancel_requested:
            self.error_occurred.emit(f"DA3 canonical result rejected: {rejection}")
        elif state is DA3CanonicalRunState.FAILED and not self._cancel_requested:
            detail = (
                str(result.get("error", "")).strip()
                if result is not None
                else f"DA3 canonical worker produced no readable result (exit {int(exit_code)})."
            )
            self.error_occurred.emit(detail or f"DA3 canonical integration failed (exit {int(exit_code)}).")
        if state is DA3CanonicalRunState.COMPLETED and result is not None and self._output_dir is not None:
            self.result_ready.emit(result, str(self._output_dir))
        self._set_state(state)
        self._run_finalized = True
        self.run_finished.emit({
            "state": state,
            "exit_code": int(exit_code),
            "session_id": self._session_id or "",
            "output_dir": str(self._output_dir or ""),
            "result": result,
            "result_accepted": state is DA3CanonicalRunState.COMPLETED,
            "result_rejection_reason": rejection,
        })

    def _process_error(self, _error: QProcess.ProcessError) -> None:
        if not self._cancel_requested and not self._shutdown_started:
            message = self._process.errorString()
            self.log_received.emit(f"DA3 canonical process error: {message}")
            self.error_occurred.emit(message)

    def _result_rejection_reason(self, result: dict[str, object] | None) -> str:
        if result is None or self._session_id is None:
            return "" if result is None else "controller has no active session identity"
        try:
            actual = self._canonical_session_id(result.get("session_id", ""))
        except ReconstructionValidationError as exc:
            return str(exc)
        if actual != self._session_id:
            return f"result session_id mismatch ({actual} != {self._session_id})"
        if bool(result.get("certified_metrology", False)):
            return "DA3 canonical result unexpectedly claims certified metrology"
        if bool(result.get("persistent_scene_ready", False)) and not bool(
            result.get("canonical_geometry_authoritative", False)
        ):
            return "persistent DA3 result lacks canonical geometry authority"
        return ""

    def _terminate_process_group(self) -> None:
        """Request bounded cancellation of the worker and its inherited DA3 child."""

        if not self.is_running:
            return
        pid = int(self._process.processId())
        if pid > 0 and os.name == "posix" and hasattr(os, "killpg"):
            try:
                os.killpg(pid, signal.SIGTERM)
                return
            except (ProcessLookupError, PermissionError, OSError):
                pass
        self._process.terminate()

    def _kill_if_running(self) -> None:
        if not self.is_running:
            return
        pid = int(self._process.processId())
        if pid > 0 and os.name == "posix" and hasattr(os, "killpg"):
            try:
                os.killpg(pid, signal.SIGKILL)
                return
            except (ProcessLookupError, PermissionError, OSError):
                pass
        self._process.kill()

    @staticmethod
    def _canonical_session_id(value: object) -> str:
        try:
            return str(UUID(str(value).strip()))
        except (ValueError, AttributeError, TypeError) as exc:
            raise ReconstructionValidationError("DA3 canonical session_id must be a valid UUID.") from exc


__all__ = [
    "DA3CanonicalIntegrationConfiguration",
    "DA3CanonicalIntegrationController",
    "DA3CanonicalRunState",
    "DA3CanonicalSourceKind",
    "build_da3_canonical_integration_command",
    "default_da3_python_path",
    "default_da3_repository_path",
]
