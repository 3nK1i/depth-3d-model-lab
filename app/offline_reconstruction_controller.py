"""Non-blocking offline reconstruction process bridge, hardened through Phase 8.8A5."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import json
from math import isfinite
import os
from pathlib import Path
import signal
import sys
from uuid import UUID

from PyQt6.QtCore import QObject, QProcess, QProcessEnvironment, QTimer, pyqtSignal

from core.reconstruction import (
    ReconstructionQualityProfile,
    SfmMatchingStrategy,
    new_reconstruction_session_id,
)
from core.reconstruction.errors import ReconstructionValidationError, SfmWorkerProtocolError
from core.reconstruction.reconstruction_progressive_sfm_contract import (
    ProgressiveSfmSnapshot,
    ProgressiveSfmStage,
)
from core.reconstruction.reconstruction_progressive_sfm_transport import (
    parse_progressive_sfm_worker_line,
)
from app.reconstruction_progressive_sfm_state import (
    ProgressiveSfmOfferDisposition,
    ProgressiveSfmSnapshotBuffer,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OFFLINE_RECONSTRUCTION_WORKER = PROJECT_ROOT / "scripts" / "run_reconstruction_offline_job.py"
OFFLINE_EVENT_PROTOCOL = "eia.reconstruction.offline.v1"
PROGRESSIVE_SFM_COALESCE_INTERVAL_MS = 50


class OfflineReconstructionSourceKind(str, Enum):
    IMAGE_DIRECTORY = "image_directory"
    IMAGE_PATHS = "image_paths"
    RECORDED_VIDEO = "recorded_video"


class OfflineReconstructionRunState(str, Enum):
    IDLE = "idle"
    STARTING = "starting"
    RUNNING = "running"
    CANCELLING = "cancelling"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass(frozen=True, slots=True)
class OfflineReconstructionConfiguration:
    """Application-facing input for one production offline reconstruction attempt."""

    source_kind: OfflineReconstructionSourceKind
    inputs: tuple[Path, ...]
    output_dir: Path
    sfm_python: Path
    profile: ReconstructionQualityProfile = ReconstructionQualityProfile.EFFICIENT
    matching: SfmMatchingStrategy = SfmMatchingStrategy.SEQUENTIAL
    max_keyframes: int | None = None
    maximum_image_size: int | None = None
    sequential_overlap: int | None = None
    video_target_fps: float = 2.0
    image_sequence_fps: float = 10.0
    recursive: bool = False
    use_gpu: bool = True
    timeout_minutes: float = 60.0

    def __post_init__(self) -> None:
        if not isinstance(self.source_kind, OfflineReconstructionSourceKind):
            raise TypeError("source_kind must be OfflineReconstructionSourceKind.")
        normalized = tuple(Path(item).expanduser().resolve() for item in self.inputs)
        if not normalized:
            raise ReconstructionValidationError("Offline reconstruction requires an input source.")
        if self.source_kind in {
            OfflineReconstructionSourceKind.IMAGE_DIRECTORY,
            OfflineReconstructionSourceKind.RECORDED_VIDEO,
        } and len(normalized) != 1:
            raise ReconstructionValidationError(
                f"{self.source_kind.value} requires exactly one input path."
            )
        if self.source_kind is OfflineReconstructionSourceKind.IMAGE_DIRECTORY:
            if not normalized[0].is_dir():
                raise ReconstructionValidationError(
                    f"Image-sequence directory does not exist: {normalized[0]}"
                )
        else:
            missing = [str(path) for path in normalized if not path.is_file()]
            if missing:
                raise ReconstructionValidationError(f"Offline input file does not exist: {missing[0]}")
        if self.source_kind is OfflineReconstructionSourceKind.IMAGE_PATHS and len(set(normalized)) != len(normalized):
            raise ReconstructionValidationError("Explicit image paths must be unique.")

        output_dir = Path(self.output_dir).expanduser().resolve()
        if output_dir.exists():
            if not output_dir.is_dir():
                raise ReconstructionValidationError(
                    f"Offline output path is not a directory: {output_dir}"
                )
            if any(output_dir.iterdir()):
                raise ReconstructionValidationError(
                    "Offline output directory must be empty to prevent stale-result reuse."
                )
        sfm_python = Path(self.sfm_python).expanduser().resolve()
        if not sfm_python.is_file():
            raise ReconstructionValidationError(
                f"External PyCOLMAP Python executable does not exist: {sfm_python}"
            )
        if not isinstance(self.profile, ReconstructionQualityProfile):
            raise TypeError("profile must be ReconstructionQualityProfile.")
        if not isinstance(self.matching, SfmMatchingStrategy):
            raise TypeError("matching must be SfmMatchingStrategy.")
        if self.max_keyframes is not None:
            if isinstance(self.max_keyframes, bool) or int(self.max_keyframes) < 3:
                raise ReconstructionValidationError("max_keyframes must be at least 3 when provided.")
        if self.maximum_image_size is not None and (
            isinstance(self.maximum_image_size, bool) or int(self.maximum_image_size) <= 0
        ):
            raise ReconstructionValidationError("maximum_image_size must be positive when provided.")
        if self.sequential_overlap is not None and (
            isinstance(self.sequential_overlap, bool) or int(self.sequential_overlap) <= 0
        ):
            raise ReconstructionValidationError("sequential_overlap must be positive when provided.")
        video_rate = float(self.video_target_fps)
        image_rate = float(self.image_sequence_fps)
        timeout_minutes = float(self.timeout_minutes)
        if not isfinite(video_rate) or video_rate <= 0.0:
            raise ReconstructionValidationError("video_target_fps must be finite and positive.")
        if not isfinite(image_rate) or image_rate <= 0.0:
            raise ReconstructionValidationError("image_sequence_fps must be finite and positive.")
        if not isfinite(timeout_minutes) or timeout_minutes <= 0.0:
            raise ReconstructionValidationError("timeout_minutes must be finite and positive.")
        if not isinstance(self.recursive, bool) or not isinstance(self.use_gpu, bool):
            raise TypeError("recursive and use_gpu must be bool values.")

        object.__setattr__(self, "inputs", normalized)
        object.__setattr__(self, "output_dir", output_dir)
        object.__setattr__(self, "sfm_python", sfm_python)
        object.__setattr__(self, "max_keyframes", None if self.max_keyframes is None else int(self.max_keyframes))
        object.__setattr__(
            self,
            "maximum_image_size",
            None if self.maximum_image_size is None else int(self.maximum_image_size),
        )
        object.__setattr__(
            self,
            "sequential_overlap",
            None if self.sequential_overlap is None else int(self.sequential_overlap),
        )
        object.__setattr__(self, "video_target_fps", video_rate)
        object.__setattr__(self, "image_sequence_fps", image_rate)
        object.__setattr__(self, "timeout_minutes", timeout_minutes)


def build_offline_reconstruction_command(
    configuration: OfflineReconstructionConfiguration,
    *,
    session_id: str,
    worker_script: Path = OFFLINE_RECONSTRUCTION_WORKER,
) -> tuple[str, list[str], Path]:
    """Build the production-depth worker command without importing PyCOLMAP."""

    if not isinstance(configuration, OfflineReconstructionConfiguration):
        raise TypeError("configuration must be OfflineReconstructionConfiguration.")
    worker = Path(worker_script).expanduser().resolve()
    if not worker.is_file():
        raise ReconstructionValidationError(f"Offline reconstruction worker is missing: {worker}")

    arguments = [
        str(worker),
        "--session-id",
        str(session_id),
        "--output-dir",
        str(configuration.output_dir),
        "--sfm-python",
        str(configuration.sfm_python),
        "--profile",
        configuration.profile.value,
        "--matching",
        configuration.matching.value,
        "--video-target-fps",
        str(configuration.video_target_fps),
        "--image-sequence-fps",
        str(configuration.image_sequence_fps),
        "--timeout-minutes",
        str(configuration.timeout_minutes),
    ]
    if configuration.max_keyframes is not None:
        arguments.extend(["--max-keyframes", str(configuration.max_keyframes)])
    if configuration.source_kind is OfflineReconstructionSourceKind.IMAGE_DIRECTORY:
        arguments.extend(["--images-dir", str(configuration.inputs[0])])
    elif configuration.source_kind is OfflineReconstructionSourceKind.RECORDED_VIDEO:
        arguments.extend(["--video", str(configuration.inputs[0])])
    else:
        for path in configuration.inputs:
            arguments.extend(["--image", str(path)])
    if configuration.maximum_image_size is not None:
        arguments.extend(["--maximum-image-size", str(configuration.maximum_image_size)])
    if configuration.sequential_overlap is not None:
        arguments.extend(["--sequential-overlap", str(configuration.sequential_overlap)])
    if configuration.recursive:
        arguments.append("--recursive")
    if not configuration.use_gpu:
        arguments.append("--cpu")
    return sys.executable, arguments, configuration.output_dir


class OfflineReconstructionController(QObject):
    """Own one non-blocking production offline worker process."""

    state_changed = pyqtSignal(object)  # OfflineReconstructionRunState
    event_received = pyqtSignal(object)  # worker protocol mapping
    progressive_sparse_snapshot_received = pyqtSignal(object)  # ProgressiveSfmSnapshot
    log_received = pyqtSignal(str)
    result_ready = pyqtSignal(object, str)  # result payload, output dir
    run_finished = pyqtSignal(object)
    error_occurred = pyqtSignal(str)

    def __init__(
        self,
        parent: QObject | None = None,
        *,
        worker_script: Path = OFFLINE_RECONSTRUCTION_WORKER,
    ) -> None:
        super().__init__(parent)
        self._worker_script = Path(worker_script).expanduser().resolve()
        self._process = QProcess(self)
        self._process.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        self._process.readyReadStandardOutput.connect(self._read_output)
        self._process.finished.connect(self._finished)
        self._process.errorOccurred.connect(self._process_error)
        self._kill_timer = QTimer(self)
        self._kill_timer.setSingleShot(True)
        self._kill_timer.setInterval(8000)
        self._kill_timer.timeout.connect(self._kill_if_running)
        self._progressive_sfm_timer = QTimer(self)
        self._progressive_sfm_timer.setSingleShot(True)
        self._progressive_sfm_timer.setInterval(PROGRESSIVE_SFM_COALESCE_INTERVAL_MS)
        self._progressive_sfm_timer.timeout.connect(self._flush_progressive_sparse_snapshot)
        self._progressive_sfm_buffer = ProgressiveSfmSnapshotBuffer()
        self._progressive_sfm_parse_rejections = 0
        self._state = OfflineReconstructionRunState.IDLE
        self._buffer = ""
        self._output_dir: Path | None = None
        self._session_id: str | None = None
        self._cancel_requested = False
        self._shutdown_started = False
        self._run_finalized = True

    @property
    def state(self) -> OfflineReconstructionRunState:
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
        # Keep the controller reserved until the queued completion handler has
        # validated/admitted the result and finalized the previous run.
        return self.is_running or not self._run_finalized

    @property
    def is_shutdown(self) -> bool:
        return self._shutdown_started

    @property
    def progressive_sfm_transport_stats(self) -> dict[str, int]:
        stats = self._progressive_sfm_buffer.stats
        return {
            "accepted": stats.accepted,
            "coalesced": stats.coalesced,
            "rejected": stats.rejected + self._progressive_sfm_parse_rejections,
            "delivered": stats.delivered,
        }

    @property
    def latest_progressive_sparse_snapshot(self) -> ProgressiveSfmSnapshot | None:
        return self._progressive_sfm_buffer.last_delivered

    def start(
        self,
        configuration: OfflineReconstructionConfiguration,
        *,
        session_id: str | None = None,
    ) -> str:
        if self._shutdown_started:
            raise ReconstructionValidationError(
                "Offline reconstruction controller is shut down and cannot start a new job."
            )
        if self.is_running:
            raise ReconstructionValidationError("An offline reconstruction job is already running.")
        if not self._run_finalized:
            raise ReconstructionValidationError(
                "The previous offline reconstruction completion is still being finalized."
            )
        sid = self._canonical_session_id(session_id or new_reconstruction_session_id())
        self._progressive_sfm_timer.stop()
        self._progressive_sfm_buffer.reset(sid)
        self._progressive_sfm_parse_rejections = 0
        program, arguments, output_dir = build_offline_reconstruction_command(
            configuration,
            session_id=sid,
            worker_script=self._worker_script,
        )
        self._validate_output_dir_for_start(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        self._buffer = ""
        self._output_dir = output_dir
        self._session_id = sid
        self._cancel_requested = False
        self._run_finalized = False
        self._set_state(OfflineReconstructionRunState.STARTING)

        environment = QProcessEnvironment.systemEnvironment()
        environment.insert("PYTHONUNBUFFERED", "1")
        self._process.setProcessEnvironment(environment)
        self._process.setWorkingDirectory(str(PROJECT_ROOT))
        self._process.setProgram(program)
        self._process.setArguments(arguments)
        self._process.start()
        if not self._process.waitForStarted(3000):
            self._run_finalized = True
            self._progressive_sfm_timer.stop()
            self._progressive_sfm_buffer.clear()
            self._set_state(OfflineReconstructionRunState.FAILED)
            raise ReconstructionValidationError(
                f"Failed to start offline reconstruction worker: {self._process.errorString()}"
            )
        self._set_state(OfflineReconstructionRunState.RUNNING)
        return sid

    def cancel(self) -> bool:
        if self._shutdown_started or not self.is_running or self._cancel_requested:
            return False
        self._cancel_requested = True
        self._set_state(OfflineReconstructionRunState.CANCELLING)
        self._process.terminate()
        self._kill_timer.start()
        return True

    def shutdown(self) -> None:
        if self._shutdown_started:
            return
        self._shutdown_started = True
        if self.is_running:
            self._cancel_requested = True
            self._set_state(OfflineReconstructionRunState.CANCELLING)
            self._process.terminate()
            if not self._process.waitForFinished(8000):
                self._kill_if_running()
                self._process.waitForFinished(3000)
        self._kill_timer.stop()
        self._progressive_sfm_timer.stop()
        self._progressive_sfm_buffer.clear()

    def _set_state(self, state: OfflineReconstructionRunState) -> None:
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
            snapshot = parse_progressive_sfm_worker_line(line)
        except SfmWorkerProtocolError as exc:
            self._progressive_sfm_parse_rejections += 1
            self.log_received.emit(
                f"Rejected malformed progressive sparse-SfM transport line: {exc}"
            )
            return
        if snapshot is not None:
            self._offer_progressive_sparse_snapshot(snapshot)
            return
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            self.log_received.emit(line)
            return
        if isinstance(payload, dict) and payload.get("protocol") == OFFLINE_EVENT_PROTOCOL:
            self.event_received.emit(payload)
            return
        self.log_received.emit(line)

    def _offer_progressive_sparse_snapshot(self, snapshot: ProgressiveSfmSnapshot) -> None:
        disposition = self._progressive_sfm_buffer.offer(snapshot)
        if disposition is ProgressiveSfmOfferDisposition.REJECTED:
            self.log_received.emit(
                "Rejected progressive sparse-SfM snapshot: "
                + self._progressive_sfm_buffer.last_rejection_reason
            )
            return
        if snapshot.stage is ProgressiveSfmStage.FINALIZING:
            self._progressive_sfm_timer.stop()
            self._flush_progressive_sparse_snapshot()
            return
        if not self._progressive_sfm_timer.isActive():
            self._progressive_sfm_timer.start()

    def _flush_progressive_sparse_snapshot(self) -> None:
        snapshot = self._progressive_sfm_buffer.take_pending()
        if snapshot is not None:
            self.progressive_sparse_snapshot_received.emit(snapshot)

    def _load_result(self) -> dict[str, object] | None:
        if self._output_dir is None:
            return None
        path = self._output_dir / "offline_result.json"
        if not path.is_file():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            self.log_received.emit(f"Could not read offline reconstruction result: {exc}")
            return None
        return payload if isinstance(payload, dict) else None

    def _finished(self, exit_code: int, _status: QProcess.ExitStatus) -> None:
        self._kill_timer.stop()
        self._progressive_sfm_timer.stop()
        if self._buffer.strip():
            self._consume_line(self._buffer.strip())
        self._buffer = ""
        self._flush_progressive_sparse_snapshot()
        result = self._load_result()
        result_rejection_reason = self._result_rejection_reason(result)

        if self._cancel_requested:
            state = OfflineReconstructionRunState.CANCELLED
        elif (
            exit_code == 0
            and result is not None
            and not result_rejection_reason
            and bool(result.get("pass"))
        ):
            state = OfflineReconstructionRunState.COMPLETED
        else:
            state = OfflineReconstructionRunState.FAILED

        if result_rejection_reason and not self._cancel_requested:
            self.error_occurred.emit(
                f"Offline reconstruction result rejected: {result_rejection_reason}"
            )
        elif state is OfflineReconstructionRunState.FAILED and not self._cancel_requested:
            if result is None:
                detail = (
                    "Offline reconstruction worker did not produce a readable offline_result.json "
                    f"(exit code {int(exit_code)})."
                )
            else:
                detail = str(result.get("error", "")).strip() or (
                    f"Offline reconstruction worker failed with exit code {int(exit_code)}."
                )
            self.error_occurred.emit(detail)
        if (
            state is OfflineReconstructionRunState.COMPLETED
            and result is not None
            and self._output_dir is not None
        ):
            self.result_ready.emit(result, str(self._output_dir))

        self._set_state(state)
        self._run_finalized = True
        self.run_finished.emit(
            {
                "state": state,
                "exit_code": int(exit_code),
                "session_id": self._session_id or "",
                "output_dir": str(self._output_dir or ""),
                "result": result,
                "result_accepted": state is OfflineReconstructionRunState.COMPLETED,
                "result_rejection_reason": result_rejection_reason,
            }
        )

    def _process_error(self, _error: QProcess.ProcessError) -> None:
        if not self._cancel_requested and not self._shutdown_started:
            message = self._process.errorString()
            self.log_received.emit(f"Offline reconstruction process error: {message}")
            self.error_occurred.emit(message)

    @staticmethod
    def _canonical_session_id(value: object) -> str:
        text = str(value).strip()
        try:
            return str(UUID(text))
        except (ValueError, AttributeError, TypeError) as exc:
            raise ReconstructionValidationError(
                "Offline reconstruction session_id must be a valid UUID."
            ) from exc

    @staticmethod
    def _validate_output_dir_for_start(output_dir: Path) -> None:
        if output_dir.exists():
            if not output_dir.is_dir():
                raise ReconstructionValidationError(
                    f"Offline output path is not a directory: {output_dir}"
                )
            if any(output_dir.iterdir()):
                raise ReconstructionValidationError(
                    "Offline output directory became non-empty before start; refusing stale-result reuse."
                )

    def _result_rejection_reason(self, result: dict[str, object] | None) -> str:
        if result is None:
            return ""
        if self._session_id is None:
            return "controller has no active session identity for the finished worker"
        try:
            actual = self._canonical_session_id(result.get("session_id", ""))
        except ReconstructionValidationError as exc:
            return str(exc)
        if actual != self._session_id:
            return (
                "result session_id does not match the active offline reconstruction session "
                f"({actual} != {self._session_id})"
            )
        return ""

    def _kill_if_running(self) -> None:
        if not self.is_running:
            return
        pid = int(self._process.processId())
        killed_group = False
        if pid > 0 and hasattr(os, "killpg"):
            try:
                os.killpg(pid, signal.SIGKILL)
                killed_group = True
            except (ProcessLookupError, PermissionError, OSError):
                killed_group = False
        if not killed_group:
            self._process.kill()
