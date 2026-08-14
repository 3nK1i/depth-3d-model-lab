"""QProcess orchestration for Phase 9.2D Auto Segment UI jobs."""
from __future__ import annotations

import json
import shutil
import sys
import tempfile
from enum import Enum
from pathlib import Path

import numpy as np
from PyQt6.QtCore import QObject, QProcess, pyqtSignal


class AutoSegmentControllerState(Enum):
    IDLE = "idle"
    RUNNING = "running"
    CANCELLING = "cancelling"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class AutoSegmentController(QObject):
    state_changed = pyqtSignal(object)
    progress_changed = pyqtSignal(int, str, str)
    log_received = pyqtSignal(str)
    result_ready = pyqtSignal(str)
    failure_received = pyqtSignal(str)

    def __init__(self, project_root: Path, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._root = Path(project_root)
        self._process: QProcess | None = None
        self._temp_dirs: list[Path] = []
        self._temp_dir: Path | None = None
        self._result_dir: Path | None = None
        self._state = AutoSegmentControllerState.IDLE
        self._stdout_buffer = ""
        self._last_error = ""

    @property
    def state(self) -> AutoSegmentControllerState:
        return self._state

    @property
    def is_running(self) -> bool:
        return self._state in {AutoSegmentControllerState.RUNNING, AutoSegmentControllerState.CANCELLING}

    def start(
        self,
        *,
        image_bgr: np.ndarray,
        model_id: str = "ultralytics-sam2-prompted",
        profile: str = "efficient",
        minimum_mask_pixels: int = 64,
        max_proposals: int = 64,
    ) -> None:
        if self.is_running:
            raise RuntimeError("An Auto Segment job is already running.")
        image = np.asarray(image_bgr, dtype=np.uint8)
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("Auto Segment input must be an HxWx3 BGR image.")
        if int(minimum_mask_pixels) < 1 or int(max_proposals) < 1:
            raise ValueError("Auto Segment limits must be positive integers.")

        self._temp_dir = Path(tempfile.mkdtemp(prefix="eia_auto_segment_"))
        self._temp_dirs.append(self._temp_dir)
        input_path = self._temp_dir / "frame.npy"
        np.save(input_path, np.ascontiguousarray(image))
        output = self._temp_dir / "result"
        self._result_dir = output

        process = QProcess(self)
        process.setWorkingDirectory(str(self._root))
        process.setProgram(sys.executable)
        process.setArguments(
            [
                str(self._root / "scripts" / "run_auto_segment_job.py"),
                "--input", str(input_path),
                "--output", str(output),
                "--model", str(model_id),
                "--profile", str(profile),
                "--minimum-mask-pixels", str(int(minimum_mask_pixels)),
                "--max-proposals", str(int(max_proposals)),
            ]
        )
        process.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        process.readyReadStandardOutput.connect(self._read_output)
        process.finished.connect(self._finished)
        process.errorOccurred.connect(lambda _error: self._fail(process.errorString()))
        self._process = process
        self._stdout_buffer = ""
        self._last_error = ""
        self._set_state(AutoSegmentControllerState.RUNNING)
        self.progress_changed.emit(0, "starting", "Starting isolated Auto Segment job")
        process.start()

    def cancel(self) -> None:
        if not self.is_running or self._process is None:
            return
        self._set_state(AutoSegmentControllerState.CANCELLING)
        self._process.terminate()
        if not self._process.waitForFinished(1800):
            self._process.kill()

    def shutdown(self) -> None:
        self.cancel()
        for path in self._temp_dirs:
            shutil.rmtree(path, ignore_errors=True)
        self._temp_dirs.clear()
        self._temp_dir = None

    def _read_output(self) -> None:
        if self._process is None:
            return
        text = bytes(self._process.readAllStandardOutput()).decode("utf-8", errors="replace")
        self._stdout_buffer += text
        while "\n" in self._stdout_buffer:
            line, self._stdout_buffer = self._stdout_buffer.split("\n", 1)
            self._handle_line(line.strip())

    def _handle_line(self, line: str) -> None:
        if not line:
            return
        self.log_received.emit(line)
        if line.startswith("EVENT "):
            try:
                payload = json.loads(line[6:])
                self.progress_changed.emit(
                    int(payload.get("progress", 0)),
                    str(payload.get("stage", "running")),
                    str(payload.get("message", "")),
                )
            except (TypeError, ValueError, json.JSONDecodeError):
                pass
        elif line.startswith("RESULT "):
            self._result_dir = Path(line[7:].strip())
        elif line.startswith("ERROR "):
            self._last_error = line[6:].strip()

    def _finished(self, exit_code: int, _status: QProcess.ExitStatus) -> None:
        if self._stdout_buffer.strip():
            self._handle_line(self._stdout_buffer.strip())
        if self._state is AutoSegmentControllerState.CANCELLING:
            self._set_state(AutoSegmentControllerState.CANCELLED)
        elif exit_code == 0 and self._result_dir is not None and self._result_dir.exists():
            self._set_state(AutoSegmentControllerState.COMPLETED)
            self.result_ready.emit(str(self._result_dir))
        else:
            self._set_state(AutoSegmentControllerState.FAILED)
            self.failure_received.emit(self._last_error or "Auto Segment failed.")
        self._process = None

    def _fail(self, message: str) -> None:
        self._last_error = str(message).strip()
        if self._state is not AutoSegmentControllerState.CANCELLING:
            self._set_state(AutoSegmentControllerState.FAILED)
            self.failure_received.emit(self._last_error or "Auto Segment process failure.")

    def _set_state(self, state: AutoSegmentControllerState) -> None:
        if state is self._state:
            return
        self._state = state
        self.state_changed.emit(state)
