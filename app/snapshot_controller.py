from __future__ import annotations

import json
import shutil
import sys
import tempfile
from enum import Enum
from pathlib import Path

import numpy as np
from PyQt6.QtCore import QObject, QProcess, pyqtSignal


class SnapshotRunState(Enum):
    IDLE = "idle"
    RUNNING = "running"
    CANCELLING = "cancelling"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class SnapshotController(QObject):
    state_changed = pyqtSignal(object)
    log_received = pyqtSignal(str)
    result_ready = pyqtSignal(str)

    def __init__(self, project_root: Path, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._root = Path(project_root)
        self._process: QProcess | None = None
        self._temp_dir: Path | None = None
        self._output_dir: Path | None = None
        self._state = SnapshotRunState.IDLE

    @property
    def is_running(self) -> bool:
        return self._state in {SnapshotRunState.RUNNING, SnapshotRunState.CANCELLING}

    def start(self, *, image: np.ndarray, model_id: str, config: dict[str, object], output_dir: Path) -> None:
        if self.is_running:
            raise RuntimeError("A snapshot job is already running.")
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("Snapshot input must be an HxWx3 BGR image.")
        self._temp_dir = Path(tempfile.mkdtemp(prefix="eia_snapshot_"))
        self._output_dir = Path(output_dir)
        input_path = self._temp_dir / "frame.npy"
        config_path = self._temp_dir / "config.json"
        np.save(input_path, np.ascontiguousarray(image, dtype=np.uint8))
        config_path.write_text(json.dumps(config), encoding="utf-8")

        process = QProcess(self)
        process.setWorkingDirectory(str(self._root))
        process.setProgram(sys.executable)
        process.setArguments([
            str(self._root / "scripts" / "run_snapshot_job.py"),
            "--model", model_id,
            "--config", str(config_path),
            "--input", str(input_path),
            "--output", str(self._output_dir),
        ])
        process.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        process.readyReadStandardOutput.connect(self._read_output)
        process.finished.connect(self._finished)
        process.errorOccurred.connect(lambda _err: self._fail(process.errorString()))
        self._process = process
        self._set_state(SnapshotRunState.RUNNING)
        process.start()

    def cancel(self) -> None:
        if not self.is_running or self._process is None:
            return
        self._set_state(SnapshotRunState.CANCELLING)
        self._process.terminate()
        if not self._process.waitForFinished(1500):
            self._process.kill()

    def shutdown(self) -> None:
        self.cancel()
        self._cleanup()

    def _read_output(self) -> None:
        if self._process is None:
            return
        text = bytes(self._process.readAllStandardOutput()).decode("utf-8", errors="replace")
        for line in text.splitlines():
            self.log_received.emit(line)

    def _finished(self, exit_code: int, _status: QProcess.ExitStatus) -> None:
        if self._state is SnapshotRunState.CANCELLING:
            self._set_state(SnapshotRunState.CANCELLED)
        elif exit_code == 0 and self._output_dir is not None:
            self._set_state(SnapshotRunState.COMPLETED)
            self.result_ready.emit(str(self._output_dir))
        else:
            self._set_state(SnapshotRunState.FAILED)
        self._process = None
        self._cleanup()

    def _fail(self, message: str) -> None:
        self.log_received.emit(message)
        if self._state is not SnapshotRunState.CANCELLING:
            self._set_state(SnapshotRunState.FAILED)

    def _set_state(self, state: SnapshotRunState) -> None:
        self._state = state
        self.state_changed.emit(state)

    def _cleanup(self) -> None:
        if self._temp_dir is not None:
            shutil.rmtree(self._temp_dir, ignore_errors=True)
            self._temp_dir = None
