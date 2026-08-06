"""QProcess controller for Step 9 inspection exports."""
from __future__ import annotations

import json
import sys
from enum import Enum
from pathlib import Path

from PyQt6.QtCore import QObject, QProcess, pyqtSignal


class InspectionExportState(Enum):
    IDLE = "idle"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class InspectionExportController(QObject):
    state_changed = pyqtSignal(object)
    progress_changed = pyqtSignal(int, str, str)
    result_ready = pyqtSignal(str, int, int)
    failure_received = pyqtSignal(str)

    def __init__(self, project_root: Path, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._root = Path(project_root)
        self._process: QProcess | None = None
        self._state = InspectionExportState.IDLE
        self._buffer = ""
        self._last_error = ""
        self._object_count = 0
        self._file_count = 0
        self._result_dir: Path | None = None

    @property
    def is_running(self) -> bool:
        return self._state is InspectionExportState.RUNNING

    def start(self, *, bundle_dir: Path, output_dir: Path, observation_ids: tuple[str, ...], locked_ids: tuple[str, ...]) -> None:
        if self.is_running:
            raise RuntimeError("An inspection export is already running.")
        args = [
            str(self._root / "scripts" / "run_measurement_export_job.py"),
            "--bundle", str(Path(bundle_dir)), "--output", str(Path(output_dir)),
        ]
        for value in observation_ids:
            args += ["--observation-id", value]
        for value in locked_ids:
            args += ["--locked-id", value]
        process = QProcess(self)
        process.setWorkingDirectory(str(self._root))
        process.setProgram(sys.executable)
        process.setArguments(args)
        process.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        process.readyReadStandardOutput.connect(self._read_output)
        process.finished.connect(self._finished)
        process.errorOccurred.connect(lambda _error: self._fail(process.errorString()))
        self._process = process
        self._buffer = ""
        self._last_error = ""
        self._object_count = 0
        self._file_count = 0
        self._result_dir = None
        self._set_state(InspectionExportState.RUNNING)
        process.start()

    def cancel(self) -> None:
        if self._process is not None and self.is_running:
            self._process.terminate()

    def shutdown(self) -> None:
        self.cancel()
        if self._process is not None:
            self._process.waitForFinished(1500)

    def _read_output(self) -> None:
        if self._process is None:
            return
        self._buffer += bytes(self._process.readAllStandardOutput()).decode("utf-8", errors="replace")
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            self._handle_line(line.strip())

    def _handle_line(self, line: str) -> None:
        if line.startswith("EVENT "):
            try:
                payload = json.loads(line[6:])
                self._object_count = int(payload.get("object_count", self._object_count))
                self._file_count = int(payload.get("file_count", self._file_count))
                self.progress_changed.emit(int(payload.get("progress", 0)), str(payload.get("stage", "exporting")), str(payload.get("message", "")))
            except (ValueError, TypeError, json.JSONDecodeError):
                pass
        elif line.startswith("RESULT "):
            self._result_dir = Path(line[7:].strip())
        elif line.startswith("ERROR "):
            self._last_error = line[6:].strip()

    def _finished(self, exit_code: int, _status: QProcess.ExitStatus) -> None:
        if self._buffer.strip():
            self._handle_line(self._buffer.strip())
        if exit_code == 0 and self._result_dir is not None and self._result_dir.is_dir():
            self._set_state(InspectionExportState.COMPLETED)
            self.result_ready.emit(str(self._result_dir), self._object_count, self._file_count)
        else:
            self._set_state(InspectionExportState.FAILED)
            self.failure_received.emit(self._last_error or "Inspection export failed.")
        self._process = None

    def _fail(self, message: str) -> None:
        self._last_error = str(message)
        self._set_state(InspectionExportState.FAILED)
        self.failure_received.emit(self._last_error)

    def _set_state(self, state: InspectionExportState) -> None:
        self._state = state
        self.state_changed.emit(state)
