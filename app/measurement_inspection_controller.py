"""QProcess controller for the Phase 7.5 Measure & Inspect mode."""
from __future__ import annotations

import json
import logging
import shutil
import sys
import tempfile
import time
from enum import Enum
from pathlib import Path

import numpy as np
from PyQt6.QtCore import QObject, QProcess, pyqtSignal

LOGGER = logging.getLogger(__name__)


class InspectionControllerState(Enum):
    IDLE = "idle"
    READY = "ready"
    RUNNING = "running"
    RECOMPUTING = "recomputing"
    CANCELLING = "cancelling"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class MeasurementInspectionController(QObject):
    state_changed = pyqtSignal(object)
    progress_changed = pyqtSignal(int, str, str)
    log_received = pyqtSignal(str)
    result_ready = pyqtSignal(str)
    failure_received = pyqtSignal(str)

    def __init__(self, project_root: Path, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._root = Path(project_root)
        self._process: QProcess | None = None
        self._temp_dir: Path | None = None
        self._temp_dirs: list[Path] = []
        self._result_dir: Path | None = None
        self._state = InspectionControllerState.IDLE
        self._stdout_buffer = ""
        self._last_error = ""
        self._job_log_path: Path | None = None

    @property
    def state(self) -> InspectionControllerState:
        return self._state

    @property
    def is_running(self) -> bool:
        return self._state in {
            InspectionControllerState.RUNNING,
            InspectionControllerState.RECOMPUTING,
            InspectionControllerState.CANCELLING,
        }

    @property
    def result_dir(self) -> Path | None:
        return self._result_dir

    def start_analysis(
        self,
        *,
        image_bgr: np.ndarray,
        depth_model_id: str,
        depth_config: dict[str, object],
        perception_model_id: str,
        depth_profile: str,
        perception_profile: str,
        max_objects: int,
        box_method: str = "auto",
    ) -> None:
        if self.is_running:
            raise RuntimeError("An inspection analysis is already running.")
        image = np.asarray(image_bgr, dtype=np.uint8)
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("Inspection input must be an HxWx3 BGR image.")
        self._prepare_temp()
        assert self._temp_dir is not None
        input_path = self._temp_dir / "frame.npy"
        np.save(input_path, np.ascontiguousarray(image))
        output = self._temp_dir / "result"
        config_path = self._temp_dir / "depth_config.json"
        config_path.write_text(json.dumps(depth_config), encoding="utf-8")
        args = [
            "--input", str(input_path),
            "--output", str(output),
            "--depth-model", depth_model_id,
            "--depth-config", str(config_path),
            "--perception-model", perception_model_id,
            "--depth-profile", depth_profile,
            "--perception-profile", perception_profile,
            "--max-objects", str(int(max_objects)),
            "--box-method", box_method,
        ]
        self._result_dir = output
        self._job_log_path = self._temp_dir / "job.log"
        self._start_process(args, InspectionControllerState.RUNNING)

    def recompute(
        self,
        *,
        base_state_path: Path,
        observation_id: str,
        manual_mask: np.ndarray | None,
        manual_revision: int,
        box_method: str,
        reset_manual: bool = False,
    ) -> None:
        if self.is_running:
            raise RuntimeError("An inspection analysis is already running.")
        self._prepare_temp()
        assert self._temp_dir is not None
        output = self._temp_dir / "recomputed"
        args = [
            "--output", str(output),
            "--base-state", str(Path(base_state_path)),
            "--box-method", box_method,
        ]
        if reset_manual:
            args.append("--reset-manual")
        if manual_mask is not None:
            mask_path = self._temp_dir / "manual_mask.npy"
            np.save(mask_path, np.asarray(manual_mask, dtype=bool))
            args += [
                "--manual-mask", str(mask_path),
                "--observation-id", str(observation_id),
                "--manual-revision", str(int(manual_revision)),
            ]
        self._result_dir = output
        self._job_log_path = self._temp_dir / "job.log"
        self._start_process(args, InspectionControllerState.RECOMPUTING)

    def cancel(self) -> None:
        if not self.is_running or self._process is None:
            return
        self._set_state(InspectionControllerState.CANCELLING)
        self._process.terminate()
        if not self._process.waitForFinished(1800):
            self._process.kill()

    def shutdown(self) -> None:
        self.cancel()
        self._cleanup_temp()

    def _prepare_temp(self) -> None:
        # Keep previous completed bundles alive while a recompute process reads
        # its trusted state file. All session directories are removed on shutdown.
        self._temp_dir = Path(tempfile.mkdtemp(prefix="eia_measure_inspect_"))
        self._temp_dirs.append(self._temp_dir)

    def _start_process(self, arguments: list[str], state: InspectionControllerState) -> None:
        process = QProcess(self)
        process.setWorkingDirectory(str(self._root))
        process.setProgram(sys.executable)
        process.setArguments([
            str(self._root / "scripts" / "run_measurement_inspection_job.py"),
            *arguments,
        ])
        process.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        process.readyReadStandardOutput.connect(self._read_output)
        process.finished.connect(self._finished)
        process.errorOccurred.connect(lambda _error: self._fail(process.errorString()))
        self._stdout_buffer = ""
        self._last_error = ""
        if self._job_log_path is not None:
            self._job_log_path.write_text("", encoding="utf-8")
        self._process = process
        self._set_state(state)
        self.progress_changed.emit(0, "starting", "Starting isolated inspection analysis")
        process.start()

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
        LOGGER.info("Measure & Inspect job: %s", line)
        if self._job_log_path is not None:
            with self._job_log_path.open("a", encoding="utf-8") as stream:
                stream.write(line + "\n")
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
            self.failure_received.emit(self._last_error)

    def _finished(self, exit_code: int, _status: QProcess.ExitStatus) -> None:
        if self._stdout_buffer.strip():
            self._handle_line(self._stdout_buffer.strip())
        if self._state is InspectionControllerState.CANCELLING:
            self._set_state(InspectionControllerState.CANCELLED)
        elif exit_code == 0 and self._result_dir is not None and self._result_dir.exists():
            self._set_state(InspectionControllerState.COMPLETED)
            self.progress_changed.emit(100, "complete", "Inspection bundle ready")
            self.result_ready.emit(str(self._result_dir))
        else:
            self._set_state(InspectionControllerState.FAILED)
            message = self._last_error or "Inspection analysis failed."
            if self._job_log_path is not None:
                message = f"{message} Job log: {self._job_log_path}"
            self.failure_received.emit(message)
        self._process = None

    def _fail(self, message: str) -> None:
        self._last_error = str(message).strip()
        self.log_received.emit(self._last_error)
        LOGGER.error("Measure & Inspect process failure: %s", self._last_error)
        if self._state is not InspectionControllerState.CANCELLING:
            self._set_state(InspectionControllerState.FAILED)
            self.failure_received.emit(message)

    def _set_state(self, state: InspectionControllerState) -> None:
        self._state = state
        self.state_changed.emit(state)

    def _cleanup_temp(self) -> None:
        for directory in self._temp_dirs:
            shutil.rmtree(directory, ignore_errors=True)
        self._temp_dirs.clear()
        self._temp_dir = None
        self._result_dir = None
