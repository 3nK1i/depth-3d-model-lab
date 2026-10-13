"""Qt-side supervisor for one isolated Digital Twin processing job.

Phase 9.6E1 intentionally owns only generic process lifecycle.  DA3 full-scene
processing and NeRF training commands are added by later adapters; this module
never imports their heavy runtimes into the production ``depth`` environment.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import signal
from time import monotonic

from PyQt6.QtCore import QObject, QProcess, QProcessEnvironment, QTimer, pyqtSignal

from app.reconstruction_processing_jobs import (
    DEFAULT_DIGITAL_TWIN_PROCESS_POLICY,
    DigitalTwinProcessingCommand,
    DigitalTwinProcessingJobSpec,
    DigitalTwinProcessingLifecycle,
    DigitalTwinProcessingSnapshot,
    apply_digital_twin_processing_event,
    begin_digital_twin_processing_snapshot,
    idle_digital_twin_processing_snapshot,
    update_digital_twin_processing_snapshot,
)


MAX_PROTOCOL_BUFFER_CHARS = 1_000_000


class DigitalTwinProcessingJobController(QObject):
    """Supervise at most one shell-free heavyweight Digital Twin build job."""

    snapshot_changed = pyqtSignal(object)
    event_received = pyqtSignal(object)
    log_received = pyqtSignal(str)
    result_ready = pyqtSignal(object)
    run_finished = pyqtSignal(object)
    error_occurred = pyqtSignal(str)

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._process = QProcess(self)
        self._process.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        self._process.readyReadStandardOutput.connect(self._read_output)
        self._process.finished.connect(self._finished)
        self._process.errorOccurred.connect(self._process_error)

        self._kill_timer = QTimer(self)
        self._kill_timer.setSingleShot(True)
        self._kill_timer.setInterval(8000)
        self._kill_timer.timeout.connect(self._kill_if_running)

        self._elapsed_timer = QTimer(self)
        self._elapsed_timer.setInterval(500)
        self._elapsed_timer.timeout.connect(self._refresh_elapsed)

        self._snapshot = idle_digital_twin_processing_snapshot()
        self._command: DigitalTwinProcessingCommand | None = None
        self._buffer = ""
        self._started_at: float | None = None
        self._cancel_requested = False
        self._shutdown_started = False
        self._protocol_failure = ""
        self._using_setsid_wrapper = False

    @property
    def snapshot(self) -> DigitalTwinProcessingSnapshot:
        return self._snapshot

    @property
    def is_running(self) -> bool:
        return self._process.state() != QProcess.ProcessState.NotRunning

    @property
    def is_active(self) -> bool:
        return self.is_running or self._snapshot.lifecycle.is_active

    @property
    def process_id(self) -> int:
        return max(0, int(self._process.processId()))

    @property
    def is_shutdown(self) -> bool:
        return self._shutdown_started

    def start(
        self,
        spec: DigitalTwinProcessingJobSpec,
        command: DigitalTwinProcessingCommand,
    ) -> str:
        if self._shutdown_started:
            raise RuntimeError("Digital Twin processing supervisor is shut down.")
        if self.is_active:
            raise RuntimeError("A Digital Twin processing job is already active.")
        if not isinstance(spec, DigitalTwinProcessingJobSpec):
            raise TypeError("spec must be DigitalTwinProcessingJobSpec.")
        if not isinstance(command, DigitalTwinProcessingCommand):
            raise TypeError("command must be DigitalTwinProcessingCommand.")

        # Materialization is owned by the future E2/E3 adapter, but the generic
        # supervisor guarantees the declared destination exists for progress and
        # provenance reporting. It never deletes/cleans an existing directory.
        spec.output_dir.mkdir(parents=True, exist_ok=True)

        self._command = command
        self._buffer = ""
        self._cancel_requested = False
        self._protocol_failure = ""
        self._started_at = monotonic()
        self._set_snapshot(begin_digital_twin_processing_snapshot(spec))

        environment = QProcessEnvironment.systemEnvironment()
        environment.insert("PYTHONUNBUFFERED", "1")
        environment.insert("DEPTH3D_DIGITAL_TWIN_JOB_ID", spec.job_id)
        environment.insert("DEPTH3D_DIGITAL_TWIN_EXPECTED_PROTOCOL", spec.expected_protocol)
        environment.insert("DEPTH3D_DIGITAL_TWIN_OUTPUT_DIR", str(spec.output_dir))
        for key, value in command.environment_overrides:
            environment.insert(key, value)
        self._process.setProcessEnvironment(environment)
        self._process.setWorkingDirectory(str(command.working_directory))

        self._using_setsid_wrapper = False
        if os.name == "posix" and hasattr(self._process, "setChildProcessModifier"):
            # Give the complete future DA3/NeRF job tree one process group so
            # cancellation can never strand an external CUDA child.
            self._process.setChildProcessModifier(os.setsid)
            self._process.setProgram(str(command.program))
            self._process.setArguments(list(command.arguments))
        elif os.name == "posix":
            setsid = shutil.which("setsid")
            if not setsid:
                self._fail_start("POSIX setsid is required for Digital Twin processing isolation.")
                raise RuntimeError(self._snapshot.error)
            self._using_setsid_wrapper = True
            self._process.setProgram(str(setsid))
            self._process.setArguments([str(command.program), *command.arguments])
        else:
            self._fail_start(
                "Digital Twin processing process-tree isolation is qualified only on POSIX/Linux."
            )
            raise RuntimeError(self._snapshot.error)

        self._process.start()
        if not self._process.waitForStarted(3000):
            message = self._process.errorString() or "Digital Twin processing job did not start."
            self._fail_start(message)
            raise RuntimeError(message)

        self._set_snapshot(
            update_digital_twin_processing_snapshot(
                self._snapshot,
                lifecycle=DigitalTwinProcessingLifecycle.RUNNING,
                stage="starting_external_runtime",
                message="External Digital Twin processing job started.",
                elapsed_seconds=self._elapsed_seconds(),
            )
        )
        self._elapsed_timer.start()
        return spec.job_id

    def cancel(self) -> bool:
        if self._shutdown_started or not self.is_running or not self._snapshot.can_cancel:
            return False
        self._cancel_requested = True
        self._set_snapshot(
            update_digital_twin_processing_snapshot(
                self._snapshot,
                lifecycle=DigitalTwinProcessingLifecycle.CANCELLING,
                stage="cancelling",
                message="Stopping isolated external processing job.",
                elapsed_seconds=self._elapsed_seconds(),
            )
        )
        self._terminate_process_group()
        self._kill_timer.start()
        return True

    def shutdown(self, *, timeout_ms: int = 11000) -> None:
        if self._shutdown_started:
            return
        self._shutdown_started = True
        self._elapsed_timer.stop()
        if not self.is_running:
            self._kill_timer.stop()
            return
        self._cancel_requested = True
        if self._snapshot.spec is not None and not self._snapshot.lifecycle.is_terminal:
            self._set_snapshot(
                update_digital_twin_processing_snapshot(
                    self._snapshot,
                    lifecycle=DigitalTwinProcessingLifecycle.CANCELLING,
                    stage="shutdown",
                    message="Application shutdown is releasing the external processing job.",
                    elapsed_seconds=self._elapsed_seconds(),
                )
            )
        self._terminate_process_group()
        wait_ms = max(500, min(int(timeout_ms), 8000))
        if not self._process.waitForFinished(wait_ms):
            self._kill_if_running()
            self._process.waitForFinished(max(500, int(timeout_ms) - wait_ms))
        self._kill_timer.stop()

    def _set_snapshot(self, snapshot: DigitalTwinProcessingSnapshot) -> None:
        if snapshot == self._snapshot:
            return
        self._snapshot = snapshot
        self.snapshot_changed.emit(snapshot)

    def _elapsed_seconds(self) -> float:
        if self._started_at is None:
            return float(self._snapshot.elapsed_seconds)
        return max(0.0, monotonic() - self._started_at)

    def _refresh_elapsed(self) -> None:
        if not self._snapshot.lifecycle.is_active or self._snapshot.spec is None:
            return
        self._set_snapshot(
            update_digital_twin_processing_snapshot(
                self._snapshot,
                elapsed_seconds=self._elapsed_seconds(),
            )
        )

    def _read_output(self) -> None:
        raw = bytes(self._process.readAllStandardOutput()).decode("utf-8", errors="replace")
        if not raw:
            return
        self._buffer += raw
        if len(self._buffer) > MAX_PROTOCOL_BUFFER_CHARS:
            self._protocol_failure = "Digital Twin processing output exceeded the bounded protocol buffer."
            self.error_occurred.emit(self._protocol_failure)
            self._buffer = ""
            self._terminate_process_group()
            self._kill_timer.start()
            return
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            self._consume_line(line.rstrip("\r"))

    def _consume_line(self, line: str) -> None:
        text = str(line).strip()
        if not text:
            return
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            self.log_received.emit(text)
            return
        if not isinstance(payload, dict) or self._snapshot.spec is None:
            self.log_received.emit(text)
            return
        if str(payload.get("protocol", "")).strip() != self._snapshot.spec.expected_protocol:
            self.log_received.emit(text)
            return
        rejection = self._event_rejection_reason(payload)
        if rejection:
            self._protocol_failure = rejection
            self.error_occurred.emit(rejection)
            self._terminate_process_group()
            self._kill_timer.start()
            return
        try:
            snapshot = apply_digital_twin_processing_event(
                self._snapshot,
                payload,
                elapsed_seconds=self._elapsed_seconds(),
            )
        except (TypeError, ValueError) as exc:
            self._protocol_failure = f"Invalid Digital Twin processing event: {exc}"
            self.error_occurred.emit(self._protocol_failure)
            self._terminate_process_group()
            self._kill_timer.start()
            return
        self._set_snapshot(snapshot)
        self.event_received.emit(payload)

    def _event_rejection_reason(self, payload: dict[str, object]) -> str:
        spec = self._snapshot.spec
        if spec is None:
            return "Digital Twin processing event has no active job identity."
        if str(payload.get("job_id", "")).strip() != spec.job_id:
            return "Digital Twin processing event rejected stale/mismatched job_id."
        event_scene_id = str(payload.get("scene_id", "")).strip()
        if event_scene_id and spec.scene_id and event_scene_id != spec.scene_id:
            return "Digital Twin processing event rejected mismatched scene_id."
        result_path = str(payload.get("result_path", "")).strip()
        if result_path:
            try:
                resolved_result = Path(result_path).expanduser().resolve()
                resolved_result.relative_to(spec.output_dir)
            except (OSError, ValueError):
                return "Digital Twin processing event rejected result_path outside declared output_dir."
        return self._unsafe_authority_claim(payload)

    @staticmethod
    def _unsafe_authority_claim(payload: dict[str, object]) -> str:
        forbidden = (
            ("canonical_geometry_authoritative", "canonical geometry authority"),
            ("metric_scale_authoritative", "metric scale authority"),
            ("measurement_authoritative", "measurement authority"),
            ("qc_authoritative", "QC authority"),
            ("certified_metrology", "certified metrology"),
        )
        for key, label in forbidden:
            if bool(payload.get(key, False)):
                return f"Digital Twin processing event rejected unsafe {label} claim."
        return ""

    def _finished(self, exit_code: int, _status: QProcess.ExitStatus) -> None:
        self._kill_timer.stop()
        self._elapsed_timer.stop()
        if self._buffer.strip():
            self._consume_line(self._buffer.strip())
        self._buffer = ""
        if self._snapshot.spec is None:
            return

        elapsed = self._elapsed_seconds()
        if self._cancel_requested:
            lifecycle = DigitalTwinProcessingLifecycle.CANCELLED
            stage = "cancelled"
            message = "Digital Twin processing job cancelled; canonical workspace remains available."
            error = ""
        elif self._protocol_failure:
            lifecycle = DigitalTwinProcessingLifecycle.FAILED
            stage = "failed"
            message = self._protocol_failure
            error = self._protocol_failure
        elif int(exit_code) == 0:
            lifecycle = DigitalTwinProcessingLifecycle.COMPLETED
            stage = "completed"
            message = self._snapshot.message or "Digital Twin processing job completed."
            error = ""
        else:
            lifecycle = DigitalTwinProcessingLifecycle.FAILED
            stage = "failed"
            message = f"Digital Twin processing job exited with code {int(exit_code)}."
            error = message

        final = update_digital_twin_processing_snapshot(
            self._snapshot,
            lifecycle=lifecycle,
            stage=stage,
            message=message,
            progress_fraction=(1.0 if lifecycle is DigitalTwinProcessingLifecycle.COMPLETED else self._snapshot.progress_fraction),
            elapsed_seconds=elapsed,
            error=error,
        )
        self._set_snapshot(final)
        if lifecycle is DigitalTwinProcessingLifecycle.COMPLETED:
            self.result_ready.emit(final)
        elif lifecycle is DigitalTwinProcessingLifecycle.FAILED:
            self.error_occurred.emit(error)
        self.run_finished.emit(final)
        self._started_at = None
        self._cancel_requested = False
        self._protocol_failure = ""
        self._command = None

    def _process_error(self, _error: QProcess.ProcessError) -> None:
        if self._cancel_requested or self._shutdown_started:
            return
        message = self._process.errorString() or "Digital Twin processing process error."
        self.error_occurred.emit(message)

    def _fail_start(self, message: str) -> None:
        if self._snapshot.spec is not None:
            self._set_snapshot(
                update_digital_twin_processing_snapshot(
                    self._snapshot,
                    lifecycle=DigitalTwinProcessingLifecycle.FAILED,
                    stage="failed_to_start",
                    message=message,
                    elapsed_seconds=self._elapsed_seconds(),
                    error=message,
                )
            )
        self._started_at = None
        self._command = None

    def _terminate_process_group(self) -> None:
        if not self.is_running:
            return
        pid = self.process_id
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
        pid = self.process_id
        if pid > 0 and os.name == "posix" and hasattr(os, "killpg"):
            try:
                os.killpg(pid, signal.SIGKILL)
                return
            except (ProcessLookupError, PermissionError, OSError):
                pass
        self._process.kill()


__all__ = [
    "DEFAULT_DIGITAL_TWIN_PROCESS_POLICY",
    "DigitalTwinProcessingJobController",
]
