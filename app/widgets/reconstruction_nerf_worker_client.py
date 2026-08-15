"""Qt-side Phase 9.5G-hardened client for the isolated interactive Nerfacto renderer.

Only lightweight protocol/process supervision lives here.  Nerfstudio, torch and
tiny-cuda-nn remain entirely inside the external worker interpreter.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import signal
from typing import Any

from PyQt6.QtCore import QObject, QProcess, QProcessEnvironment, QTimer, pyqtSignal

from core.reconstruction.nerf_interactive_runtime import default_phase9_nerf_runtime_spec
from app.reconstruction_nerf_resources import LatestRenderRequestQueue

PROTOCOL_PREFIX = "EIA_NERF_JSON "


class ReconstructionNerfWorkerClient(QObject):
    ready = pyqtSignal(object)
    frame_ready = pyqtSignal(object)
    failed = pyqtSignal(str)
    state_changed = pyqtSignal(str)

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._process = QProcess(self)
        self._process.setProcessChannelMode(QProcess.ProcessChannelMode.SeparateChannels)
        self._process.readyReadStandardOutput.connect(self._consume_stdout)
        self._process.readyReadStandardError.connect(self._consume_stderr)
        self._process.errorOccurred.connect(self._on_process_error)
        self._process.finished.connect(self._on_process_finished)
        self._stdout_buffer = ""
        self._stderr_tail: list[str] = []
        self._load_spec_path = ""
        self._session_dir = ""
        self._state = "stopped"
        self._intentional_stop = False
        self._ready = False
        self._last_request_id = -1
        self._render_queue = LatestRenderRequestQueue()
        self._worker_pid = 0
        self._last_telemetry: dict[str, Any] = {}
        self._kill_timer = QTimer(self)
        self._kill_timer.setSingleShot(True)
        self._kill_timer.timeout.connect(self._force_kill_process_group)

    @property
    def state(self) -> str:
        return self._state

    @property
    def is_running(self) -> bool:
        return self._process.state() != QProcess.ProcessState.NotRunning

    @property
    def is_ready(self) -> bool:
        return bool(self._ready and self.is_running)

    @property
    def load_spec_path(self) -> str:
        return self._load_spec_path

    @property
    def last_request_id(self) -> int:
        return int(self._last_request_id)

    @property
    def stderr_tail(self) -> tuple[str, ...]:
        return tuple(self._stderr_tail[-12:])

    @property
    def queue_snapshot(self):
        return self._render_queue.snapshot

    @property
    def worker_pid(self) -> int:
        return int(self._worker_pid)

    @property
    def last_telemetry(self) -> dict[str, Any]:
        return dict(self._last_telemetry)

    def start(self, *, load_spec_path: str | Path, session_dir: str | Path, worker_script: str | Path) -> None:
        load_spec = Path(load_spec_path).expanduser().resolve()
        session = Path(session_dir).expanduser().resolve()
        worker = Path(worker_script).expanduser().resolve()
        if not load_spec.is_file():
            raise ValueError(f"Interactive NeRF runtime load spec is unavailable: {load_spec}")
        if not worker.is_file():
            raise ValueError(f"Interactive NeRF worker script is unavailable: {worker}")
        if self.is_running:
            if self._load_spec_path == str(load_spec) and self._session_dir == str(session):
                return
            self.stop()
            raise RuntimeError("Interactive NeRF worker is stopping; retry after it exits.")

        runtime = default_phase9_nerf_runtime_spec()
        python = Path(runtime.python_executable).expanduser().resolve()
        if not python.is_file():
            raise ValueError(f"Qualified isolated NeRF interpreter is unavailable: {python}")
        setsid = shutil.which("setsid")
        if not setsid:
            raise RuntimeError("Phase 9.5G requires the POSIX setsid launcher for isolated process-group cleanup.")
        session.mkdir(parents=True, exist_ok=True)
        self._load_spec_path = str(load_spec)
        self._session_dir = str(session)
        self._stdout_buffer = ""
        self._stderr_tail.clear()
        self._ready = False
        self._intentional_stop = False
        self._last_request_id = -1
        self._render_queue.reset()
        self._worker_pid = 0
        self._last_telemetry = {}

        env = dict(os.environ)
        prefix = python.parent.parent
        env["CONDA_PREFIX"] = str(prefix)
        env["CONDA_DEFAULT_ENV"] = prefix.name
        env["PYTHONNOUSERSITE"] = "1"
        env.pop("PYTHONPATH", None)
        env["EIA_NERFSTUDIO_ROOT"] = runtime.nerfstudio_root
        env["EIA_TCNN_ROOT"] = runtime.tiny_cuda_nn_root

        def prepend(key: str, paths: list[Path]) -> None:
            parts = [str(path) for path in paths if path.is_dir()]
            existing = env.get(key, "")
            if existing:
                parts.append(existing)
            env[key] = os.pathsep.join(parts)

        prepend("PATH", [prefix / "bin"])
        nvcc = prefix / "bin" / "nvcc"
        if nvcc.is_file():
            env["CUDA_HOME"] = str(prefix)
            env["CUDACXX"] = str(nvcc)
        libs = [prefix / "lib", prefix / "lib64", prefix / "targets" / "x86_64-linux" / "lib"]
        prepend("LD_LIBRARY_PATH", libs)
        prepend("LIBRARY_PATH", libs)
        process_env = QProcessEnvironment()
        for key, value in env.items():
            process_env.insert(str(key), str(value))
        self._process.setProcessEnvironment(process_env)
        self._process.setWorkingDirectory(str(worker.parent))
        self._process.setProgram(str(setsid))
        self._process.setArguments(
            [
                str(python),
                str(worker),
                "--load-spec",
                str(load_spec),
                "--session-dir",
                str(session),
            ]
        )
        self._set_state("starting")
        self._process.start()

    def _dispatch_render(self, payload: dict[str, Any]) -> None:
        encoded = (json.dumps(payload, separators=(",", ":")) + "\n").encode("utf-8")
        written = self._process.write(encoded)
        if written < 0:
            raise RuntimeError("Interactive NeRF render request could not be queued.")
        self._set_state("rendering")

    def request_render(self, payload: dict[str, Any]) -> str:
        """Submit a render using bounded latest-view coalescing.

        At most one request is executing in the external renderer and at most one
        newest request is retained. Intermediate viewpoints are dropped rather than
        building an unbounded stdin/render backlog.
        """

        if not self.is_ready:
            raise RuntimeError("Interactive NeRF renderer is not ready.")
        if str(payload.get("type", "")) != "render":
            raise ValueError("Interactive NeRF client accepts render payloads only.")
        request_id = int(payload.get("request_id", -1))
        self._last_request_id = max(self._last_request_id, request_id)
        dispatch, item = self._render_queue.submit(payload)
        if not dispatch:
            return "coalesced"
        assert item is not None
        self._dispatch_render(item)
        return "dispatched"

    def stop(self) -> None:
        if not self.is_running:
            self._ready = False
            self._set_state("stopped")
            return
        self._intentional_stop = True
        self._ready = False
        self._render_queue.cancel_pending()
        try:
            self._process.write(b'{"type":"shutdown"}\n')
        except RuntimeError:
            pass
        self._set_state("stopping")
        QTimer.singleShot(250, self._terminate_process_group)

    def stop_blocking(self, timeout_ms: int = 3000) -> None:
        """Stop the isolated process group synchronously during app shutdown."""

        if not self.is_running:
            self._ready = False
            self._set_state("stopped")
            return
        self._intentional_stop = True
        self._ready = False
        self._render_queue.cancel_pending()
        self._kill_timer.stop()
        try:
            self._process.write(b'{"type":"shutdown"}\n')
            self._process.waitForBytesWritten(200)
        except RuntimeError:
            pass
        self._set_state("stopping")
        if self._process.waitForFinished(min(max(int(timeout_ms), 250), 1000)):
            return
        self._terminate_process_group()
        remaining = max(250, int(timeout_ms) - 1000)
        if self._process.waitForFinished(remaining):
            return
        self._force_kill_process_group()
        self._process.waitForFinished(500)

    def _set_state(self, state: str) -> None:
        state = str(state).strip()
        if state == self._state:
            return
        self._state = state
        self.state_changed.emit(state)

    def _consume_stdout(self) -> None:
        chunk = bytes(self._process.readAllStandardOutput()).decode("utf-8", errors="replace")
        self._stdout_buffer += chunk
        while "\n" in self._stdout_buffer:
            line, self._stdout_buffer = self._stdout_buffer.split("\n", 1)
            line = line.strip()
            if not line.startswith(PROTOCOL_PREFIX):
                continue
            try:
                payload = json.loads(line[len(PROTOCOL_PREFIX):])
            except json.JSONDecodeError:
                continue
            if not isinstance(payload, dict):
                continue
            kind = str(payload.get("type", ""))
            if kind == "ready" and payload.get("status") == "PASS":
                self._ready = True
                self._worker_pid = max(0, int(payload.get("worker_pid", 0) or 0))
                self._last_telemetry = dict(payload)
                self._set_state("ready")
                self.ready.emit(payload)
            elif kind == "frame" and payload.get("status") == "PASS":
                self._last_telemetry = dict(payload)
                request_id = int(payload.get("request_id", -1))
                pending = self._render_queue.complete(request_id)
                if request_id >= self._last_request_id:
                    self.frame_ready.emit(payload)
                if pending is not None and self._ready:
                    self._dispatch_render(pending)
                else:
                    self._set_state("ready")
            elif kind in {"error", "fatal"}:
                message = str(payload.get("error", "Interactive NeRF renderer failed")).strip()
                if kind == "fatal":
                    self._ready = False
                    self._render_queue.cancel_pending()
                    self._set_state("failed")
                else:
                    pending = self._render_queue.complete(int(payload.get("request_id", -1)))
                    if pending is not None and self._ready:
                        self._dispatch_render(pending)
                    else:
                        self._set_state("ready" if self._ready else "failed")
                self.failed.emit(message)
            elif kind == "stopped":
                self._ready = False
                self._set_state("stopping")

    def _consume_stderr(self) -> None:
        chunk = bytes(self._process.readAllStandardError()).decode("utf-8", errors="replace")
        for line in chunk.splitlines():
            if line.strip():
                self._stderr_tail.append(line.rstrip())
        if len(self._stderr_tail) > 60:
            del self._stderr_tail[:-60]

    def _on_process_error(self, _error) -> None:
        if self._intentional_stop:
            return
        tail = "\n".join(self.stderr_tail)
        message = self._process.errorString() or "Interactive NeRF worker process error"
        if tail:
            message = f"{message}\n{tail}"
        self._ready = False
        self._set_state("failed")
        self.failed.emit(message)

    def _on_process_finished(self, exit_code: int, _status) -> None:
        self._kill_timer.stop()
        was_intentional = self._intentional_stop
        self._ready = False
        self._intentional_stop = False
        self._render_queue.cancel_pending()
        self._worker_pid = 0
        self._last_telemetry = {}
        self._set_state("stopped")
        if not was_intentional and int(exit_code) != 0:
            tail = "\n".join(self.stderr_tail)
            message = f"Interactive NeRF worker exited with code {int(exit_code)}"
            if tail:
                message = f"{message}\n{tail}"
            self.failed.emit(message)

    def _terminate_process_group(self) -> None:
        if not self.is_running:
            return
        pid = int(self._process.processId())
        if pid <= 0:
            self._process.terminate()
            return
        try:
            os.killpg(pid, signal.SIGTERM)
        except (ProcessLookupError, PermissionError):
            self._process.terminate()
        self._kill_timer.start(2000)

    def _force_kill_process_group(self) -> None:
        if not self.is_running:
            return
        pid = int(self._process.processId())
        if pid > 0:
            try:
                os.killpg(pid, signal.SIGKILL)
                return
            except (ProcessLookupError, PermissionError):
                pass
        self._process.kill()


__all__ = ["ReconstructionNerfWorkerClient"]
