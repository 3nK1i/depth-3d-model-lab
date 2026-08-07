"""External-process adapter for the qualified stella_vslam 0.3.13 worker.

The production ``depth`` Python process never imports or links stella_vslam.
Pixels cross only through the bounded shared-memory ring fixed in Phase 8.4A;
small control/results use the versioned AF_UNIX JSON protocol.
"""

from __future__ import annotations

import os
import signal
from pathlib import Path
import shutil
import subprocess
import tempfile
from time import monotonic
from typing import Mapping
from uuid import uuid4

from core.contracts.validation import metadata_as_dict
from core.reconstruction.backends import (
    BackendCapabilities,
    BackendOperationContext,
    ReconstructionBackendKind,
)
from core.reconstruction.contracts import InputSourceKind
from core.reconstruction.errors import (
    LiveFrameStaleError,
    LiveSlamBackendUnavailableError,
    LiveSlamProtocolError,
    LiveSlamWorkerExecutionError,
    ReconstructionCancelledError,
    ReconstructionValidationError,
)
from core.reconstruction.events import ReconstructionStage
from core.reconstruction.live_contracts import (
    LiveSlamFrame,
    LiveSlamObservation,
    LiveSlamRequest,
)
from core.reconstruction.live_process import (
    LiveSlamJsonSocketClient,
    LiveSlamWorkerLifecycle,
    LiveSlamWorkerProcessSpec,
    LiveSlamWorkerState,
)
from core.reconstruction.live_protocol import (
    LiveSlamMessageType,
    LiveSlamProtocolMessage,
    make_live_control_message,
    make_live_frame_available_message,
    make_live_start_message,
    parse_live_observation_message,
    parse_live_worker_error,
)
from core.reconstruction.live_stella_environment import (
    STELLA_LIVE_BACKEND_ID,
    STELLA_LIVE_BACKEND_VERSION,
    STELLA_REQUIRED_VERSION,
    StellaLiveEnvironmentProbe,
    probe_stella_live_environment,
    write_stella_monocular_config,
)
from core.reconstruction.live_transport import LiveSharedMemoryRingSpec
from core.reconstruction.profiles import ReconstructionQualityProfile


class ExternalStellaLiveSlamBackend:
    """Synchronous backend boundary around one isolated native worker session."""

    def __init__(
        self,
        *,
        install_prefix: str | Path,
        worker_executable: str | Path,
        vocabulary_path: str | Path,
        frame_ring_spec: LiveSharedMemoryRingSpec,
        startup_timeout_s: float = 15.0,
        frame_timeout_s: float = 5.0,
        shutdown_timeout_s: float = 5.0,
        require_qualified_build: bool = True,
    ) -> None:
        if not isinstance(frame_ring_spec, LiveSharedMemoryRingSpec):
            raise TypeError("frame_ring_spec must be a LiveSharedMemoryRingSpec.")
        self._install_prefix = Path(install_prefix).expanduser().resolve()
        self._worker_executable = Path(worker_executable).expanduser().resolve()
        self._vocabulary_path = Path(vocabulary_path).expanduser().resolve()
        self._ring_spec = frame_ring_spec
        self._startup_timeout_s = float(startup_timeout_s)
        self._frame_timeout_s = float(frame_timeout_s)
        self._shutdown_timeout_s = float(shutdown_timeout_s)
        if min(self._startup_timeout_s, self._frame_timeout_s, self._shutdown_timeout_s) <= 0.0:
            raise ReconstructionValidationError("stella live worker timeouts must be positive.")
        self._require_qualified_build = bool(require_qualified_build)
        self._environment_probe: StellaLiveEnvironmentProbe | None = None
        self._lifecycle = LiveSlamWorkerLifecycle()
        self._process: subprocess.Popen[bytes] | None = None
        self._client: LiveSlamJsonSocketClient | None = None
        self._process_spec: LiveSlamWorkerProcessSpec | None = None
        self._active_request: LiveSlamRequest | None = None
        self._worker_session_id = ""
        self._tempdir: tempfile.TemporaryDirectory[str] | None = None
        self._log_handle = None
        self._log_path: Path | None = None
        self._outgoing_sequence = 0
        self._last_worker_sequence = -1

    @property
    def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(
            backend_id=STELLA_LIVE_BACKEND_ID,
            display_name="stella_vslam 0.3.13 Live SLAM (isolated)",
            backend_version=STELLA_LIVE_BACKEND_VERSION,
            kinds=(ReconstructionBackendKind.LIVE_SLAM,),
            source_kinds=(
                InputSourceKind.USB_CAMERA,
                InputSourceKind.RTSP_STREAM,
                InputSourceKind.RECORDED_VIDEO,
            ),
            quality_profiles=(
                ReconstructionQualityProfile.EFFICIENT,
                ReconstructionQualityProfile.BALANCED,
                ReconstructionQualityProfile.HIGH_QUALITY,
            ),
            supports_cancellation=True,
            supports_pause=False,
            supports_checkpoint=False,
            supports_resume=False,
            supports_metric_scale=False,
            supports_loop_closure=False,
            supports_cpu=True,
            requires_cuda=False,
            optional_dependencies=("stella_vslam==0.3.13 (isolated native worker)",),
        )

    @property
    def environment_probe(self) -> StellaLiveEnvironmentProbe | None:
        return self._environment_probe

    @property
    def worker_state(self) -> LiveSlamWorkerState:
        return self._lifecycle.state

    @property
    def worker_log_path(self) -> str:
        return "" if self._log_path is None else str(self._log_path)
    @property
    def worker_pid(self) -> int | None:
        process = self._process
        return None if process is None else process.pid

    @property
    def unix_socket_path(self) -> str:
        spec = self._process_spec
        return "" if spec is None else spec.unix_socket_path

    @property
    def resources_active(self) -> bool:
        return bool(self._process is not None or self._client is not None or self._tempdir is not None)

    def prepare(self, context: BackendOperationContext) -> None:
        _require_live_context(context)
        context.control.checkpoint()
        probe = probe_stella_live_environment(
            self._install_prefix,
            self._worker_executable,
            self._vocabulary_path,
            require_qualified_build=self._require_qualified_build,
        )
        self._environment_probe = probe
        if not probe.available:
            raise LiveSlamBackendUnavailableError(
                f"Isolated stella_vslam live backend is unavailable: {probe.reason}"
            )

    def start_live_session(
        self,
        request: LiveSlamRequest,
        context: BackendOperationContext,
    ) -> None:
        _require_live_context(context)
        if not isinstance(request, LiveSlamRequest):
            raise TypeError("request must be a LiveSlamRequest.")
        if request.backend_id != STELLA_LIVE_BACKEND_ID:
            raise ReconstructionValidationError(
                "Live SLAM request backend_id does not match isolated stella_vslam backend."
            )
        if request.session_id != context.session_id:
            raise ReconstructionValidationError(
                "Live SLAM request session_id does not match operation context."
            )
        if request.enable_loop_closure or request.enable_global_optimization:
            raise ReconstructionValidationError(
                "Phase 8.4B cannot enable loop closure or global optimization."
            )
        if self._active_request is not None or self._process is not None:
            raise ReconstructionValidationError(
                "This isolated stella_vslam backend already owns a worker process/session."
            )
        context.control.checkpoint()
        if self._environment_probe is None:
            self.prepare(context)
        assert self._environment_probe is not None

        self._tempdir = tempfile.TemporaryDirectory(prefix="eia-live-stella-")
        temp_root = Path(self._tempdir.name)
        config_path = write_stella_monocular_config(request, temp_root / "camera.yaml")
        self._log_path = temp_root / "worker.log"
        self._log_handle = self._log_path.open("wb")
        socket_path = Path("/tmp") / f"eia-ls-{os.getpid()}-{uuid4().hex[:8]}.sock"
        self._process_spec = LiveSlamWorkerProcessSpec(
            backend_id=STELLA_LIVE_BACKEND_ID,
            executable_path=str(self._worker_executable),
            unix_socket_path=str(socket_path),
            startup_timeout_s=self._startup_timeout_s,
            frame_timeout_s=self._frame_timeout_s,
            shutdown_timeout_s=self._shutdown_timeout_s,
        )
        self._lifecycle = LiveSlamWorkerLifecycle()
        self._worker_session_id = request.session_id
        self._lifecycle.transition_to(LiveSlamWorkerState.STARTING)
        self._outgoing_sequence = 0
        self._last_worker_sequence = -1

        env = _isolated_native_environment(self._install_prefix)
        command = [
            str(self._worker_executable),
            "--socket",
            str(socket_path),
            "--session-id",
            request.session_id,
            "--vocab",
            str(self._vocabulary_path),
            "--config",
            str(config_path),
            "--backend-id",
            STELLA_LIVE_BACKEND_ID,
        ]
        try:
            self._process = subprocess.Popen(
                command,
                stdin=subprocess.DEVNULL,
                stdout=self._log_handle,
                stderr=subprocess.STDOUT,
                cwd=str(temp_root),
                env=env,
                start_new_session=True,
            )
            self._client = LiveSlamJsonSocketClient.connect(
                socket_path,
                timeout_s=self._startup_timeout_s,
            )
            hello = self._receive_with_control(
                context,
                timeout_s=self._startup_timeout_s,
                allowed_types={LiveSlamMessageType.HELLO},
            )
            self._validate_hello(hello)
            ready = self._receive_with_control(
                context,
                timeout_s=self._startup_timeout_s,
                allowed_types={LiveSlamMessageType.READY},
            )
            self._validate_ready(ready)
            self._lifecycle.transition_to(LiveSlamWorkerState.READY)

            self._client.send_message(
                make_live_start_message(
                    request,
                    self._ring_spec,
                    sequence_number=self._next_outgoing_sequence(),
                )
            )
            started = self._receive_with_control(
                context,
                timeout_s=self._startup_timeout_s,
                allowed_types={LiveSlamMessageType.STARTED, LiveSlamMessageType.ERROR},
            )
            if started.message_type is LiveSlamMessageType.ERROR:
                self._raise_worker_error(started)
            self._validate_started(started)
            self._active_request = request
            self._lifecycle.transition_to(LiveSlamWorkerState.RUNNING)
        except ReconstructionCancelledError:
            self._abort_for_cancellation()
            raise
        except Exception as exc:
            self._mark_failed(str(exc))
            self._terminate_process()
            self._cleanup_resources()
            if isinstance(
                exc,
                (
                    LiveSlamBackendUnavailableError,
                    LiveSlamProtocolError,
                    LiveSlamWorkerExecutionError,
                    ReconstructionValidationError,
                ),
            ):
                raise
            raise LiveSlamWorkerExecutionError(
                f"Could not start isolated stella_vslam live worker: {exc}"
            ) from exc

    def track_live_frame(
        self,
        frame: LiveSlamFrame,
        context: BackendOperationContext,
    ) -> LiveSlamObservation:
        _require_live_context(context)
        if not isinstance(frame, LiveSlamFrame):
            raise TypeError("frame must be a LiveSlamFrame.")
        request = self._active_request
        client = self._client
        if request is None or client is None or self._lifecycle.state is not LiveSlamWorkerState.RUNNING:
            raise ReconstructionValidationError("stella_vslam live session has not been started.")
        if frame.session_id != request.session_id:
            raise ReconstructionValidationError("Live frame session_id does not match active session.")
        if frame.source_id != request.source.source_id:
            raise ReconstructionValidationError("Live frame source_id does not match active source.")
        if frame.camera_id != request.camera.camera_id:
            raise ReconstructionValidationError("Live frame camera_id does not match calibrated camera.")
        if frame.image_size != request.camera.image_size:
            raise ReconstructionValidationError("Live frame image size does not match calibration.")
        try:
            context.control.checkpoint()
        except ReconstructionCancelledError:
            self._abort_for_cancellation()
            raise
        client.send_message(
            make_live_frame_available_message(
                frame,
                sequence_number=self._next_outgoing_sequence(),
            )
        )
        try:
            response = self._receive_with_control(
                context,
                timeout_s=self._frame_timeout_s,
                allowed_types={
                    LiveSlamMessageType.OBSERVATION,
                    LiveSlamMessageType.ERROR,
                    LiveSlamMessageType.HEARTBEAT,
                },
                skip_heartbeats=True,
            )
            if response.message_type is LiveSlamMessageType.ERROR:
                self._raise_worker_error(response)
            observation = parse_live_observation_message(
                response,
                expected_session_id=request.session_id,
                expected_frame=frame,
            )
            if observation.pose is not None:
                if observation.pose.camera_id != request.camera.camera_id:
                    raise LiveSlamProtocolError("Native pose camera_id does not match calibration.")
                if observation.pose.camera_frame_id != request.camera.camera_frame.frame_id:
                    raise LiveSlamProtocolError("Native pose camera frame does not match calibration.")
                if observation.pose.timestamp_ns != frame.capture_timestamp_ns:
                    raise LiveSlamProtocolError("Native pose timestamp does not match submitted frame.")
            context.control.checkpoint()
            return observation
        except ReconstructionCancelledError:
            self._abort_for_cancellation()
            raise
        except LiveFrameStaleError:
            raise
        except (LiveSlamProtocolError, LiveSlamWorkerExecutionError) as exc:
            self._mark_failed(str(exc))
            self._terminate_process()
            self._active_request = None
            self._cleanup_resources()
            raise

    def stop_live_session(self, context: BackendOperationContext) -> None:
        _require_live_context(context)
        if self._active_request is None:
            return
        if self._client is None or self._lifecycle.state is not LiveSlamWorkerState.RUNNING:
            raise LiveSlamWorkerExecutionError("Active live session has no healthy worker channel.")
        context.control.checkpoint()
        self._client.send_message(
            make_live_control_message(
                LiveSlamMessageType.STOP,
                self._active_request.session_id,
                sequence_number=self._next_outgoing_sequence(),
            )
        )
        try:
            response = self._receive_with_control(
                context,
                timeout_s=self._shutdown_timeout_s,
                allowed_types={LiveSlamMessageType.STOPPED, LiveSlamMessageType.ERROR},
            )
        except ReconstructionCancelledError:
            self._abort_for_cancellation()
            raise
        if response.message_type is LiveSlamMessageType.ERROR:
            self._raise_worker_error(response)
        self._active_request = None
        self._lifecycle.transition_to(LiveSlamWorkerState.READY)

    def shutdown(self) -> None:
        if self._process is None:
            self._cleanup_resources()
            if self._lifecycle.state is LiveSlamWorkerState.CREATED:
                self._lifecycle.transition_to(LiveSlamWorkerState.STOPPED)
            elif self._lifecycle.state is LiveSlamWorkerState.FAILED:
                if self._lifecycle.can_transition_to(LiveSlamWorkerState.STOPPED):
                    self._lifecycle.transition_to(LiveSlamWorkerState.STOPPED)
            return
        session_id = self._worker_session_id
        if self._lifecycle.state in {
            LiveSlamWorkerState.STARTING,
            LiveSlamWorkerState.READY,
            LiveSlamWorkerState.RUNNING,
            LiveSlamWorkerState.FAILED,
        } and self._lifecycle.can_transition_to(LiveSlamWorkerState.STOPPING):
            self._lifecycle.transition_to(LiveSlamWorkerState.STOPPING)
        if self._client is not None and session_id:
            try:
                self._client.send_message(
                    make_live_control_message(
                        LiveSlamMessageType.SHUTDOWN,
                        session_id,
                        sequence_number=self._next_outgoing_sequence(),
                    )
                )
            except Exception:
                pass
        process = self._process
        try:
            process.wait(timeout=self._shutdown_timeout_s)
        except subprocess.TimeoutExpired:
            self._terminate_process()
        finally:
            self._active_request = None
            self._cleanup_resources()
        if self._lifecycle.state is LiveSlamWorkerState.STOPPING:
            self._lifecycle.transition_to(LiveSlamWorkerState.STOPPED)
        elif self._lifecycle.state is LiveSlamWorkerState.FAILED:
            if self._lifecycle.can_transition_to(LiveSlamWorkerState.STOPPED):
                self._lifecycle.transition_to(LiveSlamWorkerState.STOPPED)

    def _validate_hello(self, message: LiveSlamProtocolMessage) -> None:
        payload = _message_payload(message)
        if payload.get("backend_id") != STELLA_LIVE_BACKEND_ID:
            raise LiveSlamProtocolError("Native worker backend_id mismatch during HELLO.")
        if self._require_qualified_build and payload.get("stella_version") != STELLA_REQUIRED_VERSION:
            raise LiveSlamProtocolError("Native worker stella_vslam version is not the qualified 0.3.13.")
        if payload.get("pose_input_convention") != "T_camera_from_reconstruction":
            raise LiveSlamProtocolError("Native worker must declare stella camera-from-map input pose convention.")
        if payload.get("pose_output_convention") != "T_reconstruction_from_camera":
            raise LiveSlamProtocolError("Native worker must declare EIA camera-to-reconstruction output convention.")

    def _validate_ready(self, message: LiveSlamProtocolMessage) -> None:
        payload = _message_payload(message)
        if payload.get("protocol_version") != 1:
            raise LiveSlamProtocolError("Native worker READY protocol version mismatch.")

    def _validate_started(self, message: LiveSlamProtocolMessage) -> None:
        payload = _message_payload(message)
        if payload.get("loop_detector_enabled") is not False:
            raise LiveSlamProtocolError("Phase 8.4B requires the stella loop detector to be disabled.")
        if payload.get("phase8_global_optimization_feature_enabled") is not False:
            raise LiveSlamProtocolError("Phase 8.4B cannot enable global optimization feature ownership.")
        if payload.get("scale_status") != "arbitrary" or payload.get("unit_to_metre") is not None:
            raise LiveSlamProtocolError("Native monocular worker must report arbitrary scale.")

    def _receive_with_control(
        self,
        context: BackendOperationContext,
        *,
        timeout_s: float,
        allowed_types: set[LiveSlamMessageType],
        skip_heartbeats: bool = False,
    ) -> LiveSlamProtocolMessage:
        client = self._client
        process = self._process
        if client is None or process is None:
            raise LiveSlamWorkerExecutionError("Live SLAM worker channel is not active.")
        deadline = monotonic() + timeout_s
        while True:
            context.control.checkpoint()
            return_code = process.poll()
            if return_code is not None:
                raise LiveSlamWorkerExecutionError(
                    f"Isolated live SLAM worker exited unexpectedly with code {return_code}."
                    f"{self._log_tail_suffix()}"
                )
            remaining = deadline - monotonic()
            if remaining <= 0.0:
                raise LiveSlamWorkerExecutionError(
                    f"Timed out waiting for isolated live SLAM worker response.{self._log_tail_suffix()}"
                )
            try:
                message = client.receive_message(timeout_s=min(0.1, remaining))
            except TimeoutError:
                continue
            self._validate_worker_sequence(message)
            if message.session_id != context.session_id:
                raise LiveSlamProtocolError("Native worker response session_id mismatch.")
            if skip_heartbeats and message.message_type is LiveSlamMessageType.HEARTBEAT:
                continue
            if message.message_type not in allowed_types:
                raise LiveSlamProtocolError(
                    f"Unexpected live SLAM worker message: {message.message_type.value}."
                )
            return message

    def _validate_worker_sequence(self, message: LiveSlamProtocolMessage) -> None:
        if message.sequence_number <= self._last_worker_sequence:
            raise LiveSlamProtocolError("Native worker message sequence is not strictly increasing.")
        self._last_worker_sequence = message.sequence_number

    def _raise_worker_error(self, message: LiveSlamProtocolMessage) -> None:
        payload = parse_live_worker_error(message)
        code = str(payload["code"])
        text = str(payload["message"])
        if code == "stale_frame":
            raise LiveFrameStaleError(text)
        raise LiveSlamWorkerExecutionError(f"Native live SLAM worker error [{code}]: {text}")

    def _next_outgoing_sequence(self) -> int:
        value = self._outgoing_sequence
        self._outgoing_sequence += 1
        return value

    def _abort_for_cancellation(self) -> None:
        if self._lifecycle.can_transition_to(LiveSlamWorkerState.STOPPING):
            self._lifecycle.transition_to(LiveSlamWorkerState.STOPPING)
        self._terminate_process()
        self._active_request = None
        self._cleanup_resources()
        if self._lifecycle.state is LiveSlamWorkerState.STOPPING:
            self._lifecycle.transition_to(LiveSlamWorkerState.STOPPED)

    def _mark_failed(self, reason: str) -> None:
        try:
            self._lifecycle.fail(reason or "native worker failure")
        except ReconstructionValidationError:
            pass

    def _terminate_process(self) -> None:
        process = self._process
        if process is None:
            return
        if process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except (ProcessLookupError, PermissionError, OSError):
                process.terminate()
            try:
                process.wait(timeout=min(2.0, self._shutdown_timeout_s))
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except (ProcessLookupError, PermissionError, OSError):
                    process.kill()
                process.wait(timeout=2.0)

    def _cleanup_resources(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None
        if self._process_spec is not None:
            try:
                Path(self._process_spec.unix_socket_path).unlink(missing_ok=True)
            except OSError:
                pass
        self._process = None
        self._process_spec = None
        self._worker_session_id = ""
        if self._log_handle is not None:
            try:
                self._log_handle.close()
            except OSError:
                pass
            self._log_handle = None
        if self._tempdir is not None:
            self._tempdir.cleanup()
            self._tempdir = None
        self._log_path = None

    def _log_tail_suffix(self) -> str:
        if self._log_handle is not None:
            try:
                self._log_handle.flush()
            except OSError:
                pass
        path = self._log_path
        if path is None or not path.is_file():
            return ""
        try:
            lines = path.read_text(encoding="utf-8", errors="replace").splitlines()[-8:]
        except OSError:
            return ""
        if not lines:
            return ""
        return " Worker log tail: " + " | ".join(line.strip() for line in lines if line.strip())


def _isolated_native_environment(prefix: Path) -> dict[str, str]:
    env = dict(os.environ)
    env["PATH"] = "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
    env["LD_LIBRARY_PATH"] = f"{prefix / 'lib'}:{prefix / 'lib64'}"
    for key in ("PYTHONPATH", "CONDA_PREFIX", "CONDA_DEFAULT_ENV", "CONDA_PROMPT_MODIFIER"):
        env.pop(key, None)
    return env


def _message_payload(message: LiveSlamProtocolMessage) -> Mapping[str, object]:
    payload = metadata_as_dict(message.payload)  # type: ignore[arg-type]
    if not isinstance(payload, Mapping):
        raise LiveSlamProtocolError("Native worker message payload must be an object.")
    return payload


def _require_live_context(context: BackendOperationContext) -> None:
    if not isinstance(context, BackendOperationContext):
        raise TypeError("context must be BackendOperationContext.")
    if context.stage is not ReconstructionStage.LIVE_TRACKING:
        raise ReconstructionValidationError("stella_vslam live backend requires LIVE_TRACKING stage.")
