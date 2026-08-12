"""Non-blocking hardened live-capture runtime for Phase 8.4D.

The capture/UI thread only offers the newest :class:`SourceFrame`. A dedicated
Python thread owns SourceFrame->shared-memory publication and synchronous live
SLAM calls. Worker execution/protocol failures use a small bounded restart
budget; the capture side remains latest-frame-only and never builds a backlog.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from threading import Condition, Lock, Thread
from time import monotonic
from typing import Protocol

from core.capture.frame import SourceFrame
from core.reconstruction.backends import BackendOperationContext, LiveSlamBackend
from core.reconstruction.control import ReconstructionOperationControl
from core.reconstruction.errors import (
    LiveFrameStaleError,
    LiveSlamProtocolError,
    LiveSlamWorkerExecutionError,
    ReconstructionCancelledError,
    ReconstructionValidationError,
)
from core.reconstruction.events import ReconstructionEvent, ReconstructionStage
from core.reconstruction.live_contracts import (
    LiveSlamObservation,
    LiveSlamRequest,
    LiveTrackingState,
)
from core.reconstruction.live_frames import LiveSourceFrameAdapter
from core.reconstruction.live_hardening import (
    LiveSlamHardeningTelemetry,
    LiveSlamRecoveryPolicy,
)
from core.reconstruction.live_transport import LiveSharedMemoryFrameRing, LiveSharedMemoryRingSpec


class LiveCaptureRuntimeState(str, Enum):
    IDLE = "idle"
    STARTING = "starting"
    RUNNING = "running"
    PAUSED = "paused"
    RECOVERING = "recovering"
    STOPPING = "stopping"
    STOPPED = "stopped"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class LiveCaptureRuntimeStats:
    offered_frames: int
    processed_frames: int
    replaced_frames: int
    rejected_frames: int
    observations: int
    last_sequence_index: int | None
    latest_tracking_state: str
    last_error: str
    state: LiveCaptureRuntimeState
    stale_frames: int = 0
    worker_restarts: int = 0
    worker_restart_failures: int = 0
    tracking_loss_events: int = 0
    tracking_recoveries: int = 0
    stop_escalations: int = 0
    last_processing_latency_ns: int | None = None
    max_processing_latency_ns: int = 0
    mean_processing_latency_ns: float | None = None
    cleanup_complete: bool = False


class LiveBackendFactory(Protocol):
    def __call__(self, ring_spec: LiveSharedMemoryRingSpec) -> LiveSlamBackend: ...


class LiveCaptureSlamRuntime:
    """Own one live-SLAM session behind a latest-frame-only mailbox.

    Recovery is intentionally bounded. A worker crash, timeout, or malformed
    protocol response can restart only the external backend process. Capture
    ownership, shared-memory ownership, source identity, and session identity
    remain in this runtime.
    """

    def __init__(
        self,
        *,
        request: LiveSlamRequest,
        backend_factory: LiveBackendFactory,
        slot_count: int = 3,
        recovery_policy: LiveSlamRecoveryPolicy | None = None,
        observation_callback: Callable[[LiveSlamObservation], None] | None = None,
        state_callback: Callable[[LiveCaptureRuntimeState], None] | None = None,
        error_callback: Callable[[str], None] | None = None,
        event_callback: Callable[[ReconstructionEvent], None] | None = None,
    ) -> None:
        if not isinstance(request, LiveSlamRequest):
            raise TypeError("request must be LiveSlamRequest.")
        if not callable(backend_factory):
            raise TypeError("backend_factory must be callable.")
        if isinstance(slot_count, bool) or int(slot_count) < 2:
            raise ReconstructionValidationError("Live capture runtime requires at least two ring slots.")
        if recovery_policy is not None and not isinstance(recovery_policy, LiveSlamRecoveryPolicy):
            raise TypeError("recovery_policy must be LiveSlamRecoveryPolicy or None.")

        self._request = request
        self._backend_factory = backend_factory
        self._slot_count = int(slot_count)
        self._recovery_policy = recovery_policy or LiveSlamRecoveryPolicy()
        self._observation_callback = observation_callback
        self._state_callback = state_callback
        self._error_callback = error_callback
        self._event_callback = event_callback or (lambda _event: None)

        self._condition = Condition(Lock())
        self._stats_lock = Lock()
        self._pending: SourceFrame | None = None
        self._thread: Thread | None = None
        self._state = LiveCaptureRuntimeState.IDLE
        self._stop_requested = False
        self._paused = False
        self._control = ReconstructionOperationControl()
        self._backend: LiveSlamBackend | None = None
        self._ring: LiveSharedMemoryFrameRing | None = None
        self._adapter: LiveSourceFrameAdapter | None = None

        self._offered_frames = 0
        self._processed_frames = 0
        self._replaced_frames = 0
        self._rejected_frames = 0
        self._observations = 0
        self._last_sequence_index: int | None = None
        self._latest_tracking_state = ""
        self._last_error = ""
        self._stale_frames = 0
        self._consecutive_stale_frames = 0
        self._worker_restarts = 0
        self._worker_restart_failures = 0
        self._tracking_loss_events = 0
        self._tracking_recoveries = 0
        self._stop_escalations = 0
        self._last_processing_latency_ns: int | None = None
        self._max_processing_latency_ns = 0
        self._total_processing_latency_ns = 0
        self._cleanup_complete = False

    @property
    def request(self) -> LiveSlamRequest:
        return self._request

    @property
    def recovery_policy(self) -> LiveSlamRecoveryPolicy:
        return self._recovery_policy

    @property
    def state(self) -> LiveCaptureRuntimeState:
        with self._stats_lock:
            return self._state

    @property
    def is_active(self) -> bool:
        return self.state in {
            LiveCaptureRuntimeState.STARTING,
            LiveCaptureRuntimeState.RUNNING,
            LiveCaptureRuntimeState.PAUSED,
            LiveCaptureRuntimeState.RECOVERING,
            LiveCaptureRuntimeState.STOPPING,
        }

    @property
    def stats(self) -> LiveCaptureRuntimeStats:
        with self._stats_lock:
            mean_latency = (
                None
                if self._observations <= 0
                else self._total_processing_latency_ns / self._observations
            )
            return LiveCaptureRuntimeStats(
                offered_frames=self._offered_frames,
                processed_frames=self._processed_frames,
                replaced_frames=self._replaced_frames,
                rejected_frames=self._rejected_frames,
                observations=self._observations,
                last_sequence_index=self._last_sequence_index,
                latest_tracking_state=self._latest_tracking_state,
                last_error=self._last_error,
                state=self._state,
                stale_frames=self._stale_frames,
                worker_restarts=self._worker_restarts,
                worker_restart_failures=self._worker_restart_failures,
                tracking_loss_events=self._tracking_loss_events,
                tracking_recoveries=self._tracking_recoveries,
                stop_escalations=self._stop_escalations,
                last_processing_latency_ns=self._last_processing_latency_ns,
                max_processing_latency_ns=self._max_processing_latency_ns,
                mean_processing_latency_ns=mean_latency,
                cleanup_complete=self._cleanup_complete,
            )

    @property
    def hardening_telemetry(self) -> LiveSlamHardeningTelemetry:
        stats = self.stats
        return LiveSlamHardeningTelemetry(
            offered_frames=stats.offered_frames,
            processed_frames=stats.processed_frames,
            replaced_frames=stats.replaced_frames,
            rejected_frames=stats.rejected_frames,
            observations=stats.observations,
            stale_frames=stats.stale_frames,
            worker_restarts=stats.worker_restarts,
            worker_restart_failures=stats.worker_restart_failures,
            tracking_loss_events=stats.tracking_loss_events,
            tracking_recoveries=stats.tracking_recoveries,
            stop_escalations=stats.stop_escalations,
            last_processing_latency_ns=stats.last_processing_latency_ns,
            max_processing_latency_ns=stats.max_processing_latency_ns,
            mean_processing_latency_ns=stats.mean_processing_latency_ns,
            cleanup_complete=stats.cleanup_complete,
        )

    def start(self) -> None:
        with self._condition:
            if self._thread is not None and self._thread.is_alive():
                raise ReconstructionValidationError("Live capture SLAM runtime is already active.")
            self._stop_requested = False
            self._paused = False
            self._pending = None
            self._control = ReconstructionOperationControl()
            with self._stats_lock:
                self._cleanup_complete = False
                self._last_error = ""
            self._set_state_locked(LiveCaptureRuntimeState.STARTING)
            self._thread = Thread(
                target=self._run,
                name=f"live-slam-{self._request.session_id[:8]}",
                daemon=True,
            )
            self._thread.start()

    def offer_source_frame(self, frame: SourceFrame) -> bool:
        """Offer the newest capture frame; return True if it replaced a pending frame."""
        if not isinstance(frame, SourceFrame):
            raise TypeError("frame must be SourceFrame.")
        with self._condition:
            if self._state not in {
                LiveCaptureRuntimeState.STARTING,
                LiveCaptureRuntimeState.RUNNING,
                LiveCaptureRuntimeState.RECOVERING,
            }:
                with self._stats_lock:
                    self._rejected_frames += 1
                return False
            replaced = self._pending is not None
            self._pending = frame
            with self._stats_lock:
                self._offered_frames += 1
                if replaced:
                    self._replaced_frames += 1
            self._condition.notify_all()
            return replaced

    def pause(self) -> bool:
        with self._condition:
            if self._state not in {
                LiveCaptureRuntimeState.RUNNING,
                LiveCaptureRuntimeState.RECOVERING,
            }:
                return False
            self._paused = True
            self._pending = None
            self._set_state_locked(LiveCaptureRuntimeState.PAUSED)
            return True

    def resume(self) -> bool:
        with self._condition:
            if self._state is not LiveCaptureRuntimeState.PAUSED:
                return False
            self._paused = False
            self._set_state_locked(LiveCaptureRuntimeState.RUNNING)
            self._condition.notify_all()
            return True

    def request_stop(self) -> bool:
        """Request cooperative stop without blocking the caller/UI thread."""
        with self._condition:
            thread = self._thread
            if thread is None:
                if self._state is LiveCaptureRuntimeState.IDLE:
                    self._set_state_locked(LiveCaptureRuntimeState.STOPPED)
                return False
            if self._state not in {LiveCaptureRuntimeState.STOPPED, LiveCaptureRuntimeState.FAILED}:
                self._set_state_locked(LiveCaptureRuntimeState.STOPPING)
            self._stop_requested = True
            self._paused = False
            self._pending = None
            self._condition.notify_all()
            return True

    def stop(self, *, timeout_s: float = 7.0) -> bool:
        timeout = max(0.0, float(timeout_s))
        self.request_stop()
        with self._condition:
            thread = self._thread
        if thread is None:
            return True
        graceful_wait = min(1.0, timeout)
        thread.join(timeout=graceful_wait)
        remaining = max(0.0, timeout - graceful_wait)
        if thread.is_alive() and remaining > 0.0:
            with self._stats_lock:
                self._stop_escalations += 1
            self._control.request_cancel()
            with self._condition:
                self._condition.notify_all()
            thread.join(timeout=remaining)
        return not thread.is_alive()

    def wait_for_state(self, target: LiveCaptureRuntimeState, *, timeout_s: float = 5.0) -> bool:
        if not isinstance(target, LiveCaptureRuntimeState):
            raise TypeError("target must be LiveCaptureRuntimeState.")
        deadline = monotonic() + max(0.0, float(timeout_s))
        while monotonic() < deadline:
            if self.state is target:
                return True
            with self._condition:
                self._condition.wait(timeout=min(0.02, max(0.0, deadline - monotonic())))
        return self.state is target

    def _run(self) -> None:
        backend: LiveSlamBackend | None = None
        ring: LiveSharedMemoryFrameRing | None = None
        terminal_failure = False
        context = BackendOperationContext(
            session_id=self._request.session_id,
            stage=ReconstructionStage.LIVE_TRACKING,
            profile=self._request.profile,
            control=self._control,
            emit_event=self._event_callback,
        )
        try:
            max_payload = (
                self._request.camera.image_size.width
                * self._request.camera.image_size.height
                * 3
            )
            ring = LiveSharedMemoryFrameRing.create(
                slot_count=self._slot_count,
                max_payload_bytes=max_payload,
            )
            self._ring = ring
            self._adapter = LiveSourceFrameAdapter(
                session_id=self._request.session_id,
                source=self._request.source,
                camera=self._request.camera,
                ring=ring,
            )
            backend = self._create_started_backend(ring.spec, context)
            self._backend = backend
            with self._condition:
                if self._stop_requested:
                    raise ReconstructionCancelledError("Live capture stop requested during startup.")
                self._set_state_locked(LiveCaptureRuntimeState.RUNNING)
                self._condition.notify_all()

            while True:
                with self._condition:
                    while not self._stop_requested and (self._paused or self._pending is None):
                        self._condition.wait(timeout=0.1)
                    if self._stop_requested:
                        break
                    frame = self._pending
                    self._pending = None
                assert frame is not None
                assert self._adapter is not None
                envelope = self._adapter.publish(frame)
                try:
                    observation = backend.track_live_frame(envelope, context)
                except LiveFrameStaleError as exc:
                    if self._handle_stale_frame(exc):
                        continue
                    raise
                except (LiveSlamWorkerExecutionError, LiveSlamProtocolError) as exc:
                    backend = self._recover_backend(backend, ring.spec, context, exc)
                    self._backend = backend
                    continue

                self._record_observation(observation)
                if self._observation_callback is not None:
                    self._observation_callback(observation)

            try:
                backend.stop_live_session(context)
            except ReconstructionCancelledError:
                pass
            except (LiveSlamWorkerExecutionError, LiveSlamProtocolError):
                # Stop is best-effort here; shutdown below owns final process cleanup.
                pass
        except ReconstructionCancelledError:
            pass
        except Exception as exc:  # worker boundary: surface one controlled failure
            message = f"{type(exc).__name__}: {exc}"
            with self._stats_lock:
                self._last_error = message
            # H3 keeps the live runtime resource-owning until backend/shared-memory
            # cleanup is complete.  Publishing FAILED here would make application
            # resource guards see the runtime as idle while teardown was still
            # releasing native/process resources.  STOPPING remains active; FAILED
            # is emitted only after the finally block has completed cleanup.
            terminal_failure = True
            self._set_state(LiveCaptureRuntimeState.STOPPING)
            if self._error_callback is not None:
                self._error_callback(message)
        finally:
            if backend is not None:
                try:
                    backend.shutdown()
                except Exception:
                    pass
            if ring is not None:
                try:
                    ring.shutdown()
                except Exception:
                    pass
            self._backend = None
            self._ring = None
            self._adapter = None
            with self._stats_lock:
                self._cleanup_complete = True
            with self._condition:
                self._pending = None
                self._thread = None
                self._set_state_locked(
                    LiveCaptureRuntimeState.FAILED
                    if terminal_failure
                    else LiveCaptureRuntimeState.STOPPED
                )
                self._condition.notify_all()

    def _create_started_backend(
        self,
        ring_spec: LiveSharedMemoryRingSpec,
        context: BackendOperationContext,
    ) -> LiveSlamBackend:
        backend = self._backend_factory(ring_spec)
        if backend.capabilities.backend_id != self._request.backend_id:
            raise ReconstructionValidationError(
                "Live backend factory returned a backend that does not match request.backend_id."
            )
        if self._request.source.kind not in backend.capabilities.source_kinds:
            raise ReconstructionValidationError(
                "Live backend does not support the active capture source kind."
            )
        backend.prepare(context)
        backend.start_live_session(self._request, context)
        return backend

    def _recover_backend(
        self,
        backend: LiveSlamBackend,
        ring_spec: LiveSharedMemoryRingSpec,
        context: BackendOperationContext,
        cause: Exception,
    ) -> LiveSlamBackend:
        policy = self._recovery_policy
        last_exc: Exception = cause
        try:
            backend.shutdown()
        except Exception:
            pass

        while True:
            with self._stats_lock:
                if self._worker_restarts >= policy.max_worker_restarts:
                    break
                self._worker_restarts += 1
                self._latest_tracking_state = LiveTrackingState.RECOVERING.value
            self._set_state(LiveCaptureRuntimeState.RECOVERING)
            self._interruptible_backoff(policy.restart_backoff_s)
            try:
                recovered = self._create_started_backend(ring_spec, context)
            except ReconstructionCancelledError:
                raise
            except Exception as exc:
                last_exc = exc
                with self._stats_lock:
                    self._worker_restart_failures += 1
                continue
            with self._condition:
                self._set_state_locked(
                    LiveCaptureRuntimeState.PAUSED
                    if self._paused
                    else LiveCaptureRuntimeState.RUNNING
                )
                self._condition.notify_all()
            return recovered

        raise LiveSlamWorkerExecutionError(
            "Live SLAM worker recovery budget exhausted after "
            f"{policy.max_worker_restarts} restart attempt(s): {last_exc}"
        ) from last_exc

    def _interruptible_backoff(self, seconds: float) -> None:
        deadline = monotonic() + max(0.0, seconds)
        with self._condition:
            while not self._stop_requested and monotonic() < deadline:
                self._condition.wait(timeout=min(0.05, max(0.0, deadline - monotonic())))
            if self._stop_requested:
                raise ReconstructionCancelledError("Live SLAM stop requested during worker recovery.")
        self._control.checkpoint()

    def _handle_stale_frame(self, _exc: LiveFrameStaleError) -> bool:
        with self._stats_lock:
            self._stale_frames += 1
            self._consecutive_stale_frames += 1
            allowed = (
                self._consecutive_stale_frames
                <= self._recovery_policy.max_consecutive_stale_frames
            )
        return allowed

    def _record_observation(self, observation: LiveSlamObservation) -> None:
        latency = observation.quality.processing_latency_ns
        current_state = observation.tracking_state
        with self._stats_lock:
            previous_state = self._latest_tracking_state
            self._processed_frames += 1
            self._observations += 1
            self._last_sequence_index = observation.sequence_index
            self._latest_tracking_state = current_state.value
            self._consecutive_stale_frames = 0
            self._last_processing_latency_ns = latency
            self._max_processing_latency_ns = max(self._max_processing_latency_ns, latency)
            self._total_processing_latency_ns += latency
            if current_state is LiveTrackingState.LOST and previous_state != LiveTrackingState.LOST.value:
                self._tracking_loss_events += 1
            if current_state is LiveTrackingState.TRACKING and previous_state in {
                LiveTrackingState.LOST.value,
                LiveTrackingState.DEGRADED.value,
                LiveTrackingState.RECOVERING.value,
            }:
                self._tracking_recoveries += 1

    def _set_state(self, state: LiveCaptureRuntimeState) -> None:
        with self._condition:
            self._set_state_locked(state)
            self._condition.notify_all()

    def _set_state_locked(self, state: LiveCaptureRuntimeState) -> None:
        callback = None
        with self._stats_lock:
            if self._state is state:
                return
            self._state = state
            callback = self._state_callback
        if callback is not None:
            callback(state)
