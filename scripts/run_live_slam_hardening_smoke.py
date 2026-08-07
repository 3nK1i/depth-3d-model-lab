#!/usr/bin/env python3
"""Dependency-light Phase 8.4D worker recovery/telemetry smoke."""

from __future__ import annotations

from pathlib import Path
import sys
from time import monotonic, sleep

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.capture.frame import SourceFrame
from core.reconstruction import (
    DeterministicMockLiveSlamBackend,
    LiveCaptureRuntimeState,
    LiveCaptureSlamRuntime,
    LiveSlamRecoveryPolicy,
    LiveSlamWorkerExecutionError,
)
from tests.reconstruction_live_helpers import LIVE_IMAGE_SIZE, make_live_request


def main() -> int:
    created = 0
    observations = []
    states = []

    class CrashFirstBackend(DeterministicMockLiveSlamBackend):
        def __init__(self, should_crash: bool) -> None:
            super().__init__(
                initialization_frames=1,
                lost_sequences=(6,),
                degraded_sequences=(7,),
            )
            self._should_crash = should_crash

        def track_live_frame(self, frame, context):
            if self._should_crash:
                self._should_crash = False
                raise LiveSlamWorkerExecutionError("phase8.4d synthetic worker crash")
            return super().track_live_frame(frame, context)

    def factory(_spec):
        nonlocal created
        created += 1
        return CrashFirstBackend(should_crash=created == 1)

    runtime = LiveCaptureSlamRuntime(
        request=make_live_request(),
        backend_factory=factory,
        recovery_policy=LiveSlamRecoveryPolicy(
            max_worker_restarts=2,
            restart_backoff_s=0.01,
            max_consecutive_stale_frames=3,
        ),
        observation_callback=observations.append,
        state_callback=states.append,
    )
    runtime.start()
    if not runtime.wait_for_state(LiveCaptureRuntimeState.RUNNING, timeout_s=2.0):
        raise RuntimeError(f"runtime failed to start: {runtime.stats}")

    base = monotonic()
    for index in range(20):
        runtime.offer_source_frame(
            SourceFrame(
                np.full(
                    (LIVE_IMAGE_SIZE.height, LIVE_IMAGE_SIZE.width, 3),
                    index,
                    dtype=np.uint8,
                ),
                frame_id=index,
                timestamp=base + index * 0.01,
            )
        )
        sleep(0.025)

    deadline = monotonic() + 2.0
    while runtime.stats.processed_frames < 15 and monotonic() < deadline:
        sleep(0.01)
    if not runtime.stop(timeout_s=2.0):
        raise RuntimeError("runtime did not stop cleanly")

    stats = runtime.stats
    passed = bool(
        stats.worker_restarts == 1
        and stats.worker_restart_failures == 0
        and LiveCaptureRuntimeState.RECOVERING in states
        and stats.tracking_loss_events >= 1
        and stats.tracking_recoveries >= 1
        and stats.cleanup_complete
        and stats.last_error == ""
        and observations
    )
    if not passed:
        raise RuntimeError(f"Phase 8.4D hardening smoke failed: {stats}")

    print("Phase 8.4D live SLAM hardening smoke: PASS")
    print(
        f"offered={stats.offered_frames} processed={stats.processed_frames} "
        f"replaced={stats.replaced_frames} observations={stats.observations}"
    )
    print(
        f"worker_restarts={stats.worker_restarts} "
        f"worker_restart_failures={stats.worker_restart_failures}"
    )
    print(
        f"tracking_loss_events={stats.tracking_loss_events} "
        f"tracking_recoveries={stats.tracking_recoveries}"
    )
    print(
        f"max_processing_latency_ns={stats.max_processing_latency_ns} "
        f"mean_processing_latency_ns={stats.mean_processing_latency_ns:.1f}"
    )
    print("cleanup_complete=True")
    print("pose_convention=T_reconstruction_from_camera")
    print("gauge=arbitrary unit_to_metre=None")
    print("loop_closure_enabled=False global_optimization_feature_enabled=False")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
