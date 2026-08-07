#!/usr/bin/env python3
"""Dependency-light Phase 8.4C capture->SLAM runtime smoke using the mock backend."""

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
)
from tests.reconstruction_live_helpers import LIVE_IMAGE_SIZE, make_live_request


def main() -> int:
    observations = []
    runtime = LiveCaptureSlamRuntime(
        request=make_live_request(),
        backend_factory=lambda _spec: DeterministicMockLiveSlamBackend(),
        observation_callback=observations.append,
    )
    runtime.start()
    if not runtime.wait_for_state(LiveCaptureRuntimeState.RUNNING, timeout_s=2.0):
        raise RuntimeError(f"Runtime failed to start: {runtime.stats}")

    base = monotonic()
    image = np.zeros((LIVE_IMAGE_SIZE.height, LIVE_IMAGE_SIZE.width, 3), dtype=np.uint8)
    for index in range(200):
        runtime.offer_source_frame(SourceFrame(image.copy(), index, base + index * 0.001))

    deadline = monotonic() + 2.0
    while not observations and monotonic() < deadline:
        sleep(0.01)
    if not observations:
        raise RuntimeError("No live SLAM observation was produced.")
    if not runtime.stop(timeout_s=2.0):
        raise RuntimeError("Live capture runtime did not stop cleanly.")

    stats = runtime.stats
    if stats.offered_frames != 200 or stats.replaced_frames <= 0:
        raise RuntimeError(f"Latest-frame backpressure was not exercised: {stats}")
    latest = observations[-1]
    print("Phase 8.4C live capture runtime smoke: PASS")
    print(f"offered={stats.offered_frames} processed={stats.processed_frames} replaced={stats.replaced_frames}")
    print(f"observations={stats.observations} tracking_state={latest.tracking_state.value}")
    print("pose_convention=T_reconstruction_from_camera")
    print("gauge=arbitrary unit_to_metre=None")
    print("capture_thread_blocking=False latest_frame_only=True")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
