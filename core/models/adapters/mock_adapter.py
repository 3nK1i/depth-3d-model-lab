"""Development-only mock model (spec §24).

Generates a synthetic depth gradient — never real AI output. This is
what Phase 3's entire safe start/pause/stop/restart lifecycle is
proven against before any real model (Phase 5+) exists, so that when a
real model is eventually loaded, the lifecycle underneath it is already
trustworthy.

Every result carries ``model_id=MOCK_MODEL_ID`` specifically so the UI
layer can detect it and render the required
"DEVELOPMENT MOCK — NOT AI OUTPUT" label — mock output must never be
presented as if it came from a real model.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from core.models.base import BaseGeometryModel
from core.models.catalog import MOCK_DISPLAY_LABEL, MOCK_MODEL_ID
from core.models.registry import get_model_descriptor
from core.models.result import GeometryResult
from core.runtime.messages import FrameRequest


class SimulatedOOMError(Exception):
    """Raised by the mock adapter when configured to simulate CUDA OOM."""


class MockGeometryModel(BaseGeometryModel):
    """Synthetic depth gradient generator for exercising the runtime
    lifecycle without any real model dependency.

    Behavior is tunable via ``load()``'s config dict:
        latency_ms: float           — simulated per-frame inference time
        simulate_oom: bool          — raise SimulatedOOMError on next infer
        simulate_hang_seconds: float — block for this long (tests the
                                       watchdog / restart-worker path)
        simulate_load_error: str     — fail during load with this message
        simulate_warmup_error: str   — fail during warm-up with this message
        simulate_unload_error: str   — fail during unload with this message
    """

    def __init__(self) -> None:
        self._loaded = False
        self._latency_ms = 20.0
        self._simulate_oom = False
        self._simulate_hang_seconds = 0.0
        self._simulate_load_error = ""
        self._simulate_warmup_error = ""
        self._simulate_unload_error = ""
        self._frame_counter = 0
        self._device = "cpu"
        self._precision = "fp32"

    def load(self, device: str, precision: str, config: dict[str, Any]) -> None:
        self._device = str(device or "cpu")
        self._precision = str(precision or "fp32").lower()
        self._latency_ms = float(config.get("latency_ms", 20.0))
        self._simulate_oom = bool(config.get("simulate_oom", False))
        self._simulate_hang_seconds = float(config.get("simulate_hang_seconds", 0.0))
        self._simulate_load_error = str(config.get("simulate_load_error", ""))
        self._simulate_warmup_error = str(config.get("simulate_warmup_error", ""))
        self._simulate_unload_error = str(config.get("simulate_unload_error", ""))
        if self._simulate_load_error:
            raise RuntimeError(self._simulate_load_error)
        # Room-scale range in meters, matching the units the rest of the
        # app assumes (depth colormap near/far, projection) — NOT a raw
        # unitless 0-1 gradient, which produced a point cloud smaller
        # than a 1-unit cube and made the camera auto-fit zoom in so
        # tight that the floor grid (sized for a normal scene) all but
        # vanished from view.
        self._depth_near_m = float(config.get("depth_near_m", 0.5))
        self._depth_far_m = float(config.get("depth_far_m", 4.0))
        # A real adapter would load weights onto `device` at `precision`
        # here; the mock has nothing to load.
        self._loaded = True

    def warmup(self) -> None:
        if not self._loaded:
            raise RuntimeError("MockGeometryModel.warmup() called before load()")
        if self._simulate_warmup_error:
            raise RuntimeError(self._simulate_warmup_error)
        time.sleep(0.05)

    def infer(self, frame: FrameRequest) -> GeometryResult:
        if not self._loaded:
            raise RuntimeError("MockGeometryModel.infer() called before load()")

        if self._simulate_hang_seconds > 0:
            time.sleep(self._simulate_hang_seconds)
        if self._simulate_oom:
            raise SimulatedOOMError("Simulated CUDA out-of-memory (mock)")

        worker_started = time.perf_counter()
        queue_wait_ms = None
        if frame.submitted_at is not None:
            queue_wait_ms = max(0.0, (time.monotonic() - frame.submitted_at) * 1000.0)

        model_started = time.perf_counter()
        if self._latency_ms > 0:
            time.sleep(self._latency_ms / 1000.0)
        elapsed_ms = (time.perf_counter() - model_started) * 1000.0

        post_started = time.perf_counter()
        depth = self._synthetic_depth(frame.image)
        postprocess_ms = (time.perf_counter() - post_started) * 1000.0
        worker_total_ms = (time.perf_counter() - worker_started) * 1000.0
        self._frame_counter += 1

        return GeometryResult(
            session_id=frame.session_id,
            frame_id=frame.frame_id,
            timestamp=time.time(),
            rgb=frame.image,
            depth=depth,
            latency_ms=elapsed_ms,
            preprocess_ms=0.0,
            postprocess_ms=postprocess_ms,
            worker_total_ms=worker_total_ms,
            queue_wait_ms=queue_wait_ms,
            is_metric=False,
            model_id=MOCK_MODEL_ID,
            model_name=MOCK_DISPLAY_LABEL,
            precision="FP32",
            depth_scale="synthetic",
            intrinsics_source="none",
            confidence_kind="none",
        )

    def unload(self) -> dict[str, Any]:
        released_frames = self._frame_counter
        unload_error = self._simulate_unload_error
        self._loaded = False
        self._frame_counter = 0
        self._simulate_unload_error = ""
        if unload_error:
            raise RuntimeError(unload_error)
        return {
            "released": True,
            "released_frames": released_frames,
            "device": "cpu",
        }

    def get_metadata(self) -> dict[str, Any]:
        descriptor = get_model_descriptor(MOCK_MODEL_ID)
        return descriptor.metadata_with_runtime(
            display_name="Development Mock",
            device=self._device,
            precision="FP32",
        )

    def _synthetic_depth(self, image: np.ndarray) -> np.ndarray:
        """A deterministic diagonal gradient, sized to the input frame
        and scaled into ``[depth_near_m, depth_far_m]`` — a real
        room-scale range in meters, clearly synthetic, never mistakable
        for a real depth estimate."""
        height, width = image.shape[:2]
        y = np.linspace(0.0, 1.0, height, dtype=np.float32).reshape(-1, 1)
        x = np.linspace(0.0, 1.0, width, dtype=np.float32).reshape(1, -1)
        gradient = (x + y) / 2.0
        return self._depth_near_m + gradient * (self._depth_far_m - self._depth_near_m)