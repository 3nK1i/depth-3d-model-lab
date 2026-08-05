"""Stable Video mode controller.

This controller does not own an inference worker. It orchestrates the existing
production runtime, mirrors rendered results into the Stable Video page, and
optionally records synchronized RGB/depth outputs through the atomic exporter.
"""

from __future__ import annotations

import time
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
from PyQt6.QtCore import QObject, pyqtSignal

from app.widgets.performance_snapshot import PerformanceSnapshot
from core.runtime.messages import InferenceResult
from core.stable_video.export import StableVideoExportSession


class StableVideoRunState(Enum):
    IDLE = "idle"
    STARTING = "starting"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPING = "stopping"
    COMPLETED = "completed"
    FAILED = "failed"


class StableVideoController(QObject):
    state_changed = pyqtSignal(object)
    log_received = pyqtSignal(str)
    result_ready = pyqtSignal(object, object, object)
    export_ready = pyqtSignal(str)
    statistics_changed = pyqtSignal(object)

    def __init__(self, project_root: Path, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._root = Path(project_root)
        self._state = StableVideoRunState.IDLE
        self._export: StableVideoExportSession | None = None
        self._output_dir: Path | None = None
        self._started_at: float | None = None
        self._latencies: list[float] = []
        self._frames = 0
        self._last_result_frame_id: int | None = None
        self._last_resolution: tuple[int, int] | None = None
        self._reset_count = 0
        self._last_semantics: dict[str, object] = {}
        self._model_id = ""
        self._model_name = ""
        self._temporal_mode = "generic_smoothing"
        self._smoothing_strength = 0.5
        self._export_enabled = True
        self._numeric_depth = False
        self._export_fps = 20.0

    @property
    def state(self) -> StableVideoRunState:
        return self._state

    @property
    def is_active(self) -> bool:
        return self._state in {
            StableVideoRunState.STARTING,
            StableVideoRunState.RUNNING,
            StableVideoRunState.PAUSED,
            StableVideoRunState.STOPPING,
        }

    @property
    def output_dir(self) -> Path | None:
        return self._output_dir

    def begin(
        self,
        *,
        output_dir: Path,
        model_id: str,
        model_name: str,
        temporal_mode: str,
        smoothing_strength: float,
        export_enabled: bool,
        export_numeric_depth: bool,
        export_fps: float,
    ) -> None:
        if self.is_active:
            raise RuntimeError("Stable Video is already running.")
        self._output_dir = Path(output_dir)
        self._model_id = str(model_id)
        self._model_name = str(model_name)
        self._temporal_mode = str(temporal_mode)
        self._smoothing_strength = float(smoothing_strength)
        self._export_enabled = bool(export_enabled)
        self._numeric_depth = bool(export_numeric_depth)
        self._export_fps = float(export_fps)
        self._started_at = time.monotonic()
        self._latencies.clear()
        self._frames = 0
        self._last_result_frame_id = None
        self._last_resolution = None
        self._reset_count = 0
        self._last_semantics = {}
        self._export = None
        if self._export_enabled:
            self._export = StableVideoExportSession(
                self._output_dir,
                fps=self._export_fps,
                model_id=self._model_id,
                model_name=self._model_name,
                temporal_mode=self._temporal_mode,
                smoothing_strength=self._smoothing_strength,
                export_numeric_depth=self._numeric_depth,
            )
        self._set_state(StableVideoRunState.STARTING)
        self.log_received.emit(
            f"Stable Video prepared: {self._model_name} · {self._temporal_mode}."
        )

    def mark_running(self) -> None:
        if self._state in {StableVideoRunState.STARTING, StableVideoRunState.PAUSED}:
            self._set_state(StableVideoRunState.RUNNING)

    def mark_paused(self) -> None:
        if self._state is StableVideoRunState.RUNNING:
            self._set_state(StableVideoRunState.PAUSED)

    def mark_reset(self, reason: str) -> None:
        self._reset_count += 1
        self._last_result_frame_id = None
        self.log_received.emit(f"Temporal sequence reset: {reason}.")
        self._emit_statistics(None)

    def consume_result(
        self,
        result: InferenceResult,
        rgb: np.ndarray | None,
        snapshot: PerformanceSnapshot,
    ) -> None:
        if self._state is not StableVideoRunState.RUNNING:
            return
        if result.depth is None or rgb is None:
            return
        if self._last_result_frame_id is not None and result.frame_id <= self._last_result_frame_id:
            self.mark_reset("frame counter restarted")
        resolution = (int(rgb.shape[1]), int(rgb.shape[0]))
        if self._last_resolution is not None and resolution != self._last_resolution:
            self.mark_reset(
                f"source resolution changed from {self._last_resolution[0]}×{self._last_resolution[1]} "
                f"to {resolution[0]}×{resolution[1]}"
            )
        self._last_resolution = resolution
        self._last_result_frame_id = result.frame_id
        self._last_semantics = {
            "depth_scale": result.depth_scale or ("metric" if result.is_metric else "relative"),
            "intrinsics_source": result.intrinsics_source or "unknown",
            "precision": result.precision,
        }
        self._frames += 1
        self._latencies.append(float(snapshot.latency_ms or result.latency_ms or 0.0))
        if len(self._latencies) > 300:
            del self._latencies[:-300]

        if self._export is not None:
            performance = {
                "frame_index": self._frames - 1,
                "source_frame_id": result.frame_id,
                "model_latency_ms": result.latency_ms,
                "end_to_end_latency_ms": snapshot.latency_ms,
                "inference_fps": snapshot.inference_fps,
                "source_fps": snapshot.camera_fps,
                "model_vram_gib": snapshot.vram_used_gb,
                "peak_vram_gib": snapshot.vram_peak_reserved_gb,
                "dropped_before_inference": snapshot.dropped_before_inference,
                "dropped_before_render": snapshot.dropped_before_render,
            }
            try:
                self._export.append(
                    rgb_bgr=rgb,
                    depth=result.depth,
                    valid_mask=result.valid_mask,
                    performance=performance,
                )
            except Exception as exc:  # noqa: BLE001
                self.log_received.emit(f"Stable Video export failed: {exc}")
                self._fail()
                return
        self.result_ready.emit(result, rgb, snapshot)
        self._emit_statistics(snapshot)

    def finish(self, *, completed: bool = True) -> None:
        if not self.is_active and self._state is not StableVideoRunState.FAILED:
            return
        self._set_state(StableVideoRunState.STOPPING)
        if self._export is not None:
            try:
                summary = self._export.finalize(
                    completed=completed,
                    extra_manifest={
                        **self._last_semantics,
                        "sequence_resets": self._reset_count,
                    },
                )
                if completed:
                    self._output_dir = summary.output_dir
                    self.export_ready.emit(str(summary.output_dir))
            except Exception as exc:  # noqa: BLE001
                self.log_received.emit(f"Stable Video finalization failed: {exc}")
                self._fail()
                return
            finally:
                self._export = None
        self._set_state(
            StableVideoRunState.COMPLETED if completed else StableVideoRunState.IDLE
        )

    def abort(self) -> None:
        if self._export is not None:
            self._export.abort()
            self._export = None
        self._set_state(StableVideoRunState.IDLE)

    def fail(self, message: str) -> None:
        self.log_received.emit(message)
        self._fail()

    def shutdown(self) -> None:
        if self.is_active:
            self.finish(completed=False)

    def _fail(self) -> None:
        if self._export is not None:
            self._export.abort()
            self._export = None
        self._set_state(StableVideoRunState.FAILED)

    def _set_state(self, state: StableVideoRunState) -> None:
        self._state = state
        self.state_changed.emit(state)

    def _emit_statistics(self, snapshot: PerformanceSnapshot | None) -> None:
        elapsed = 0.0 if self._started_at is None else max(0.0, time.monotonic() - self._started_at)
        avg_latency = float(np.mean(self._latencies)) if self._latencies else None
        peak_latency = float(np.max(self._latencies)) if self._latencies else None
        data: dict[str, Any] = {
            "frames": self._frames,
            "elapsed_seconds": elapsed,
            "average_latency_ms": avg_latency,
            "peak_latency_ms": peak_latency,
            "reset_count": self._reset_count,
            "processing_fps": None if elapsed <= 0 else self._frames / elapsed,
            "source_fps": None if snapshot is None else snapshot.camera_fps,
            "dropped_frames": 0 if snapshot is None else (
                snapshot.dropped_before_inference + snapshot.dropped_before_render
            ),
            "model_vram_gib": None if snapshot is None else snapshot.vram_used_gb,
        }
        self.statistics_changed.emit(data)
