"""Structured data and truthful formatting for the performance footer."""

from __future__ import annotations

from dataclasses import dataclass, replace

PLACEHOLDER = "—"

STATUS_COLOR_GREEN = "green"
STATUS_COLOR_AMBER = "amber"
STATUS_COLOR_RED = "red"
STATUS_COLOR_NEUTRAL = "neutral"

_STATUS_COLORS: dict[str, str] = {
    "IDLE": STATUS_COLOR_NEUTRAL,
    "CONNECTING": STATUS_COLOR_NEUTRAL,
    "READY": STATUS_COLOR_NEUTRAL,
    "SOURCE READY": STATUS_COLOR_NEUTRAL,
    "MODEL READY": STATUS_COLOR_NEUTRAL,
    "RESULT READY": STATUS_COLOR_GREEN,
    "RUNNING": STATUS_COLOR_GREEN,
    "PAUSED": STATUS_COLOR_AMBER,
    "STOPPING": STATUS_COLOR_NEUTRAL,
    "ERROR": STATUS_COLOR_RED,
}


def status_color(status_text: str) -> str:
    return _STATUS_COLORS.get(status_text.upper(), STATUS_COLOR_NEUTRAL)


def format_fps(value: float | None) -> str:
    return PLACEHOLDER if value is None else f"{value:.1f}"


def format_latency(value_ms: float | None) -> str:
    return PLACEHOLDER if value_ms is None else f"{value_ms:.1f} ms"


def format_points(count: int | None) -> str:
    return PLACEHOLDER if count is None else f"{count:,}"


def format_resolution(size: tuple[int, int] | None) -> str:
    return PLACEHOLDER if size is None else f"{size[0]}\u00d7{size[1]}"


def format_vram(used_gb: float | None, total_gb: float | None) -> str:
    if used_gb is None or total_gb is None:
        return PLACEHOLDER
    return f"{used_gb:.1f} / {total_gb:.0f} GB"


def vram_ratio(used_gb: float | None, total_gb: float | None) -> float:
    if used_gb is None or total_gb is None or total_gb <= 0:
        return 0.0
    return max(0.0, min(1.0, used_gb / total_gb))


def format_uptime(seconds: int | None) -> str:
    if seconds is None:
        return PLACEHOLDER
    hours, remainder = divmod(max(0, seconds), 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def format_or_dash(value: str | None) -> str:
    return value if value else PLACEHOLDER


def _ms_line(label: str, value: float | None) -> str | None:
    return None if value is None else f"{label}: {value:.1f} ms"


def _gib_line(label: str, value: float | None) -> str | None:
    return None if value is None else f"{label}: {value:.2f} GiB"


def _joined(lines: list[str | None]) -> str:
    return "\n".join(line for line in lines if line)


@dataclass(frozen=True, slots=True)
class PerformanceSnapshot:
    """Every displayed value plus its diagnostic breakdown for one session."""

    status: str
    camera_fps: float | None = None
    inference_fps: float | None = None
    # Visible LATENCY is end-to-end submit -> completed 3D update.
    latency_ms: float | None = None
    render_fps: float | None = None
    point_count: int | None = None
    resolution: tuple[int, int] | None = None
    # Visible VRAM is PyTorch reserved memory, the amount held by its allocator.
    vram_used_gb: float | None = None
    vram_total_gb: float | None = None
    model_name: str | None = None
    precision: str | None = None
    uptime_seconds: int | None = None

    # Timing breakdown.
    model_latency_ms: float | None = None
    preprocess_ms: float | None = None
    postprocess_ms: float | None = None
    worker_total_ms: float | None = None
    queue_wait_ms: float | None = None
    geometry_ms: float | None = None
    geometry_cpu_ms: float | None = None
    depth_view_ms: float | None = None
    canvas_update_ms: float | None = None
    source_to_render_ms: float | None = None

    # CUDA allocator/device details.
    vram_allocated_gb: float | None = None
    vram_reserved_gb: float | None = None
    vram_peak_reserved_gb: float | None = None
    vram_device_used_gb: float | None = None

    # Model lifecycle details.
    model_load_time_ms: float | None = None
    warmup_time_ms: float | None = None
    total_load_time_ms: float | None = None
    model_device: str | None = None
    checkpoint: str | None = None
    num_tokens: int | None = None
    resolution_level: int | None = None

    # Latest-frame/backpressure diagnostics.
    source_frames_seen: int = 0
    frames_submitted: int = 0
    results_received: int = 0
    results_rendered: int = 0
    dropped_before_inference: int = 0
    dropped_before_render: int = 0
    submit_rejected: int = 0
    inference_errors: int = 0
    point_stride: int | None = None
    point_budget: int | None = None

    @classmethod
    def empty(cls) -> "PerformanceSnapshot":
        return cls(status="IDLE")

    @classmethod
    def preview(cls) -> "PerformanceSnapshot":
        return cls(
            status="RUNNING",
            camera_fps=30.0,
            inference_fps=21.4,
            latency_ms=46.8,
            model_latency_ms=31.2,
            render_fps=60.0,
            point_count=842_713,
            resolution=(768, 768),
            vram_used_gb=23.6,
            vram_reserved_gb=23.6,
            vram_total_gb=32.0,
            model_name="MoGe-2 ViT-L",
            precision="FP16",
            uptime_seconds=12 * 60 + 43,
        )

    def with_status(self, status: str) -> "PerformanceSnapshot":
        return replace(self, status=status)

    def with_uptime(self, seconds: int | None) -> "PerformanceSnapshot":
        return replace(self, uptime_seconds=seconds)

    def with_runtime_counters(
        self,
        *,
        uptime_seconds: int | None,
        source_frames_seen: int,
        frames_submitted: int,
        results_received: int,
        results_rendered: int,
        dropped_before_inference: int,
        dropped_before_render: int,
        submit_rejected: int,
        inference_errors: int,
    ) -> "PerformanceSnapshot":
        return replace(
            self,
            uptime_seconds=uptime_seconds,
            source_frames_seen=source_frames_seen,
            frames_submitted=frames_submitted,
            results_received=results_received,
            results_rendered=results_rendered,
            dropped_before_inference=dropped_before_inference,
            dropped_before_render=dropped_before_render,
            submit_rejected=submit_rejected,
            inference_errors=inference_errors,
        )

    def status_tooltip(self) -> str:
        dropped = self.dropped_before_inference + self.dropped_before_render
        return _joined(
            [
                "Latest-frame pipeline diagnostics",
                f"Source frames seen: {self.source_frames_seen:,}",
                f"Submitted to worker: {self.frames_submitted:,}",
                f"Results received: {self.results_received:,}",
                f"3D updates rendered: {self.results_rendered:,}",
                f"Dropped/replaced total: {dropped:,}",
                f"Before inference: {self.dropped_before_inference:,}",
                f"Before render: {self.dropped_before_render:,}",
                f"Submit rejections: {self.submit_rejected:,}",
                f"Inference / geometry errors: {self.inference_errors:,}",
            ]
        )

    def camera_fps_tooltip(self) -> str:
        return (
            "Decoded source-frame rate measured before AI inference.\n"
            "Unavailable for a static image source."
        )

    def inference_fps_tooltip(self) -> str:
        return (
            "Completed worker results per second over a sliding window.\n"
            f"Submitted: {self.frames_submitted:,}  Received: {self.results_received:,}"
        )

    def latency_tooltip(self) -> str:
        return _joined(
            [
                "Visible value: end-to-end frame submission to completed 3D update.",
                _ms_line("End-to-end", self.latency_ms),
                _ms_line("Model forward pass", self.model_latency_ms),
                _ms_line("Worker preprocessing", self.preprocess_ms),
                _ms_line("Worker postprocessing / GPU-to-CPU", self.postprocess_ms),
                _ms_line("Worker total", self.worker_total_ms),
                _ms_line("Queue / IPC wait estimate", self.queue_wait_ms),
                _ms_line("UI depth + geometry + canvas", self.geometry_ms),
                _ms_line("  Depth-map update", self.depth_view_ms),
                _ms_line("  Point filtering / sampling", self.geometry_cpu_ms),
                _ms_line("  VisPy buffer update", self.canvas_update_ms),
                _ms_line("Source capture to render", self.source_to_render_ms),
            ]
        )

    def render_fps_tooltip(self) -> str:
        return (
            "Completed point-cloud buffer updates per second.\n"
            "This is the 3D update rate, not the monitor refresh rate.\n"
            f"Rendered: {self.results_rendered:,}  Replaced before render: "
            f"{self.dropped_before_render:,}"
        )

    def points_tooltip(self) -> str:
        parts = ["Points sent to the current VisPy point-cloud visual."]
        if self.point_stride is not None:
            parts.append(f"Worker sampling stride: {self.point_stride}")
        if self.point_budget is not None:
            parts.append(f"Configured point budget: {self.point_budget:,}")
        return "\n".join(parts)

    def vram_tooltip(self) -> str:
        return _joined(
            [
                "Visible value: PyTorch CUDA reserved memory / device total.",
                _gib_line("Allocated tensors", self.vram_allocated_gb),
                _gib_line("Reserved by PyTorch allocator", self.vram_reserved_gb),
                _gib_line("Peak reserved for this frame", self.vram_peak_reserved_gb),
                _gib_line("Device-wide used memory", self.vram_device_used_gb),
                _gib_line("Device total", self.vram_total_gb),
            ]
        )

    def model_tooltip(self) -> str:
        return _joined(
            [
                self.model_name or "Model details",
                f"Device: {self.model_device}" if self.model_device else None,
                f"Precision: {self.precision}" if self.precision else None,
                f"Checkpoint: {self.checkpoint}" if self.checkpoint else None,
                f"Tokens: {self.num_tokens}" if self.num_tokens is not None else None,
                (
                    f"Resolution level: {self.resolution_level}"
                    if self.resolution_level is not None
                    else None
                ),
                _ms_line("Model load", self.model_load_time_ms),
                _ms_line("Warm-up", self.warmup_time_ms),
                _ms_line("Total load + warm-up", self.total_load_time_ms),
            ]
        )
