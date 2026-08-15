"""Phase 9.5G interactive NeRF performance/resource lifecycle contracts.

The production process never imports Nerfstudio, torch, or tiny-cuda-nn here.
This module models bounded request coalescing plus worker-reported CUDA telemetry
and lifecycle state for the already-qualified external interactive renderer.
Canonical geometry remains the sole picking/measurement authority.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import isfinite
from typing import Any, Mapping

from core.hardware.gpu_info import HardwareInfo, detect_hardware

MIB = 1024.0 * 1024.0


class NerfViewerResidencyState(str, Enum):
    UNLOADED = "unloaded"
    STARTING = "starting"
    RESIDENT = "resident"
    STOPPING = "stopping"
    RELEASED = "released"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class NerfInteractiveResourcePolicy:
    device_label: str
    resource_tier: str
    max_inflight_renders: int = 1
    max_pending_renders: int = 1
    max_retained_frames: int = 2
    shutdown_timeout_ms: int = 3000
    external_worker_required: bool = True
    canonical_geometry_authoritative: bool = True
    certified_metrology: bool = False

    def __post_init__(self) -> None:
        if int(self.max_inflight_renders) != 1:
            raise ValueError("Interactive NeRF supports exactly one in-flight render.")
        if int(self.max_pending_renders) != 1:
            raise ValueError("Interactive NeRF supports exactly one coalesced pending render.")
        if int(self.max_retained_frames) < 1 or int(self.max_retained_frames) > 4:
            raise ValueError("Interactive NeRF retained-frame bound must be between 1 and 4.")
        if int(self.shutdown_timeout_ms) < 500:
            raise ValueError("Interactive NeRF shutdown timeout is too small.")
        if not self.external_worker_required or not self.canonical_geometry_authoritative or self.certified_metrology:
            raise ValueError("Interactive NeRF resource policy cannot weaken isolation/authority rules.")
        object.__setattr__(self, "device_label", str(self.device_label).strip() or "Unknown device")
        object.__setattr__(self, "resource_tier", str(self.resource_tier).strip() or "conservative")
        object.__setattr__(self, "max_inflight_renders", 1)
        object.__setattr__(self, "max_pending_renders", 1)
        object.__setattr__(self, "external_worker_required", True)
        object.__setattr__(self, "canonical_geometry_authoritative", True)
        object.__setattr__(self, "certified_metrology", False)

    @property
    def queue_text(self) -> str:
        return "1 render in flight + 1 latest-view pending • older requests coalesced"


def build_nerf_interactive_resource_policy(
    hardware: HardwareInfo | None = None,
) -> NerfInteractiveResourcePolicy:
    hardware = hardware or detect_hardware()
    if not isinstance(hardware, HardwareInfo):
        raise TypeError("hardware must be HardwareInfo.")
    total_gib = hardware.total_memory_gib
    if not hardware.has_cuda_device or total_gib is None:
        tier = "cuda-required-or-unknown"
    elif total_gib <= 6.5:
        tier = "conservative-6gb"
    elif total_gib <= 16.0:
        tier = "balanced"
    else:
        tier = "high-memory"
    return NerfInteractiveResourcePolicy(
        device_label=hardware.short_label,
        resource_tier=tier,
    )


@dataclass(frozen=True, slots=True)
class LatestRenderQueueSnapshot:
    submitted_count: int
    dispatched_count: int
    coalesced_count: int
    completed_count: int
    render_in_flight: bool
    pending_request_id: int | None

    def __post_init__(self) -> None:
        for name in ("submitted_count", "dispatched_count", "coalesced_count", "completed_count"):
            if int(getattr(self, name)) < 0:
                raise ValueError(f"{name} must be non-negative.")
        if self.dispatched_count > self.submitted_count:
            raise ValueError("dispatched_count cannot exceed submitted_count.")
        if self.completed_count > self.dispatched_count:
            raise ValueError("completed_count cannot exceed dispatched_count.")

    @property
    def bounded(self) -> bool:
        return True


class LatestRenderRequestQueue:
    """One-in-flight/one-pending latest-view queue for expensive NeRF renders."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._submitted = 0
        self._dispatched = 0
        self._coalesced = 0
        self._completed = 0
        self._in_flight = False
        self._pending: dict[str, Any] | None = None

    def submit(self, payload: Mapping[str, Any]) -> tuple[bool, dict[str, Any] | None]:
        item = dict(payload)
        if str(item.get("type", "")) != "render":
            raise ValueError("LatestRenderRequestQueue accepts render payloads only.")
        request_id = int(item.get("request_id", -1))
        if request_id < 0:
            raise ValueError("Render request_id must be non-negative.")
        self._submitted += 1
        if self._in_flight:
            self._pending = item
            self._coalesced += 1
            return False, None
        self._in_flight = True
        self._dispatched += 1
        return True, item

    def complete(self, request_id: int) -> dict[str, Any] | None:
        if self._in_flight:
            self._completed += 1
        self._in_flight = False
        pending = self._pending
        self._pending = None
        if pending is not None:
            self._in_flight = True
            self._dispatched += 1
        return pending

    def cancel_pending(self) -> None:
        self._pending = None
        self._in_flight = False

    @property
    def snapshot(self) -> LatestRenderQueueSnapshot:
        pending_id = None if self._pending is None else int(self._pending.get("request_id", -1))
        return LatestRenderQueueSnapshot(
            submitted_count=self._submitted,
            dispatched_count=self._dispatched,
            coalesced_count=self._coalesced,
            completed_count=self._completed,
            render_in_flight=self._in_flight,
            pending_request_id=pending_id,
        )


@dataclass(frozen=True, slots=True)
class NerfViewerRuntimeHealth:
    state: NerfViewerResidencyState
    device_label: str
    resource_tier: str
    worker_pid: int
    total_vram_mib: float
    cuda_allocated_mib: float
    cuda_reserved_mib: float
    cuda_peak_allocated_mib: float
    last_render_ms: float
    worker_start_count: int
    worker_release_count: int
    queue: LatestRenderQueueSnapshot
    retained_frame_count: int
    canonical_geometry_authoritative: bool
    nerf_canonical_geometry_authoritative: bool = False
    nerf_metric_scale_authoritative: bool = False
    nerf_measurement_authoritative: bool = False
    nerf_qc_authoritative: bool = False
    certified_metrology: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.state, NerfViewerResidencyState):
            raise TypeError("state must be NerfViewerResidencyState.")
        for name in ("worker_pid", "worker_start_count", "worker_release_count", "retained_frame_count"):
            if int(getattr(self, name)) < 0:
                raise ValueError(f"{name} must be non-negative.")
        for name in (
            "total_vram_mib",
            "cuda_allocated_mib",
            "cuda_reserved_mib",
            "cuda_peak_allocated_mib",
            "last_render_ms",
        ):
            value = float(getattr(self, name))
            if not isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative.")
        if int(self.retained_frame_count) > 2:
            raise ValueError("Interactive NeRF retained frame count exceeded the qualified bound.")
        if not self.canonical_geometry_authoritative:
            raise ValueError("Interactive NeRF health requires canonical geometry authority.")
        if any(
            bool(value)
            for value in (
                self.nerf_canonical_geometry_authoritative,
                self.nerf_metric_scale_authoritative,
                self.nerf_measurement_authoritative,
                self.nerf_qc_authoritative,
                self.certified_metrology,
            )
        ):
            raise ValueError("Interactive NeRF runtime health cannot claim geometry/metrology authority.")

    @property
    def resident(self) -> bool:
        return self.state is NerfViewerResidencyState.RESIDENT

    @property
    def gpu_vram_measured(self) -> bool:
        return self.total_vram_mib > 0.0 and self.cuda_reserved_mib > 0.0

    @property
    def process_unreserved_capacity_mib(self) -> float | None:
        """Device total minus this worker's torch-reserved bytes.

        This is *not* global free VRAM because other processes may also own GPU
        memory.  Real target qualification separately records nvidia-smi process
        memory and release-by-PID evidence.
        """

        if self.total_vram_mib <= 0.0:
            return None
        return max(0.0, self.total_vram_mib - self.cuda_reserved_mib)

    @property
    def resources_text(self) -> str:
        if self.state in {NerfViewerResidencyState.UNLOADED, NerfViewerResidencyState.RELEASED}:
            return f"External worker released • {self.device_label}"
        if self.state is NerfViewerResidencyState.STARTING:
            return f"Loading isolated renderer • {self.device_label}"
        if self.state is NerfViewerResidencyState.STOPPING:
            return "Stopping isolated renderer • process-group cleanup pending"
        if self.state is NerfViewerResidencyState.FAILED:
            return "Interactive renderer failed • canonical view retained"
        if self.gpu_vram_measured:
            return (
                f"Worker CUDA reserved {self.cuda_reserved_mib:.0f} MiB • "
                f"allocated {self.cuda_allocated_mib:.0f} MiB • device total {self.total_vram_mib:.0f} MiB"
            )
        return "External renderer resident • CUDA telemetry pending"

    @property
    def performance_text(self) -> str:
        q = self.queue
        if self.last_render_ms <= 0.0:
            return f"No render measured • {q.dispatched_count} dispatched • {q.coalesced_count} coalesced"
        return (
            f"Last {self.last_render_ms:.0f} ms • {q.completed_count} completed • "
            f"{q.coalesced_count} coalesced"
        )

    @property
    def lifecycle_text(self) -> str:
        q = self.queue
        if self.state is NerfViewerResidencyState.RELEASED:
            return (
                f"Released on hide/switch • {self.worker_release_count} releases • reload on demand"
            )
        if self.resident:
            pending = "1 pending" if q.pending_request_id is not None else "no pending render"
            return (
                f"Resident isolated worker • {self.worker_start_count} starts • {pending} • "
                f"{self.retained_frame_count}/2 frames retained"
            )
        return f"{self.state.value.title()} • {self.worker_start_count} starts • {self.worker_release_count} releases"

    @property
    def authority_text(self) -> str:
        return "Appearance only • canonical picking/measurement unchanged • metrology not certified"

    def as_dict(self) -> dict[str, object]:
        return {
            "state": self.state.value,
            "resident": self.resident,
            "device_label": self.device_label,
            "resource_tier": self.resource_tier,
            "worker_pid": int(self.worker_pid),
            "total_vram_mib": float(self.total_vram_mib),
            "cuda_allocated_mib": float(self.cuda_allocated_mib),
            "cuda_reserved_mib": float(self.cuda_reserved_mib),
            "cuda_peak_allocated_mib": float(self.cuda_peak_allocated_mib),
            "gpu_vram_measured": self.gpu_vram_measured,
            "process_unreserved_capacity_mib": self.process_unreserved_capacity_mib,
            "last_render_ms": float(self.last_render_ms),
            "worker_start_count": int(self.worker_start_count),
            "worker_release_count": int(self.worker_release_count),
            "submitted_render_count": int(self.queue.submitted_count),
            "dispatched_render_count": int(self.queue.dispatched_count),
            "coalesced_render_count": int(self.queue.coalesced_count),
            "completed_render_count": int(self.queue.completed_count),
            "render_in_flight": bool(self.queue.render_in_flight),
            "pending_request_id": self.queue.pending_request_id,
            "retained_frame_count": int(self.retained_frame_count),
            "canonical_geometry_authoritative": True,
            "nerf_canonical_geometry_authoritative": False,
            "nerf_metric_scale_authoritative": False,
            "nerf_measurement_authoritative": False,
            "nerf_qc_authoritative": False,
            "certified_metrology": False,
            "timings_are_observation_only": True,
        }


def build_nerf_viewer_runtime_health(
    *,
    policy: NerfInteractiveResourcePolicy,
    worker_state: str,
    worker_pid: int,
    telemetry: Mapping[str, Any] | None,
    last_render_ms: float,
    worker_start_count: int,
    worker_release_count: int,
    queue: LatestRenderQueueSnapshot,
    canonical_geometry_authoritative: bool,
) -> NerfViewerRuntimeHealth:
    state_map = {
        "starting": NerfViewerResidencyState.STARTING,
        "ready": NerfViewerResidencyState.RESIDENT,
        "rendering": NerfViewerResidencyState.RESIDENT,
        "stopping": NerfViewerResidencyState.STOPPING,
        "stopped": NerfViewerResidencyState.RELEASED if worker_release_count else NerfViewerResidencyState.UNLOADED,
        "failed": NerfViewerResidencyState.FAILED,
    }
    data = dict(telemetry or {})
    return NerfViewerRuntimeHealth(
        state=state_map.get(str(worker_state).strip().lower(), NerfViewerResidencyState.UNLOADED),
        device_label=policy.device_label,
        resource_tier=policy.resource_tier,
        worker_pid=max(0, int(worker_pid)),
        total_vram_mib=max(0.0, float(data.get("total_vram_mib", 0.0) or 0.0)),
        cuda_allocated_mib=max(0.0, float(data.get("cuda_allocated_mib", 0.0) or 0.0)),
        cuda_reserved_mib=max(0.0, float(data.get("cuda_reserved_mib", 0.0) or 0.0)),
        cuda_peak_allocated_mib=max(0.0, float(data.get("cuda_peak_allocated_mib", 0.0) or 0.0)),
        last_render_ms=max(0.0, float(last_render_ms)),
        worker_start_count=max(0, int(worker_start_count)),
        worker_release_count=max(0, int(worker_release_count)),
        queue=queue,
        retained_frame_count=max(0, int(data.get("retained_frame_count", 0) or 0)),
        canonical_geometry_authoritative=bool(canonical_geometry_authoritative),
    )


__all__ = [
    "LatestRenderQueueSnapshot",
    "LatestRenderRequestQueue",
    "NerfInteractiveResourcePolicy",
    "NerfViewerResidencyState",
    "NerfViewerRuntimeHealth",
    "build_nerf_interactive_resource_policy",
    "build_nerf_viewer_runtime_health",
]
