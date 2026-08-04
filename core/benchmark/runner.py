"""Model-agnostic benchmark execution using registry metadata only."""

from __future__ import annotations

import gc
import math
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from core.benchmark.assets import (
    BenchmarkAssetManifest,
    ensure_benchmark_assets,
    load_asset_frames,
)
from core.geometry.sampling import stride_for_point_budget
from core.hardware.gpu_info import HardwareInfo, detect_hardware
from core.hardware.profiles import ComputeProfile, resolve_profile
from core.models.metadata import ModelDescriptor
from core.models.registry import (
    build_model,
    check_model_availability,
    get_model_descriptor,
    iter_model_descriptors,
    resolve_model_profile_metadata,
    resolve_model_runtime_config,
)
from core.models.runtime_config import ModelRuntimeContext
from core.runtime.messages import FrameRequest
from core.version import __version__


@dataclass(frozen=True, slots=True)
class BenchmarkOptions:
    profile: ComputeProfile = ComputeProfile.AUTO
    iterations: int | None = None
    sequence_frames: int | None = None
    point_budget: int = 100_000
    smoke_test: bool = False


def production_benchmark_descriptors() -> tuple[ModelDescriptor, ...]:
    return tuple(
        descriptor
        for descriptor in iter_model_descriptors(selectable_only=True, enabled_only=True)
        if descriptor.benchmark.included and not descriptor.development_only
    )


def benchmark_model(
    model_id: str,
    *,
    assets_root: str | Path,
    options: BenchmarkOptions | None = None,
    hardware: HardwareInfo | None = None,
) -> dict[str, object]:
    """Load, warm up, measure, and unload one registered adapter."""

    options = options or BenchmarkOptions()
    descriptor = get_model_descriptor(model_id)
    policy = descriptor.benchmark
    hardware = hardware or detect_hardware()
    availability = check_model_availability(model_id, hardware)
    base = {
        "schema_version": 1,
        "application_version": __version__,
        "model_id": descriptor.model_id,
        "display_name": descriptor.display_name,
        "classification": policy.classification,
        "workload": policy.workload,
        "registry_metadata": descriptor.as_metadata(),
    }
    if not policy.included:
        return {**base, "status": "excluded", "availability_reason": policy.notes}
    if not availability.available:
        return {
            **base,
            "status": "unavailable",
            "availability_reason": availability.reason,
            "availability": {
                "missing_dependencies": list(availability.missing_dependencies),
                "missing_assets": list(availability.missing_assets),
                "checkpoint_status": availability.checkpoint_status,
                "hardware_supported": availability.hardware_supported,
            },
        }

    manifest = ensure_benchmark_assets(assets_root)
    resolved = resolve_profile(options.profile, hardware)
    point_budget = max(1, int(options.point_budget))
    context = ModelRuntimeContext(
        hardware_info=hardware,
        resolved_profile=resolved,
        point_budget=point_budget,
    )
    config = resolve_model_runtime_config(model_id, context)
    profile_metadata = resolve_model_profile_metadata(model_id, resolved)
    device = str(config.get("device", resolved.device))
    precision = str(config.get("precision", resolved.precision.value))

    runtime = {
        "requested_profile": options.profile.value,
        "effective_profile": resolved.effective.value,
        "downgraded": resolved.downgraded,
        "profile_warning": resolved.warning or "",
        "device": device,
        "precision": precision.upper(),
        "variant": profile_metadata.variant_label,
        "detail": profile_metadata.detail_label,
        "point_budget": point_budget,
    }
    adapter = build_model(model_id)
    load_ms: float | None = None
    warmup_ms: float | None = None
    unload_ms: float | None = None
    cleanup: dict[str, Any] | None = None
    results: list[Any] = []
    end_to_end_ms: list[float] = []
    payload: dict[str, object]

    try:
        started = time.perf_counter()
        adapter.load(device=device, precision=precision, config=config)
        load_ms = (time.perf_counter() - started) * 1000.0

        started = time.perf_counter()
        adapter.warmup()
        warmup_ms = (time.perf_counter() - started) * 1000.0
        # Every adapter exposes the same neutral sequence-reset hook. Calling it
        # here prevents warm-up frames from affecting temporal benchmarks while
        # remaining a no-op for stateless models.
        adapter.reset_sequence()

        frames = _workload_frames(manifest, descriptor, options)
        _reset_peak_cuda(device)
        for frame_id, image in enumerate(frames):
            submitted = time.monotonic()
            request = FrameRequest(
                session_id=1,
                frame_id=frame_id,
                image=np.ascontiguousarray(image),
                point_budget=point_budget,
                submitted_at=submitted,
                source_timestamp=submitted,
            )
            started = time.perf_counter()
            result = adapter.infer(request)
            end_to_end_ms.append((time.perf_counter() - started) * 1000.0)
            results.append(result)

        if not results:
            raise RuntimeError("Benchmark workload produced no inference results.")
        payload = {
            **base,
            "status": "passed",
            "runtime": runtime,
            "timing": _timing_summary(
                results,
                end_to_end_ms=end_to_end_ms,
                load_ms=load_ms,
                warmup_ms=warmup_ms,
            ),
            "memory": _memory_summary(results, device=device),
            "output": _output_summary(results, point_budget=point_budget),
            "iterations": len(results),
            "asset_manifest": str(manifest.manifest),
        }
    except Exception as exc:  # noqa: BLE001 - benchmark must report and continue
        payload = {
            **base,
            "status": "failed",
            "runtime": runtime,
            "timing": {"load_ms": load_ms, "warmup_ms": warmup_ms},
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
    finally:
        started = time.perf_counter()
        try:
            cleanup = adapter.unload()
        except Exception as exc:  # noqa: BLE001
            cleanup = {"unload_error": f"{type(exc).__name__}: {exc}"}
        unload_ms = (time.perf_counter() - started) * 1000.0
        _cleanup_cuda(device)
        del adapter
        gc.collect()

    timing = payload.setdefault("timing", {})
    if isinstance(timing, dict):
        timing["unload_ms"] = unload_ms
    payload["cleanup"] = cleanup or {}
    return payload


def _workload_frames(
    manifest: BenchmarkAssetManifest,
    descriptor: ModelDescriptor,
    options: BenchmarkOptions,
) -> tuple[np.ndarray, ...]:
    policy = descriptor.benchmark
    available = load_asset_frames(manifest, policy.asset_key)
    if policy.workload == "sequence":
        frame_count = options.sequence_frames or policy.sequence_frames
        if options.smoke_test:
            frame_count = min(frame_count, 2)
        frame_count = max(1, int(frame_count))
        if len(available) == 1:
            available = tuple(available[0].copy() for _ in range(frame_count))
        repeats = math.ceil(frame_count / len(available))
        return tuple((available * repeats)[:frame_count])

    iterations = options.iterations or policy.default_iterations
    if options.smoke_test:
        iterations = 1
    iterations = max(1, int(iterations))
    # The common image workload alternates landscape and portrait assets. Very
    # slow quality models may declare one iteration in registry metadata.
    portrait = load_asset_frames(manifest, "portrait")[0]
    common_frames = (available[0], portrait)
    repeats = math.ceil(iterations / len(common_frames))
    return tuple(frame.copy() for frame in (common_frames * repeats)[:iterations])


def _timing_summary(
    results: Iterable[Any],
    *,
    end_to_end_ms: list[float],
    load_ms: float,
    warmup_ms: float,
) -> dict[str, object]:
    values = list(results)
    model = [float(item.latency_ms) for item in values]
    worker = [
        float(item.worker_total_ms)
        for item in values
        if item.worker_total_ms is not None
    ]
    preprocess = [
        float(item.preprocess_ms)
        for item in values
        if item.preprocess_ms is not None
    ]
    postprocess = [
        float(item.postprocess_ms)
        for item in values
        if item.postprocess_ms is not None
    ]
    e2e = _stats(end_to_end_ms)
    mean_seconds = float(e2e["mean"]) / 1000.0
    return {
        "load_ms": load_ms,
        "warmup_ms": warmup_ms,
        "model_latency_ms": _stats(model),
        "worker_total_ms": _stats(worker) if worker else None,
        "preprocess_ms": _stats(preprocess) if preprocess else None,
        "postprocess_ms": _stats(postprocess) if postprocess else None,
        "end_to_end_ms": e2e,
        "fps": None if mean_seconds <= 0 else 1.0 / mean_seconds,
    }


def _stats(values: Iterable[float]) -> dict[str, float]:
    data = [float(value) for value in values]
    if not data:
        raise ValueError("Cannot summarize an empty benchmark sample.")
    ordered = sorted(data)
    return {
        "count": float(len(data)),
        "mean": statistics.fmean(data),
        "median": statistics.median(data),
        "minimum": ordered[0],
        "maximum": ordered[-1],
        "stdev": statistics.pstdev(data) if len(data) > 1 else 0.0,
    }


def _output_summary(results: Iterable[Any], *, point_budget: int) -> dict[str, object]:
    values = list(results)
    if not values:
        raise RuntimeError("No model outputs are available for summary.")
    resolutions: list[list[int]] = []
    point_counts: list[int] = []
    point_kinds: set[str] = set()
    for result in values:
        if result.depth is None:
            raise RuntimeError("Adapter returned no depth output during benchmark.")
        depth = np.asarray(result.depth)
        if depth.ndim != 2:
            raise RuntimeError(f"Depth output must be HxW; got {depth.shape!r}.")
        height, width = depth.shape
        resolution = [int(width), int(height)]
        if resolution not in resolutions:
            resolutions.append(resolution)
        if result.points is not None:
            points = np.asarray(result.points)
            count = int(np.prod(points.shape[:2])) if points.ndim == 3 else int(points.shape[0])
            kind = "direct"
        else:
            stride = stride_for_point_budget(height, width, point_budget)
            sampled = depth[::stride, ::stride]
            mask = np.isfinite(sampled) & (sampled > 0)
            if result.valid_mask is not None:
                valid = np.asarray(result.valid_mask, dtype=bool)[::stride, ::stride]
                if valid.shape == mask.shape:
                    mask &= valid
            count = int(mask.sum())
            kind = "projectable"
        point_counts.append(min(count, point_budget))
        point_kinds.add(kind)

    latest = values[-1]
    return {
        "resolution": resolutions[-1],
        "resolutions": resolutions,
        "point_count": int(round(statistics.fmean(point_counts))),
        "point_count_minimum": min(point_counts),
        "point_count_maximum": max(point_counts),
        "point_count_kind": next(iter(point_kinds)) if len(point_kinds) == 1 else "mixed",
        "depth_scale": latest.depth_scale,
        "is_metric": bool(latest.is_metric),
        "intrinsics_source": latest.intrinsics_source,
        "confidence_kind": latest.confidence_kind,
        "has_points": latest.points is not None,
        "has_normals": latest.normals is not None,
        "has_confidence": latest.confidence is not None,
        "has_valid_mask": latest.valid_mask is not None,
    }


def _memory_summary(results: Iterable[Any], *, device: str) -> dict[str, object]:
    values = list(results)
    return {
        "allocated_gib": _maximum(item.vram_allocated_gib for item in values),
        "reserved_gib": _maximum(item.vram_reserved_gib for item in values),
        "peak_reserved_gib": max(
            value
            for value in (
                _maximum(item.vram_peak_reserved_gib for item in values),
                _torch_peak_reserved_gib(device),
            )
            if value is not None
        ) if any(
            value is not None
            for value in (
                _maximum(item.vram_peak_reserved_gib for item in values),
                _torch_peak_reserved_gib(device),
            )
        ) else None,
        "device_used_gib": _maximum(item.vram_device_used_gib for item in values),
        "total_gib": _maximum(item.vram_total_gib for item in values),
    }


def _maximum(values: Iterable[float | None]) -> float | None:
    valid = [float(value) for value in values if value is not None]
    return max(valid) if valid else None


def hardware_snapshot(hardware: HardwareInfo | None = None) -> dict[str, object]:
    hardware = hardware or detect_hardware()
    return {
        "backend": hardware.backend.value,
        "device": hardware.device,
        "name": hardware.name,
        "total_memory_gib": hardware.total_memory_gib,
        "free_memory_gib": hardware.free_memory_gib,
        "compute_capability": list(hardware.compute_capability) if hardware.compute_capability else None,
        "supports_fp16": hardware.supports_fp16,
        "supports_bf16": hardware.supports_bf16,
        "torch_version": hardware.torch_version,
        "torch_cuda_version": hardware.torch_cuda_version,
        "cuda_ready": hardware.cuda_ready,
        "warning": hardware.warning,
    }


def _reset_peak_cuda(device: str) -> None:
    if not str(device).startswith("cuda"):
        return
    try:
        import torch

        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    except Exception:
        return


def _torch_peak_reserved_gib(device: str) -> float | None:
    if not str(device).startswith("cuda"):
        return None
    try:
        import torch

        torch.cuda.synchronize(device)
        return float(torch.cuda.max_memory_reserved(device)) / 1024**3
    except Exception:
        return None


def _cleanup_cuda(device: str) -> None:
    if not str(device).startswith("cuda"):
        return
    try:
        import torch

        torch.cuda.synchronize(device)
        torch.cuda.empty_cache()
    except Exception:
        return
