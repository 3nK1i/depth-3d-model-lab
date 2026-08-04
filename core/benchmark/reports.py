"""Atomic JSON/CSV/Markdown report writers for Phase 6 benchmarking."""

from __future__ import annotations

import csv
import json
import os
import tempfile
from pathlib import Path
from typing import Iterable

from core.models.metadata import ModelDescriptor


def write_json(path: str | os.PathLike[str], payload: dict[str, object]) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
        delete=False,
    )
    temporary = Path(handle.name)
    try:
        with handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return destination


def write_benchmark_csv(
    path: str | os.PathLike[str],
    results: Iterable[dict[str, object]],
) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fields = (
        "model_id",
        "display_name",
        "status",
        "classification",
        "workload",
        "profile",
        "device",
        "precision",
        "load_ms",
        "warmup_ms",
        "mean_model_latency_ms",
        "mean_end_to_end_ms",
        "fps",
        "peak_reserved_vram_gib",
        "device_used_vram_gib",
        "output_width",
        "output_height",
        "point_count",
        "depth_scale",
        "is_metric",
        "intrinsics_source",
        "confidence_kind",
        "error",
    )
    rows = [_flatten_result(item) for item in results]
    handle = tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        newline="",
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
        delete=False,
    )
    temporary = Path(handle.name)
    try:
        with handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for row in rows:
                writer.writerow({field: row.get(field, "") for field in fields})
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return destination


def write_capability_matrix(
    path: str | os.PathLike[str],
    descriptors: Iterable[ModelDescriptor],
) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Phase 6 Model Capability Matrix",
        "",
        "| Model | Scale | Intrinsics | Confidence | Points | Normals | Stateful | Live | Benchmark class | License status |",
        "|---|---|---|---|---:|---:|---:|---:|---|---|",
    ]
    for descriptor in descriptors:
        caps = descriptor.capabilities
        lines.append(
            "| {name} | {scale} | {intrinsics} | {confidence} | {points} | {normals} | "
            "{stateful} | {live} | {classification} | {license_status} |".format(
                name=descriptor.display_name,
                scale=caps.depth_scale.value,
                intrinsics=caps.intrinsics_source.value,
                confidence=caps.confidence_kind.value,
                points=_yes(caps.outputs.points),
                normals=_yes(caps.outputs.normals),
                stateful=_yes(caps.stateful_sequence),
                live=_yes(caps.supports_live),
                classification=descriptor.benchmark.classification,
                license_status=descriptor.license.status,
            )
        )
    lines.extend(
        [
            "",
            "> Metric output does not imply calibrated measurement. Results using estimated intrinsics remain **Estimated — Not Calibrated**.",
            "",
        ]
    )
    return _atomic_write_text(destination, "\n".join(lines))


def write_benchmark_summary(
    path: str | os.PathLike[str],
    payload: dict[str, object],
) -> Path:
    destination = Path(path)
    results = list(payload.get("results", []))
    passed = [item for item in results if item.get("status") == "passed"]
    failed = [item for item in results if item.get("status") == "failed"]
    unavailable = [item for item in results if item.get("status") == "unavailable"]
    cancelled = [item for item in results if item.get("status") == "cancelled"]
    lines = [
        "# Phase 6 Cross-Model Benchmark Summary",
        "",
        f"- Application: `{payload.get('application', {}).get('version', '')}`",
        f"- Hardware: `{payload.get('hardware', {}).get('name', 'unknown')}`",
        f"- Passed: **{len(passed)}**",
        f"- Failed: **{len(failed)}**",
        f"- Unavailable: **{len(unavailable)}**",
        f"- Cancelled: **{len(cancelled)}**",
        "",
        "| Model | Status | Profile | Mean model latency | End-to-end | FPS | Peak model VRAM | Scale |",
        "|---|---|---|---:|---:|---:|---:|---|",
    ]
    for item in results:
        timing = item.get("timing", {}) or {}
        memory = item.get("memory", {}) or {}
        output = item.get("output", {}) or {}
        lines.append(
            "| {name} | {status} | {profile} | {latency} | {e2e} | {fps} | {vram} | {scale} |".format(
                name=item.get("display_name", item.get("model_id", "")),
                status=item.get("status", ""),
                profile=item.get("runtime", {}).get("effective_profile", ""),
                latency=_ms(timing.get("model_latency_ms", {}).get("mean")),
                e2e=_ms(timing.get("end_to_end_ms", {}).get("mean")),
                fps=_number(timing.get("fps")),
                vram=_gib(memory.get("peak_reserved_gib")),
                scale=output.get("depth_scale", ""),
            )
        )
    if failed or unavailable or cancelled:
        lines.extend(["", "## Non-passing models", ""])
        for item in (*failed, *unavailable, *cancelled):
            lines.append(
                f"- **{item.get('display_name', item.get('model_id'))}:** "
                f"{item.get('error') or item.get('availability_reason') or item.get('status')}"
            )
    lines.append("")
    return _atomic_write_text(destination, "\n".join(lines))


def _flatten_result(item: dict[str, object]) -> dict[str, object]:
    timing = item.get("timing", {}) or {}
    memory = item.get("memory", {}) or {}
    output = item.get("output", {}) or {}
    runtime = item.get("runtime", {}) or {}
    resolution = output.get("resolution") or [None, None]
    return {
        "model_id": item.get("model_id", ""),
        "display_name": item.get("display_name", ""),
        "status": item.get("status", ""),
        "classification": item.get("classification", ""),
        "workload": item.get("workload", ""),
        "profile": runtime.get("effective_profile", ""),
        "device": runtime.get("device", ""),
        "precision": runtime.get("precision", ""),
        "load_ms": timing.get("load_ms", ""),
        "warmup_ms": timing.get("warmup_ms", ""),
        "mean_model_latency_ms": (timing.get("model_latency_ms", {}) or {}).get("mean", ""),
        "mean_end_to_end_ms": (timing.get("end_to_end_ms", {}) or {}).get("mean", ""),
        "fps": timing.get("fps", ""),
        "peak_reserved_vram_gib": memory.get("peak_reserved_gib", ""),
        "device_used_vram_gib": memory.get("device_used_gib", ""),
        "output_width": resolution[0] if len(resolution) > 0 else "",
        "output_height": resolution[1] if len(resolution) > 1 else "",
        "point_count": output.get("point_count", ""),
        "depth_scale": output.get("depth_scale", ""),
        "is_metric": output.get("is_metric", ""),
        "intrinsics_source": output.get("intrinsics_source", ""),
        "confidence_kind": output.get("confidence_kind", ""),
        "error": item.get("error", ""),
    }


def _yes(value: bool) -> str:
    return "Yes" if value else "No"


def _ms(value: object) -> str:
    return "—" if value is None else f"{float(value):.1f} ms"


def _gib(value: object) -> str:
    return "—" if value is None else f"{float(value):.2f} GiB"


def _number(value: object) -> str:
    return "—" if value is None else f"{float(value):.2f}"


def _atomic_write_text(path: Path, text: str) -> Path:
    handle = tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    )
    temporary = Path(handle.name)
    try:
        with handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return path
