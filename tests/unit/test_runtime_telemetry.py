"""Pure tests for Phase 5 Step 3 telemetry helpers and formatting."""

from __future__ import annotations

import pytest

from app.widgets.performance_snapshot import PerformanceSnapshot, format_vram, vram_ratio
from core.runtime.telemetry import RateMeter, RuntimeCounters


def test_rate_meter_reports_transition_rate_over_sliding_window() -> None:
    meter = RateMeter(window_seconds=2.0)

    assert meter.mark(10.0) is None
    assert meter.mark(10.1) == pytest.approx(10.0)
    assert meter.mark(10.2) == pytest.approx(10.0)

    # Old events are trimmed while at least two recent samples are retained.
    meter.mark(12.2)
    assert meter.rate == pytest.approx(0.5)


def test_rate_meter_rejects_invalid_window() -> None:
    with pytest.raises(ValueError, match="positive"):
        RateMeter(0.0)


def test_runtime_counters_separate_pre_inference_and_pre_render_drops() -> None:
    counters = RuntimeCounters(
        pending_frames_replaced=4,
        submit_rejected=1,
        pending_results_replaced=2,
        stale_results_ignored=3,
    )

    assert counters.dropped_before_inference == 5
    assert counters.dropped_before_render == 5
    assert counters.total_dropped == 10

    counters.reset()
    assert counters.total_dropped == 0


def test_snapshot_latency_tooltip_explains_visible_end_to_end_value() -> None:
    snapshot = PerformanceSnapshot(
        status="RUNNING",
        latency_ms=101.5,
        model_latency_ms=64.2,
        preprocess_ms=1.2,
        postprocess_ms=18.4,
        worker_total_ms=84.0,
        geometry_ms=12.0,
    )

    tooltip = snapshot.latency_tooltip()
    assert "end-to-end" in tooltip.lower()
    assert "101.5 ms" in tooltip
    assert "Model forward pass: 64.2 ms" in tooltip
    assert "UI depth + geometry + canvas: 12.0 ms" in tooltip


def test_vram_display_is_reserved_and_tooltip_keeps_allocator_breakdown() -> None:
    snapshot = PerformanceSnapshot(
        status="RUNNING",
        vram_used_gb=0.75,
        vram_reserved_gb=0.75,
        vram_allocated_gb=0.60,
        vram_peak_reserved_gb=0.80,
        vram_device_used_gb=1.10,
        vram_total_gb=6.0,
    )

    assert format_vram(snapshot.vram_used_gb, snapshot.vram_total_gb) == "0.8 / 6 GB"
    assert vram_ratio(snapshot.vram_used_gb, snapshot.vram_total_gb) == pytest.approx(0.125)
    tooltip = snapshot.vram_tooltip()
    assert "Allocated tensors: 0.60 GiB" in tooltip
    assert "Reserved by PyTorch allocator: 0.75 GiB" in tooltip
    assert "Device-wide used memory: 1.10 GiB" in tooltip
