"""Complete footer formatting and snapshot-copy tests."""

from __future__ import annotations

import pytest

from app.widgets.performance_snapshot import (
    PLACEHOLDER,
    PerformanceSnapshot,
    format_fps,
    format_latency,
    format_or_dash,
    format_points,
    format_resolution,
    format_uptime,
    format_vram,
    status_color,
    vram_ratio,
)


def test_placeholder_formatters_do_not_fabricate_values() -> None:
    assert format_fps(None) == PLACEHOLDER
    assert format_latency(None) == PLACEHOLDER
    assert format_points(None) == PLACEHOLDER
    assert format_resolution(None) == PLACEHOLDER
    assert format_vram(None, 6.0) == PLACEHOLDER
    assert format_or_dash("") == PLACEHOLDER


def test_numeric_formatters_are_stable() -> None:
    assert format_fps(9.94) == "9.9"
    assert format_latency(64.24) == "64.2 ms"
    assert format_points(174_801) == "174,801"
    assert format_resolution((1536, 1024)) == "1536×1024"
    assert format_vram(0.75, 6.0) == "0.8 / 6 GB"
    assert format_uptime(3661) == "01:01:01"


def test_vram_ratio_clamps_invalid_or_out_of_range_values() -> None:
    assert vram_ratio(None, 6.0) == 0.0
    assert vram_ratio(1.0, 0.0) == 0.0
    assert vram_ratio(-1.0, 6.0) == 0.0
    assert vram_ratio(10.0, 6.0) == 1.0


def test_status_color_has_safe_fallback() -> None:
    assert status_color("running") == "green"
    assert status_color("paused") == "amber"
    assert status_color("error") == "red"
    assert status_color("unknown") == "neutral"


def test_snapshot_copy_helpers_preserve_existing_metrics() -> None:
    snapshot = PerformanceSnapshot(status="RUNNING", latency_ms=80.0, point_count=100)
    paused = snapshot.with_status("PAUSED").with_uptime(15)
    assert paused.status == "PAUSED"
    assert paused.uptime_seconds == 15
    assert paused.latency_ms == 80.0
    assert paused.point_count == 100


def test_model_tooltip_exposes_load_and_warmup_details() -> None:
    snapshot = PerformanceSnapshot(
        status="MODEL READY",
        model_name="MoGe-2 ViT-S Normal",
        model_device="cuda:0",
        precision="FP16",
        checkpoint="Ruicheng/moge-2-vits-normal",
        num_tokens=1200,
        resolution_level=5,
        hardware_tier="efficient",
        warmup_runs=1,
        tf32_enabled=False,
        native_sdpa_enabled=True,
        model_load_time_ms=100.0,
        warmup_time_ms=20.0,
        total_load_time_ms=120.0,
    )
    tooltip = snapshot.model_tooltip()
    assert "cuda:0" in tooltip
    assert "Ruicheng/moge-2-vits-normal" in tooltip
    assert "Hardware tier: Efficient" in tooltip
    assert "Warm-up runs: 1" in tooltip
    assert "TF32 enabled: False" in tooltip
    assert "Native SDPA enabled: True" in tooltip
    assert "Warm-up: 20.0 ms" in tooltip
    assert "Total load + warm-up: 120.0 ms" in tooltip
