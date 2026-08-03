"""Performance footer widget wiring and truthful display tests."""

from __future__ import annotations

from app.widgets.performance_bar import PerformanceBar
from app.widgets.performance_snapshot import PerformanceSnapshot


def test_footer_displays_complete_snapshot_and_tooltips(qtbot) -> None:
    bar = PerformanceBar()
    qtbot.addWidget(bar)
    snapshot = PerformanceSnapshot(
        status="RUNNING",
        camera_fps=14.7,
        inference_fps=6.6,
        latency_ms=95.5,
        render_fps=6.5,
        point_count=73_515,
        resolution=(1280, 1440),
        vram_used_gb=0.75,
        vram_total_gb=6.0,
        model_name="MoGe-2 ViT-S Normal",
        precision="FP16",
        uptime_seconds=65,
        source_frames_seen=100,
        frames_submitted=50,
        results_received=40,
        results_rendered=39,
        dropped_before_inference=50,
        dropped_before_render=1,
    )

    bar.update_snapshot(snapshot)

    assert bar._status_cell._badge.text() == "RUNNING"
    assert bar._cam_fps._value.text() == "14.7"
    assert bar._inf_fps._value.text() == "6.6"
    assert bar._latency._value.text() == "95.5 ms"
    assert bar._render_fps._value.text() == "6.5"
    assert bar._points._value.text() == "73,515"
    assert bar._resolution._value.text() == "1280×1440"
    assert bar._precision._value.text() == "FP16"
    assert bar._uptime._value.text() == "00:01:05"
    assert "Source frames seen: 100" in bar._status_cell.toolTip()


def test_static_image_snapshot_keeps_rates_unavailable(qtbot) -> None:
    bar = PerformanceBar()
    qtbot.addWidget(bar)
    bar.update_snapshot(
        PerformanceSnapshot(
            status="RESULT READY",
            latency_ms=64.2,
            point_count=174_801,
            model_name="MoGe-2 ViT-S Normal",
            precision="FP16",
        )
    )

    assert bar._status_cell._badge.text() == "RESULT READY"
    assert bar._cam_fps._value.text() == "—"
    assert bar._inf_fps._value.text() == "—"
    assert bar._render_fps._value.text() == "—"
    assert bar._points._value.text() == "174,801"
