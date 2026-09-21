from __future__ import annotations

from types import SimpleNamespace

from app.reconstruction_frame_view_plan_state import (
    FrameViewPlanIntentOrigin,
    build_reconstruction_frame_view_plan_state,
)
from app.reconstruction_frame_view_plan_presentation import present_reconstruction_frame_view_plan
from core.reconstruction.adaptive_view_selection import AdaptiveSelectionResourceSnapshot
from core.reconstruction.reconstruction_frame_view_plan_contract import FrameViewPlanMode, FrameViewPlanRequest

GIB = 1024**3


def _source(tmp_path, count: int = 8):
    path = tmp_path / "images"
    path.mkdir()
    for index in range(count):
        (path / f"frame-{index:03d}.jpg").write_bytes(b"fixture")
    return SimpleNamespace(ready=True, workflow="image_sequence", path=path)


def _resources(*, ram_gib: int = 8, vram_gib: int = 6):
    return AdaptiveSelectionResourceSnapshot(
        gpu_name="RTX Test",
        total_vram_bytes=vram_gib * GIB,
        free_vram_bytes=vram_gib * GIB,
        available_ram_bytes=ram_gib * GIB,
        cuda_ready=True,
    )


def test_i6_5d_waiting_source_is_neutral() -> None:
    state = build_reconstruction_frame_view_plan_state(
        source_selection=SimpleNamespace(ready=False, workflow="none", path=None),
        request=FrameViewPlanRequest(mode=FrameViewPlanMode.ADAPTIVE),
        intent_origin=FrameViewPlanIntentOrigin.DEFAULT_ADAPTIVE,
        resources=_resources(),
    )
    ui = present_reconstruction_frame_view_plan(state)
    assert ui.status_text == "Waiting for source"
    assert ui.status_kind == "neutral"
    assert ui.mode_text == "Adaptive"
    assert ui.source_text == "No source"
    assert ui.planned_text == "Not resolved"


def test_i6_5d_resolved_adaptive_plan_is_compact_and_explanatory(tmp_path) -> None:
    state = build_reconstruction_frame_view_plan_state(
        source_selection=_source(tmp_path, 8),
        request=FrameViewPlanRequest(mode=FrameViewPlanMode.ADAPTIVE),
        intent_origin=FrameViewPlanIntentOrigin.DEFAULT_ADAPTIVE,
        resources=_resources(),
    )
    ui = present_reconstruction_frame_view_plan(state)
    assert ui.status_text == "Adaptive plan ready"
    assert ui.status_kind == "ready"
    assert ui.mode_text == "Adaptive"
    assert ui.source_text == "8 source views"
    assert "planned" in ui.planned_text
    assert "RTX Test" in ui.hardware_text
    assert ui.apply_enabled is True
    assert ui.reset_enabled is False


def test_i6_5d_manual_limit_remains_request_not_exact_execution_count(tmp_path) -> None:
    state = build_reconstruction_frame_view_plan_state(
        source_selection=_source(tmp_path, 40),
        request=FrameViewPlanRequest(mode=FrameViewPlanMode.MANUAL_LIMIT, requested_max_views=30),
        intent_origin=FrameViewPlanIntentOrigin.EXPLICIT_USER_OVERRIDE,
        resources=_resources(ram_gib=1, vram_gib=1),
    )
    ui = present_reconstruction_frame_view_plan(state)
    assert ui.mode_text == "Manual limit · max 30"
    assert ui.status_text == "Manual plan ready"
    assert "maximum" in ui.boundary_tooltip
    assert ui.manual_limit_value == 30
    assert ui.manual_limit_enabled is True
    assert ui.reset_enabled is True


def test_i6_5d_all_eligible_pressure_is_warning_not_silent_cap(tmp_path) -> None:
    state = build_reconstruction_frame_view_plan_state(
        source_selection=_source(tmp_path, 200),
        request=FrameViewPlanRequest(mode=FrameViewPlanMode.ALL_ELIGIBLE),
        intent_origin=FrameViewPlanIntentOrigin.EXPLICIT_USER_OVERRIDE,
        resources=_resources(ram_gib=1, vram_gib=1),
    )
    ui = present_reconstruction_frame_view_plan(state)
    assert ui.status_text == "All eligible · resource review"
    assert ui.status_kind == "warning"
    assert ui.planned_text == "All 200 eligible"
    assert "not silently capped" in ui.boundary_tooltip


def test_i6_5d_presentation_never_claims_registered_camera_or_execution(tmp_path) -> None:
    state = build_reconstruction_frame_view_plan_state(
        source_selection=_source(tmp_path, 8),
        request=FrameViewPlanRequest(mode=FrameViewPlanMode.ADAPTIVE),
        intent_origin=FrameViewPlanIntentOrigin.DEFAULT_ADAPTIVE,
        resources=_resources(),
    )
    ui = present_reconstruction_frame_view_plan(state)
    assert "pre-sfm intent" in ui.boundary_text.lower()
    assert "PyCOLMAP" in ui.boundary_tooltip
    assert "registered camera" in ui.boundary_tooltip
