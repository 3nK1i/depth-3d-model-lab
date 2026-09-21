from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from app.reconstruction_frame_view_plan_state import (
    FrameViewPlanIntentOrigin,
    FrameViewPlanStateStatus,
    build_reconstruction_frame_view_plan_state,
)
from core.reconstruction.adaptive_view_selection import AdaptiveSelectionResourceSnapshot
from core.reconstruction.reconstruction_frame_view_plan_contract import (
    FrameViewPlanMode,
    FrameViewPlanRequest,
)

GIB = 1024**3


class Workflow(str, Enum):
    IMAGE_SEQUENCE = "image_sequence"


@dataclass(frozen=True)
class Selection:
    workflow: Workflow
    ready: bool
    path: Path
    item_count: int


def _selection(tmp_path: Path, count: int = 256) -> Selection:
    root = tmp_path / "images"
    root.mkdir()
    for index in range(count):
        (root / f"{index:05d}.jpg").write_bytes(b"x")
    return Selection(Workflow.IMAGE_SEQUENCE, True, root, count)


def _resources(ram: int, vram: int) -> AdaptiveSelectionResourceSnapshot:
    return AdaptiveSelectionResourceSnapshot(
        gpu_name="stress-gpu",
        total_vram_bytes=32 * GIB,
        free_vram_bytes=vram * GIB,
        available_ram_bytes=ram * GIB,
        cuda_ready=True,
    )


def test_i6_5c_repeated_same_inputs_produce_same_plan_digest(tmp_path: Path) -> None:
    selection = _selection(tmp_path)
    request = FrameViewPlanRequest(mode=FrameViewPlanMode.ADAPTIVE)
    values = {
        build_reconstruction_frame_view_plan_state(
            source_selection=selection,
            request=request,
            intent_origin=FrameViewPlanIntentOrigin.DEFAULT_ADAPTIVE,
            resources=_resources(16, 8),
        ).planning_provenance_sha256
        for _ in range(100)
    }
    assert len(values) == 1


def test_i6_5c_resource_changes_change_plan_provenance(tmp_path: Path) -> None:
    selection = _selection(tmp_path)
    request = FrameViewPlanRequest(mode=FrameViewPlanMode.ADAPTIVE)
    values = []
    for ram, vram in ((4, 1), (8, 2), (16, 4), (32, 8)):
        state = build_reconstruction_frame_view_plan_state(
            source_selection=selection,
            request=request,
            intent_origin=FrameViewPlanIntentOrigin.DEFAULT_ADAPTIVE,
            resources=_resources(ram, vram),
        )
        assert state.status is FrameViewPlanStateStatus.RESOLVED
        values.append(state.planning_provenance_sha256)
    assert len(set(values)) == len(values)


def test_i6_5c_manual_requests_remain_requests_across_resource_profiles(tmp_path: Path) -> None:
    selection = _selection(tmp_path)
    request = FrameViewPlanRequest(mode=FrameViewPlanMode.MANUAL_LIMIT, requested_max_views=200)
    for ram, vram in ((1, 0), (4, 1), (16, 8), (64, 24)):
        state = build_reconstruction_frame_view_plan_state(
            source_selection=selection,
            request=request,
            intent_origin=FrameViewPlanIntentOrigin.EXPLICIT_USER_OVERRIDE,
            resources=_resources(ram, vram),
        )
        assert state.request.requested_max_views == 200
        assert state.effective_target_views is not None
        assert state.effective_target_views <= 200
