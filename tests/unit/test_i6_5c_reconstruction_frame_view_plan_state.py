from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import pytest

from app.reconstruction_frame_view_plan_state import (
    FrameViewPlanIntentOrigin,
    FrameViewPlanStateStatus,
    build_reconstruction_frame_view_plan_state,
    source_provenance_from_selection,
)
from core.reconstruction.adaptive_view_selection import AdaptiveSelectionResourceSnapshot
from core.reconstruction.reconstruction_frame_view_plan_contract import (
    FrameViewPlanMode,
    FrameViewPlanRequest,
)

GIB = 1024**3


class Workflow(str, Enum):
    NONE = "none"
    IMAGE_SEQUENCE = "image_sequence"
    RECORDED_VIDEO = "recorded_video"
    LIVE_SOURCE = "live_source"


@dataclass(frozen=True)
class Selection:
    workflow: Workflow
    ready: bool
    path: Path | None = None
    item_count: int | None = None


def resources(ram=8, free_vram=2):
    return AdaptiveSelectionResourceSnapshot(
        gpu_name="RTX 2060",
        total_vram_bytes=6 * GIB,
        free_vram_bytes=free_vram * GIB,
        available_ram_bytes=ram * GIB,
        cuda_ready=True,
    )


def images(tmp_path: Path, count=8) -> Path:
    root = tmp_path / "images"
    root.mkdir()
    for i in range(count):
        (root / f"frame-{i:03d}.jpg").write_bytes(f"frame-{i}".encode())
    return root


def build(selection, request=None, res=None, origin=FrameViewPlanIntentOrigin.DEFAULT_ADAPTIVE):
    return build_reconstruction_frame_view_plan_state(
        source_selection=selection,
        request=request or FrameViewPlanRequest(mode=FrameViewPlanMode.ADAPTIVE),
        intent_origin=origin,
        resources=res,
    )


def test_i6_5c_waits_for_source_without_inventing_plan() -> None:
    state = build(Selection(Workflow.NONE, False))
    assert state.status is FrameViewPlanStateStatus.WAITING_SOURCE
    assert state.plan is None
    assert state.reason_codes == ("source_not_ready",)


def test_i6_5c_ready_source_waits_for_explicit_resource_snapshot(tmp_path: Path) -> None:
    root = images(tmp_path)
    state = build(Selection(Workflow.IMAGE_SEQUENCE, True, root, 8))
    assert state.status is FrameViewPlanStateStatus.WAITING_RESOURCES
    assert state.source is not None
    assert state.source.estimated_total_views == 8
    assert state.plan is None


def test_i6_5c_adaptive_resolves_deterministically(tmp_path: Path) -> None:
    root = images(tmp_path, 128)
    state = build(Selection(Workflow.IMAGE_SEQUENCE, True, root, 128), res=resources())
    assert state.status is FrameViewPlanStateStatus.RESOLVED
    assert state.request.mode is FrameViewPlanMode.ADAPTIVE
    assert state.effective_target_views is not None
    assert len(state.planning_provenance_sha256) == 64
    assert state.plan is not None
    assert state.plan.as_dict()["policy"]["retry_until_pass"] is False


def test_i6_5c_manual_limit_preserves_request_vs_effective(tmp_path: Path) -> None:
    root = images(tmp_path, 120)
    request = FrameViewPlanRequest(mode=FrameViewPlanMode.MANUAL_LIMIT, requested_max_views=100)
    state = build(
        Selection(Workflow.IMAGE_SEQUENCE, True, root, 120),
        request=request,
        res=resources(),
        origin=FrameViewPlanIntentOrigin.EXPLICIT_USER_OVERRIDE,
    )
    assert state.status is FrameViewPlanStateStatus.RESOLVED
    assert state.request.requested_max_views == 100
    assert state.effective_target_views is not None
    assert state.effective_target_views <= 100
    assert state.intent_origin is FrameViewPlanIntentOrigin.EXPLICIT_USER_OVERRIDE


def test_i6_5c_all_eligible_stays_uncapped_under_resource_pressure(tmp_path: Path) -> None:
    root = images(tmp_path, 160)
    state = build(
        Selection(Workflow.IMAGE_SEQUENCE, True, root, 160),
        request=FrameViewPlanRequest(mode=FrameViewPlanMode.ALL_ELIGIBLE),
        res=resources(ram=1, free_vram=0),
        origin=FrameViewPlanIntentOrigin.EXPLICIT_USER_OVERRIDE,
    )
    assert state.status is FrameViewPlanStateStatus.RESOLVED
    assert state.effective_target_views is None
    assert state.plan is not None
    assert state.plan.all_eligible_exceeds_resource_bound is True


def test_i6_5c_video_does_not_misuse_file_count_as_frame_count(tmp_path: Path) -> None:
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"video")
    provenance = source_provenance_from_selection(
        Selection(Workflow.RECORDED_VIDEO, True, video, 1)
    )
    assert provenance is not None
    assert provenance.estimated_total_views is None
    assert provenance.source_kind == "recorded_video"


def test_i6_5c_live_source_is_not_offline_pre_sfm_plan(tmp_path: Path) -> None:
    state = build(Selection(Workflow.LIVE_SOURCE, True), res=resources())
    assert state.status is FrameViewPlanStateStatus.UNSUPPORTED_WORKFLOW
    assert state.plan is None


def test_i6_5c_source_manifest_fingerprint_changes_when_metadata_changes(tmp_path: Path) -> None:
    root = images(tmp_path, 4)
    selection = Selection(Workflow.IMAGE_SEQUENCE, True, root, 4)
    first = source_provenance_from_selection(selection)
    assert first is not None
    target = root / "frame-000.jpg"
    target.write_bytes(b"changed-content-length")
    second = source_provenance_from_selection(selection)
    assert second is not None
    assert first.source_fingerprint_sha256 != second.source_fingerprint_sha256


def test_i6_5c_missing_selected_source_fails_closed(tmp_path: Path) -> None:
    root = images(tmp_path, 4)
    selection = Selection(Workflow.IMAGE_SEQUENCE, True, root, 4)
    for item in root.iterdir():
        item.unlink()
    state = build(selection, res=resources())
    assert state.status is FrameViewPlanStateStatus.BLOCKED
    assert state.plan is None
    assert state.reason_codes[0] == "source_provenance_invalid"


def test_i6_5c_state_serialization_declares_zero_runtime_effect(tmp_path: Path) -> None:
    root = images(tmp_path, 8)
    state = build(Selection(Workflow.IMAGE_SEQUENCE, True, root, 8), res=resources())
    payload = state.as_dict()
    assert payload["effects"]["source_sampling_executed"] is False
    assert payload["effects"]["sfm_execution_started"] is False
    assert payload["authority"]["planned_views_are_registered_cameras"] is False


def test_i6_5c_state_is_immutable(tmp_path: Path) -> None:
    root = images(tmp_path, 4)
    state = build(Selection(Workflow.IMAGE_SEQUENCE, True, root, 4), res=resources())
    with pytest.raises(Exception):
        state.status = FrameViewPlanStateStatus.WAITING_SOURCE  # type: ignore[misc]
