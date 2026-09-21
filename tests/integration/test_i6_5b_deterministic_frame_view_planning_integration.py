from __future__ import annotations

from core.reconstruction.adaptive_view_selection import AdaptiveSelectionResourceSnapshot
from core.reconstruction.reconstruction_frame_view_plan_contract import (
    FrameViewPlanMode,
    FrameViewPlanRequest,
    FrameViewSourceProvenance,
)
from core.reconstruction.reconstruction_frame_view_planning import resolve_deterministic_frame_view_plan


GIB = 1024**3


def _source(total: int | None = 2400):
    return FrameViewSourceProvenance(
        source_id="integration-sequence",
        source_kind="recorded_video",
        source_uri="/dataset/sequence.mp4",
        source_fingerprint_sha256="2" * 64,
        estimated_total_views=total,
        expected_frame_rate_hz=30.0,
    )


def _resources(*, ram: int, vram: int):
    return AdaptiveSelectionResourceSnapshot(
        gpu_name=f"synthetic-{vram}g",
        total_vram_bytes=max(6, vram) * GIB,
        free_vram_bytes=vram * GIB,
        available_ram_bytes=ram * GIB,
        cuda_ready=True,
    )


def test_more_available_resources_never_reduce_adaptive_target_for_same_source():
    request = FrameViewPlanRequest(mode=FrameViewPlanMode.ADAPTIVE)
    low = resolve_deterministic_frame_view_plan(source=_source(), request=request, resources=_resources(ram=8, vram=2))
    high = resolve_deterministic_frame_view_plan(source=_source(), request=request, resources=_resources(ram=64, vram=28))
    assert low.resolution.effective_target_views is not None
    assert high.resolution.effective_target_views is not None
    assert high.resolution.effective_target_views >= low.resolution.effective_target_views
    assert high.resolution.effective_target_views <= 2400


def test_manual_request_preserves_requested_value_but_records_smaller_effective_plan():
    request = FrameViewPlanRequest(mode=FrameViewPlanMode.MANUAL_LIMIT, requested_max_views=200)
    result = resolve_deterministic_frame_view_plan(source=_source(), request=request, resources=_resources(ram=8, vram=2))
    assert result.resolution.request.requested_max_views == 200
    assert result.resolution.effective_target_views == 64
    assert result.resolution.plan_input_sha256
    assert result.planning_provenance_sha256


def test_all_eligible_resource_pressure_is_visible_without_mode_substitution():
    request = FrameViewPlanRequest(mode=FrameViewPlanMode.ALL_ELIGIBLE)
    result = resolve_deterministic_frame_view_plan(source=_source(), request=request, resources=_resources(ram=8, vram=2))
    assert result.resolution.request.mode is FrameViewPlanMode.ALL_ELIGIBLE
    assert result.resolution.effective_target_views is None
    assert result.all_eligible_exceeds_resource_bound is True
    assert result.as_dict()["policy"]["request_mode_substituted"] is False


def test_repeated_planning_is_stable_for_identical_source_request_and_resources():
    request = FrameViewPlanRequest(mode=FrameViewPlanMode.ADAPTIVE)
    digests = {
        resolve_deterministic_frame_view_plan(
            source=_source(total=512),
            request=request,
            resources=_resources(ram=16, vram=4),
        ).planning_provenance_sha256
        for _ in range(25)
    }
    assert len(digests) == 1
