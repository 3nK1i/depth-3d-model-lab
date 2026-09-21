from __future__ import annotations

import pytest

from core.reconstruction.adaptive_view_selection import AdaptiveSelectionResourceSnapshot
from core.reconstruction.errors import ReconstructionValidationError
from core.reconstruction.reconstruction_frame_view_plan_contract import (
    FrameViewPlanMode,
    FrameViewPlanRequest,
    FrameViewSourceProvenance,
)
from core.reconstruction.reconstruction_frame_view_planning import (
    I65B_PHASE,
    resolve_deterministic_frame_view_plan,
)


def _source(*, total: int | None = 128, fingerprint: str = "1" * 64):
    return FrameViewSourceProvenance(
        source_id="south-building-sequence",
        source_kind="image_sequence",
        source_uri="/dataset/south-building/images",
        source_fingerprint_sha256=fingerprint,
        estimated_total_views=total,
        expected_frame_rate_hz=30.0,
    )


def _resources(*, ram_gib: int | None, free_vram_gib: int | None, name: str = "synthetic"):
    gib = 1024**3
    return AdaptiveSelectionResourceSnapshot(
        gpu_name=name,
        total_vram_bytes=None if free_vram_gib is None else max(6, free_vram_gib) * gib,
        free_vram_bytes=None if free_vram_gib is None else free_vram_gib * gib,
        available_ram_bytes=None if ram_gib is None else ram_gib * gib,
        cuda_ready=free_vram_gib is not None,
    )


def test_adaptive_resolution_reuses_existing_resource_planner_without_global_cap():
    result = resolve_deterministic_frame_view_plan(
        source=_source(total=2400),
        request=FrameViewPlanRequest(mode=FrameViewPlanMode.ADAPTIVE),
        resources=_resources(ram_gib=8, free_vram_gib=2),
    )
    assert result.phase == I65B_PHASE
    assert result.resolution.effective_target_views == 64
    assert result.resolution.resource_target_views == 64
    assert result.resource_bound_applied is True
    assert result.resolution.permanent_global_view_cap is None
    assert "legacy_i4_3f2_resource_planner_reused" in result.resolution.reason_codes


def test_known_small_source_clamps_resource_target_to_source_count():
    result = resolve_deterministic_frame_view_plan(
        source=_source(total=47),
        request=FrameViewPlanRequest(mode=FrameViewPlanMode.ADAPTIVE),
        resources=_resources(ram_gib=8, free_vram_gib=2),
    )
    assert result.resolution.effective_target_views == 47
    assert result.resolution.resource_target_views == 47
    assert result.hardware_bounds.raw_resource_target_views == 64
    assert result.hardware_bounds.effective_resource_target_views == 47
    assert result.hardware_bounds.source_count_clamped is True


def test_manual_limit_is_bounded_by_request_and_resource_target():
    result = resolve_deterministic_frame_view_plan(
        source=_source(total=500),
        request=FrameViewPlanRequest(mode=FrameViewPlanMode.MANUAL_LIMIT, requested_max_views=120),
        resources=_resources(ram_gib=8, free_vram_gib=2),
    )
    assert result.resolution.effective_target_views == 64
    assert result.resource_bound_applied is True
    assert "manual_limit_resource_bound_applied" in result.resolution.reason_codes


def test_manual_limit_is_bounded_by_known_source_count_without_claiming_resource_pressure():
    result = resolve_deterministic_frame_view_plan(
        source=_source(total=40),
        request=FrameViewPlanRequest(mode=FrameViewPlanMode.MANUAL_LIMIT, requested_max_views=80),
        resources=_resources(ram_gib=64, free_vram_gib=28),
    )
    assert result.resolution.effective_target_views == 40
    assert result.resource_bound_applied is False
    assert "manual_limit_source_count_bound_applied" in result.resolution.reason_codes


def test_all_eligible_never_receives_a_silent_adaptive_cap():
    result = resolve_deterministic_frame_view_plan(
        source=_source(total=2400),
        request=FrameViewPlanRequest(mode=FrameViewPlanMode.ALL_ELIGIBLE),
        resources=_resources(ram_gib=8, free_vram_gib=2),
    )
    assert result.resolution.effective_target_views is None
    assert result.resolution.resource_target_views == 64
    assert result.resource_bound_applied is False
    assert result.all_eligible_exceeds_resource_bound is True
    assert result.execution_resource_preflight_required is True
    assert "all_eligible_exceeds_resource_planning_bound" in result.resolution.reason_codes


def test_all_eligible_within_resource_bound_remains_uncapped():
    result = resolve_deterministic_frame_view_plan(
        source=_source(total=32),
        request=FrameViewPlanRequest(mode=FrameViewPlanMode.ALL_ELIGIBLE),
        resources=_resources(ram_gib=8, free_vram_gib=2),
    )
    assert result.resolution.effective_target_views is None
    assert result.resolution.resource_target_views == 32
    assert result.all_eligible_exceeds_resource_bound is False
    assert "all_eligible_within_resource_planning_bound" in result.resolution.reason_codes


def test_unknown_resources_leave_adaptive_request_uncapped_instead_of_inventing_limit():
    result = resolve_deterministic_frame_view_plan(
        source=_source(total=None),
        request=FrameViewPlanRequest(mode=FrameViewPlanMode.ADAPTIVE),
        resources=_resources(ram_gib=None, free_vram_gib=None),
    )
    assert result.resolution.effective_target_views is None
    assert result.resolution.resource_target_views is None
    assert result.resource_bound_applied is False
    assert "resource_capacity_unknown_no_fixed_pre_sfm_cap" in result.resolution.reason_codes


def test_unknown_resources_preserve_explicit_manual_limit():
    result = resolve_deterministic_frame_view_plan(
        source=_source(total=None),
        request=FrameViewPlanRequest(mode=FrameViewPlanMode.MANUAL_LIMIT, requested_max_views=33),
        resources=_resources(ram_gib=None, free_vram_gib=None),
    )
    assert result.resolution.effective_target_views == 33
    assert result.resolution.resource_target_views is None
    assert result.resource_bound_applied is False


def test_hardware_provenance_digest_is_deterministic_and_resource_sensitive():
    request = FrameViewPlanRequest(mode=FrameViewPlanMode.ADAPTIVE)
    source = _source(total=500)
    first = resolve_deterministic_frame_view_plan(
        source=source,
        request=request,
        resources=_resources(ram_gib=8, free_vram_gib=2, name="gpu-a"),
    )
    second = resolve_deterministic_frame_view_plan(
        source=source,
        request=request,
        resources=_resources(ram_gib=8, free_vram_gib=2, name="gpu-a"),
    )
    changed = resolve_deterministic_frame_view_plan(
        source=source,
        request=request,
        resources=_resources(ram_gib=16, free_vram_gib=4, name="gpu-b"),
    )
    assert first.planning_provenance_sha256 == second.planning_provenance_sha256
    assert first.hardware_bounds.hardware_context_sha256 == second.hardware_bounds.hardware_context_sha256
    assert first.planning_provenance_sha256 != changed.planning_provenance_sha256


def test_resource_planning_is_driven_by_numeric_resources_not_gpu_brand_name():
    source = _source(total=500)
    request = FrameViewPlanRequest(mode=FrameViewPlanMode.ADAPTIVE)
    left = resolve_deterministic_frame_view_plan(
        source=source,
        request=request,
        resources=_resources(ram_gib=8, free_vram_gib=2, name="GPU A"),
    )
    right = resolve_deterministic_frame_view_plan(
        source=source,
        request=request,
        resources=_resources(ram_gib=8, free_vram_gib=2, name="GPU B"),
    )
    assert left.resolution.effective_target_views == right.resolution.effective_target_views
    assert left.resolution.resource_target_views == right.resolution.resource_target_views
    assert left.hardware_bounds.planning_capacity_units == right.hardware_bounds.planning_capacity_units


def test_source_with_fewer_than_three_known_views_fails_closed():
    with pytest.raises(ReconstructionValidationError, match="at least 3"):
        resolve_deterministic_frame_view_plan(
            source=_source(total=2),
            request=FrameViewPlanRequest(mode=FrameViewPlanMode.ADAPTIVE),
            resources=_resources(ram_gib=8, free_vram_gib=2),
        )


def test_result_explicitly_denies_retry_until_pass_and_execution_effects():
    result = resolve_deterministic_frame_view_plan(
        source=_source(total=128),
        request=FrameViewPlanRequest(mode=FrameViewPlanMode.ADAPTIVE),
        resources=_resources(ram_gib=8, free_vram_gib=2),
    )
    payload = result.as_dict()
    assert payload["policy"]["retry_until_pass"] is False
    assert payload["policy"]["silent_replan_after_sfm_failure"] is False
    assert payload["effects"]["sampling_executed"] is False
    assert payload["effects"]["worker_modified"] is False
    assert payload["authority"]["planned_views_are_registered_cameras"] is False
