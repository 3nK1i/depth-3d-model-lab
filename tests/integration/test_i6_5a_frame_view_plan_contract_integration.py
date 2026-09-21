from core.reconstruction.adaptive_production_planning import build_adaptive_pre_sfm_frame_plan
from core.reconstruction.adaptive_view_selection import AdaptiveSelectionResourceSnapshot
from core.reconstruction.reconstruction_frame_view_plan_contract import (
    FrameViewPlanMode,
    FrameViewPlanRequest,
    FrameViewPlanResolution,
    FrameViewPlanRealization,
    FrameViewRecord,
    FrameViewRecordDisposition,
    FrameViewSourceProvenance,
)


SHA = "b" * 64


def _source(total=128):
    return FrameViewSourceProvenance(
        source_id="image-sequence-south-building",
        source_kind="image_sequence",
        source_uri="/dataset/south-building/images",
        source_fingerprint_sha256=SHA,
        estimated_total_views=total,
        expected_frame_rate_hz=30.0,
    )


def test_i4_3f2_adaptive_target_can_be_bound_without_changing_existing_planner():
    resources = AdaptiveSelectionResourceSnapshot(
        gpu_name="RTX 2060",
        total_vram_bytes=6 * 1024**3,
        free_vram_bytes=4 * 1024**3,
        available_ram_bytes=8 * 1024**3,
        cuda_ready=True,
    )
    legacy = build_adaptive_pre_sfm_frame_plan(
        source_kind="image_directory",
        estimated_total_frames=128,
        expected_frame_rate_hz=30.0,
        resources=resources,
    )
    resolution = FrameViewPlanResolution(
        source=_source(),
        request=FrameViewPlanRequest(mode=FrameViewPlanMode.ADAPTIVE),
        effective_target_views=legacy.target_sampled_frames,
        resource_target_views=legacy.resource_target_frames,
        reason_codes=legacy.reason_codes,
    )
    assert resolution.effective_target_views == legacy.target_sampled_frames
    assert resolution.permanent_global_view_cap is None


def test_manual_limit_contract_preserves_requested_and_effective_values():
    request = FrameViewPlanRequest(mode=FrameViewPlanMode.MANUAL_LIMIT, requested_max_views=40)
    resolution = FrameViewPlanResolution(
        source=_source(),
        request=request,
        effective_target_views=40,
        resource_target_views=64,
        reason_codes=("manual_limit_requested", "manual_limit_effective"),
    )
    assert resolution.request.requested_max_views == 40
    assert resolution.effective_target_views == 40
    assert resolution.resource_target_views == 64


def test_all_eligible_realization_can_record_every_view_without_claiming_registration():
    source = _source(total=6)
    resolution = FrameViewPlanResolution(
        source=source,
        request=FrameViewPlanRequest(mode=FrameViewPlanMode.ALL_ELIGIBLE),
        effective_target_views=None,
        resource_target_views=None,
        reason_codes=("all_eligible_requested",),
    )
    records = tuple(
        FrameViewRecord(
            source_view_id=f"view-{i}",
            sequence_index=i,
            source_locator=f"/dataset/{i}.jpg",
            source_timestamp_ns=i * 33_333_333,
            disposition=FrameViewRecordDisposition.PLANNED,
            reason_codes=("all_eligible_view_planned",),
        )
        for i in range(6)
    )
    realized = FrameViewPlanRealization(resolution=resolution, records=records)
    assert realized.planned_view_count == 6
    assert "registered" not in realized.as_dict()["counts"]
    assert realized.as_dict()["authority"]["planned_views_are_registered_cameras"] is False


def test_contract_has_no_provider_or_auto_selection_effect():
    payload = FrameViewPlanResolution(
        source=_source(),
        request=FrameViewPlanRequest(mode=FrameViewPlanMode.ADAPTIVE),
        effective_target_views=80,
        resource_target_views=80,
        reason_codes=("adaptive_plan_resolved",),
    ).as_dict()
    assert payload["effects"]["provider_selected"] is False
    assert payload["effects"]["production_auto_activated"] is False
    assert payload["effects"]["execution_started"] is False
    assert payload["effects"]["scientific_threshold_modified"] is False
