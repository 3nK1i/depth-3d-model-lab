from core.reconstruction.reconstruction_frame_view_plan_contract import (
    FrameViewPlanMode,
    FrameViewPlanRequest,
    FrameViewPlanResolution,
    FrameViewPlanRealization,
    FrameViewRecord,
    FrameViewRecordDisposition,
    FrameViewSourceProvenance,
)


def _large_realization(total=512, target=128):
    source = FrameViewSourceProvenance(
        source_id="stress-sequence",
        source_kind="image_sequence",
        source_uri="/stress/images",
        source_fingerprint_sha256="c" * 64,
        estimated_total_views=total,
        expected_frame_rate_hz=24.0,
    )
    resolution = FrameViewPlanResolution(
        source=source,
        request=FrameViewPlanRequest(mode=FrameViewPlanMode.ADAPTIVE),
        effective_target_views=target,
        resource_target_views=target,
        reason_codes=("stress_adaptive_plan",),
    )
    selected = set(round(i * (total - 1) / (target - 1)) for i in range(target))
    records = tuple(
        FrameViewRecord(
            source_view_id=f"view-{i:05d}",
            sequence_index=i,
            source_locator=f"/stress/images/{i:05d}.jpg",
            source_timestamp_ns=i * 41_666_667,
            disposition=(
                FrameViewRecordDisposition.PLANNED
                if i in selected
                else FrameViewRecordDisposition.ELIGIBLE_NOT_PLANNED
            ),
            reason_codes=(
                "selected_by_effective_plan" if i in selected else "not_selected_by_effective_target",
            ),
        )
        for i in range(total)
    )
    return FrameViewPlanRealization(resolution=resolution, records=records)


def test_large_plan_provenance_is_stable_across_repeated_construction():
    digests = {_large_realization().plan_provenance_sha256 for _ in range(20)}
    assert len(digests) == 1


def test_large_plan_counts_remain_exact_without_global_cap():
    realized = _large_realization(total=1000, target=257)
    assert realized.observed_source_view_count == 1000
    assert realized.eligible_view_count == 1000
    assert realized.planned_view_count == 257
    assert realized.resolution.permanent_global_view_cap is None
