from __future__ import annotations

from core.reconstruction.adaptive_view_selection import AdaptiveSelectionResourceSnapshot
from core.reconstruction.reconstruction_frame_view_plan_contract import (
    FrameViewPlanMode,
    FrameViewPlanRequest,
    FrameViewSourceProvenance,
)
from core.reconstruction.reconstruction_frame_view_planning import resolve_deterministic_frame_view_plan


GIB = 1024**3
RESOURCES = AdaptiveSelectionResourceSnapshot(
    gpu_name="stress-resource-profile",
    total_vram_bytes=12 * GIB,
    free_vram_bytes=6 * GIB,
    available_ram_bytes=24 * GIB,
    cuda_ready=True,
)


def _source(total: int):
    return FrameViewSourceProvenance(
        source_id=f"stress-{total}",
        source_kind="image_sequence",
        source_uri=f"/stress/{total}",
        source_fingerprint_sha256=(f"{total:064x}"[-64:]),
        estimated_total_views=total,
        expected_frame_rate_hz=24.0,
    )


def test_many_source_sizes_never_exceed_known_source_or_create_global_cap():
    request = FrameViewPlanRequest(mode=FrameViewPlanMode.ADAPTIVE)
    for total in range(3, 1500, 17):
        result = resolve_deterministic_frame_view_plan(source=_source(total), request=request, resources=RESOURCES)
        target = result.resolution.effective_target_views
        resource = result.resolution.resource_target_views
        assert target is None or 3 <= target <= total
        assert resource is None or 3 <= resource <= total
        assert result.resolution.permanent_global_view_cap is None


def test_manual_limit_stress_never_exceeds_request_source_or_known_resource_bound():
    for total in range(3, 700, 23):
        requested = max(3, min(333, total + 77))
        request = FrameViewPlanRequest(mode=FrameViewPlanMode.MANUAL_LIMIT, requested_max_views=requested)
        result = resolve_deterministic_frame_view_plan(source=_source(total), request=request, resources=RESOURCES)
        target = result.resolution.effective_target_views
        assert target is not None
        assert 3 <= target <= requested
        assert target <= total
        if result.resolution.resource_target_views is not None:
            assert target <= result.resolution.resource_target_views
