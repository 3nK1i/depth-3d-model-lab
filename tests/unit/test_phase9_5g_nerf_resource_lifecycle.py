from __future__ import annotations

from dataclasses import replace

import pytest

from app.reconstruction_nerf_resources import (
    LatestRenderRequestQueue,
    NerfInteractiveResourcePolicy,
    NerfViewerResidencyState,
    build_nerf_interactive_resource_policy,
    build_nerf_viewer_runtime_health,
)
from core.hardware.gpu_info import AcceleratorBackend, HardwareInfo


def _hardware(*, gib: int, cuda: bool = True) -> HardwareInfo:
    return HardwareInfo(
        AcceleratorBackend.CUDA if cuda else AcceleratorBackend.CPU,
        "cuda:0" if cuda else "cpu",
        0 if cuda else None,
        "NVIDIA GeForce RTX 2060" if cuda else "CPU",
        gib * 1024**3 if cuda else None,
        gib * 1024**3 if cuda else None,
        (7, 5) if cuda else None,
        cuda,
        False,
        True,
        cuda,
        "test",
        "12.8" if cuda else None,
        ("sm_75",) if cuda else (),
        True if cuda else None,
    )


def _render(request_id: int) -> dict[str, object]:
    return {
        "type": "render",
        "request_id": request_id,
        "scene_role": "appearance_only",
    }


def test_5g_resource_policy_selects_conservative_rtx2060_tier_without_claiming_authority() -> None:
    policy = build_nerf_interactive_resource_policy(_hardware(gib=6))
    assert policy.resource_tier == "conservative-6gb"
    assert policy.device_label == "RTX 2060 · 6.0 GB"
    assert policy.max_inflight_renders == 1
    assert policy.max_pending_renders == 1
    assert policy.max_retained_frames == 2
    assert policy.external_worker_required is True
    assert policy.canonical_geometry_authoritative is True
    assert policy.certified_metrology is False


def test_5g_latest_render_queue_keeps_only_newest_pending_view() -> None:
    queue = LatestRenderRequestQueue()
    dispatch, first = queue.submit(_render(1))
    assert dispatch is True
    assert first and first["request_id"] == 1
    for request_id in range(2, 101):
        dispatch, item = queue.submit(_render(request_id))
        assert dispatch is False
        assert item is None
    snapshot = queue.snapshot
    assert snapshot.render_in_flight is True
    assert snapshot.pending_request_id == 100
    assert snapshot.submitted_count == 100
    assert snapshot.dispatched_count == 1
    assert snapshot.coalesced_count == 99
    pending = queue.complete(1)
    assert pending and pending["request_id"] == 100
    assert queue.snapshot.dispatched_count == 2
    assert queue.snapshot.render_in_flight is True
    assert queue.complete(100) is None
    assert queue.snapshot.completed_count == 2
    assert queue.snapshot.render_in_flight is False


def test_5g_cancel_pending_releases_queue_residency() -> None:
    queue = LatestRenderRequestQueue()
    queue.submit(_render(1))
    queue.submit(_render(2))
    queue.cancel_pending()
    assert queue.snapshot.render_in_flight is False
    assert queue.snapshot.pending_request_id is None


def test_5g_runtime_health_reports_worker_cuda_telemetry_truthfully() -> None:
    policy = build_nerf_interactive_resource_policy(_hardware(gib=6))
    queue = LatestRenderRequestQueue()
    queue.submit(_render(1))
    queue.complete(1)
    health = build_nerf_viewer_runtime_health(
        policy=policy,
        worker_state="ready",
        worker_pid=4321,
        telemetry={
            "total_vram_mib": 5744.5625,
            "cuda_allocated_mib": 820.0,
            "cuda_reserved_mib": 950.0,
            "cuda_peak_allocated_mib": 900.0,
            "retained_frame_count": 2,
        },
        last_render_ms=525.0,
        worker_start_count=1,
        worker_release_count=0,
        queue=queue.snapshot,
        canonical_geometry_authoritative=True,
    )
    assert health.state is NerfViewerResidencyState.RESIDENT
    assert health.resident is True
    assert health.gpu_vram_measured is True
    assert health.process_unreserved_capacity_mib == pytest.approx(4794.5625)
    assert "Worker CUDA reserved 950 MiB" in health.resources_text
    assert "Last 525 ms" in health.performance_text
    payload = health.as_dict()
    assert payload["canonical_geometry_authoritative"] is True
    assert payload["nerf_canonical_geometry_authoritative"] is False
    assert payload["nerf_measurement_authoritative"] is False
    assert payload["certified_metrology"] is False
    assert payload["timings_are_observation_only"] is True


def test_5g_runtime_health_marks_stopped_worker_released_after_real_release_count() -> None:
    policy = build_nerf_interactive_resource_policy(_hardware(gib=6))
    queue = LatestRenderRequestQueue()
    health = build_nerf_viewer_runtime_health(
        policy=policy,
        worker_state="stopped",
        worker_pid=0,
        telemetry={},
        last_render_ms=900.0,
        worker_start_count=2,
        worker_release_count=2,
        queue=queue.snapshot,
        canonical_geometry_authoritative=True,
    )
    assert health.state is NerfViewerResidencyState.RELEASED
    assert health.resident is False
    assert "Released on hide/switch" in health.lifecycle_text


def test_5g_health_rejects_authority_promotion_and_frame_retention_overflow() -> None:
    policy = NerfInteractiveResourcePolicy(device_label="GPU", resource_tier="test")
    queue = LatestRenderRequestQueue().snapshot
    with pytest.raises(ValueError, match="authority"):
        build_nerf_viewer_runtime_health(
            policy=policy,
            worker_state="ready",
            worker_pid=1,
            telemetry={"retained_frame_count": 1},
            last_render_ms=1.0,
            worker_start_count=1,
            worker_release_count=0,
            queue=queue,
            canonical_geometry_authoritative=False,
        )
    health = build_nerf_viewer_runtime_health(
        policy=policy,
        worker_state="ready",
        worker_pid=1,
        telemetry={"retained_frame_count": 2},
        last_render_ms=1.0,
        worker_start_count=1,
        worker_release_count=0,
        queue=queue,
        canonical_geometry_authoritative=True,
    )
    with pytest.raises(ValueError, match="retained frame"):
        replace(health, retained_frame_count=3)
