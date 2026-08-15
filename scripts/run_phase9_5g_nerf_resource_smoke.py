#!/usr/bin/env python3
"""Dependency-light Phase 9.5G NeRF resource/lifecycle smoke."""

from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.reconstruction_nerf_resources import (
    LatestRenderRequestQueue,
    NerfInteractiveResourcePolicy,
    build_nerf_viewer_runtime_health,
)


def main() -> int:
    queue = LatestRenderRequestQueue()
    first = {"type": "render", "request_id": 1}
    latest = {"type": "render", "request_id": 5}
    dispatched, _ = queue.submit(first)
    for request_id in range(2, 6):
        queue.submit({"type": "render", "request_id": request_id})
    pending = queue.complete(1)
    queue.complete(5)
    policy = NerfInteractiveResourcePolicy(
        device_label="RTX 2060 · 6.0 GB",
        resource_tier="conservative-6gb",
    )
    health = build_nerf_viewer_runtime_health(
        policy=policy,
        worker_state="ready",
        worker_pid=12345,
        telemetry={
            "total_vram_mib": 5744.5625,
            "cuda_allocated_mib": 800.0,
            "cuda_reserved_mib": 960.0,
            "cuda_peak_allocated_mib": 900.0,
            "retained_frame_count": 2,
        },
        last_render_ms=525.0,
        worker_start_count=1,
        worker_release_count=0,
        queue=queue.snapshot,
        canonical_geometry_authoritative=True,
    )
    passed = (
        dispatched
        and pending == latest
        and queue.snapshot.submitted_count == 5
        and queue.snapshot.dispatched_count == 2
        and queue.snapshot.coalesced_count == 4
        and queue.snapshot.completed_count == 2
        and not queue.snapshot.render_in_flight
        and health.resident
        and health.gpu_vram_measured
        and health.retained_frame_count == 2
        and not health.nerf_canonical_geometry_authoritative
        and not health.nerf_metric_scale_authoritative
        and not health.nerf_measurement_authoritative
        and not health.nerf_qc_authoritative
        and not health.certified_metrology
    )
    summary = {
        "phase": "9.5G",
        "status": "PASS" if passed else "FAIL",
        "resource_tier": policy.resource_tier,
        "max_inflight_renders": policy.max_inflight_renders,
        "max_pending_renders": policy.max_pending_renders,
        "max_retained_frames": policy.max_retained_frames,
        "submitted_render_count": queue.snapshot.submitted_count,
        "dispatched_render_count": queue.snapshot.dispatched_count,
        "coalesced_render_count": queue.snapshot.coalesced_count,
        "completed_render_count": queue.snapshot.completed_count,
        "queue_bounded": queue.snapshot.bounded,
        "gpu_vram_measured": health.gpu_vram_measured,
        "cuda_allocated_mib": health.cuda_allocated_mib,
        "cuda_reserved_mib": health.cuda_reserved_mib,
        "retained_frame_count": health.retained_frame_count,
        "canonical_geometry_authoritative": True,
        "nerf_canonical_geometry_authoritative": False,
        "nerf_metric_scale_authoritative": False,
        "nerf_measurement_authoritative": False,
        "nerf_qc_authoritative": False,
        "certified_metrology": False,
        "backend_started": False,
        "training_started": False,
        "production_heavy_modules": [],
        "timings_are_observation_only": True,
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
