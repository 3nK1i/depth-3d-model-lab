from __future__ import annotations

from app.reconstruction_nerf_resources import LatestRenderRequestQueue


def _render(request_id: int) -> dict[str, object]:
    return {"type": "render", "request_id": request_id}


def test_5g_ten_thousand_navigation_updates_remain_one_inflight_one_latest_pending() -> None:
    queue = LatestRenderRequestQueue()
    dispatch, _ = queue.submit(_render(1))
    assert dispatch is True
    for request_id in range(2, 10_001):
        dispatch, _ = queue.submit(_render(request_id))
        assert dispatch is False
        snap = queue.snapshot
        assert snap.render_in_flight is True
        assert snap.pending_request_id == request_id
    snap = queue.snapshot
    assert snap.submitted_count == 10_000
    assert snap.dispatched_count == 1
    assert snap.coalesced_count == 9_999
    pending = queue.complete(1)
    assert pending and pending["request_id"] == 10_000
    assert queue.snapshot.dispatched_count == 2
    queue.complete(10_000)
    assert queue.snapshot.completed_count == 2
    assert queue.snapshot.render_in_flight is False
