from __future__ import annotations

from app.reconstruction_nerf_resources import LatestRenderRequestQueue


def test_5h_rapid_navigation_never_grows_beyond_one_pending_latest_view() -> None:
    queue = LatestRenderRequestQueue()
    dispatched, payload = queue.submit({"type": "render", "request_id": 0})
    assert dispatched is True and payload is not None
    for request_id in range(1, 50001):
        dispatched, payload = queue.submit({"type": "render", "request_id": request_id})
        assert dispatched is False and payload is None
        snapshot = queue.snapshot
        assert snapshot.render_in_flight is True
        assert snapshot.pending_request_id == request_id
    pending = queue.complete(0)
    assert pending is not None
    assert pending["request_id"] == 50000
    assert queue.snapshot.dispatched_count == 2
    assert queue.snapshot.coalesced_count == 50000
