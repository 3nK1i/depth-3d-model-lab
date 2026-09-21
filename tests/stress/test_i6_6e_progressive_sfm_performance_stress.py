from __future__ import annotations

from app.reconstruction_progressive_sfm_ux import ProgressiveSfmViewerPerformance


def test_i6_6e_performance_accumulator_stays_constant_space_over_many_updates() -> None:
    perf = ProgressiveSfmViewerPerformance()
    for revision in range(1, 10001):
        perf = perf.record_success(
            update_ms=float(revision % 17),
            payload_bytes=1000 + (revision % 1024),
        )
    assert perf.applied_updates == 10000
    assert perf.max_update_ms == 16.0
    assert perf.max_payload_bytes <= 2023
    assert not hasattr(perf, "history")


def test_i6_6e_rejections_are_counted_without_changing_success_metrics() -> None:
    perf = ProgressiveSfmViewerPerformance().record_success(update_ms=3.0, payload_bytes=2048)
    for _ in range(5000):
        perf = perf.record_rejection()
    assert perf.applied_updates == 1
    assert perf.rejected_updates == 5000
    assert perf.last_update_ms == 3.0
    assert perf.last_payload_bytes == 2048
