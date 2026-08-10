from __future__ import annotations

from types import SimpleNamespace

from app.reconstruction_diagnostics import LiveTrackingDiagnosticsAccumulator


def _observation(
    *,
    state: str = "tracking",
    latency_ms: float = 15.0,
    revision: int = 3,
    keyframes: int = 7,
    landmarks: int = 420,
    tracked: int = 0,
    inliers: int = 0,
    ratio: float | None = None,
    reason: str = "stella_pose_valid",
):
    return SimpleNamespace(
        tracking_state=SimpleNamespace(value=state),
        quality=SimpleNamespace(
            processing_latency_ns=int(latency_ms * 1_000_000),
            tracked_features=tracked,
            inlier_features=inliers,
            inlier_ratio=ratio,
            reason=reason,
        ),
        map_snapshot=SimpleNamespace(
            revision=revision,
            keyframe_count=keyframes,
            landmark_count=landmarks,
        ),
    )


def test_c5_tracking_diagnostics_use_real_latency_and_map_counts_without_fake_feature_score() -> None:
    accumulator = LiveTrackingDiagnosticsAccumulator()
    first = accumulator.update(_observation())
    second = accumulator.update(
        _observation(latency_ms=18.0, revision=4, keyframes=8, landmarks=455)
    )

    assert first.tracking_label == "TRACKING"
    assert first.status_kind == "ready"
    assert first.latest_latency_ms == 15.0
    assert first.map_text == "r3  •  7 KF  •  420 LM"
    assert first.map_growth_text == "—"
    assert first.feature_quality_available is False
    assert first.feature_text == "Not reported by Stella"

    assert second.latest_latency_ms == 18.0
    assert second.median_latency_ms == 16.5
    assert second.p95_latency_ms is not None and 17.0 < second.p95_latency_ms <= 18.0
    assert second.map_text == "r4  •  8 KF  •  455 LM"
    assert second.map_growth_text == "+1 KF  •  +35 LM"


def test_c5_feature_diagnostics_are_only_present_when_native_counts_exist() -> None:
    accumulator = LiveTrackingDiagnosticsAccumulator()
    presentation = accumulator.update(
        _observation(tracked=120, inliers=84, ratio=0.7)
    )
    assert presentation.feature_quality_available is True
    assert presentation.feature_text == "84/120  •  70% inliers"
    assert presentation.inlier_ratio == 0.7


def test_c5_tracking_state_presentation_preserves_lost_and_recovering_authority() -> None:
    accumulator = LiveTrackingDiagnosticsAccumulator()
    lost = accumulator.update(_observation(state="lost", reason="stella_tracking_lost"))
    assert lost.tracking_label == "LOST"
    assert lost.status_kind == "error"
    assert lost.reason == "stella_tracking_lost"

    recovering = accumulator.update(
        _observation(state="recovering", revision=4, reason="stella_tracking_recovered")
    )
    assert recovering.tracking_label == "RECOVERING"
    assert recovering.status_kind == "warning"


def test_c5_latency_history_is_bounded_and_terminal_presentation_is_explicit() -> None:
    accumulator = LiveTrackingDiagnosticsAccumulator()
    for index in range(200):
        accumulator.update(_observation(latency_ms=10.0 + index * 0.01))
    assert accumulator.latency_sample_count == accumulator.MAX_LATENCY_SAMPLES == 120
    terminal = accumulator.mark_terminal(True)
    assert terminal.terminal is True
    assert terminal.display_tracking_label == "LAST • TRACKING"

    reset = accumulator.reset_session()
    assert accumulator.latency_sample_count == 0
    assert reset.tracking_label == "WAITING"
    assert reset.map_text == "No live map yet"
