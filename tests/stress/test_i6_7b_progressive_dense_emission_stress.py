from __future__ import annotations

from pathlib import Path

import numpy as np

from core.reconstruction.dense_provider_architecture import (
    DenseProviderImplementation,
    DenseProviderProgressEvent,
    DenseProviderProgressStage,
)
from core.reconstruction.reconstruction_progressive_dense_emission import (
    ProgressiveDenseEmissionPolicy,
    ProgressiveDenseEmissionSession,
    load_dense_geometry_preview,
)
from tests.i4_3b_helpers import explicit_selection, make_sfm_result


class Sink:
    def __init__(self) -> None:
        self.snapshots = []

    def publish(self, snapshot) -> None:
        self.snapshots.append(snapshot)


def test_i67b_ten_thousand_status_events_remain_bounded_without_history_growth(tmp_path: Path) -> None:
    sfm = make_sfm_result(tmp_path, view_count=3)
    request, decision = explicit_selection(3)
    sink = Sink()
    session = ProgressiveDenseEmissionSession(
        source_sfm_result=sfm,
        request=request,
        decision=decision,
        reconstruction_frame_id="sfm-frame-1",
        snapshot_sink=sink,
        dense_run_id="aaaaaaaa-bbbb-4ccc-8ddd-eeeeeeeeeeee",
        policy=ProgressiveDenseEmissionPolicy(target_status_snapshots=24),
    )
    for i in range(10_000):
        session.publish(
            DenseProviderProgressEvent(
                provider_id="colmap-mvs-external",
                implementation=DenseProviderImplementation.COLMAP_MVS,
                stage=DenseProviderProgressStage.PROVIDER_EXECUTION,
                stage_detail="stress",
                progress_fraction=0.10 + 0.60 * (i / 9_999.0),
                processed_views=min(3, int(3 * i / 9_999.0)),
                total_views=3,
            )
        )
    assert len(sink.snapshots) <= 26
    assert session.dropped_status_snapshot_count >= 9_970
    assert len(session.failure_records) == 0


def test_i67b_hundred_thousand_point_source_emits_only_bounded_preview(tmp_path: Path) -> None:
    count = 100_000
    points = np.column_stack((np.arange(count), np.ones(count), np.zeros(count))).astype(np.float32)
    colors = np.zeros((count, 3), dtype=np.uint8)
    path = tmp_path / "large.npz"
    np.savez_compressed(path, points_reconstruction=points, colors_rgb=colors)
    loaded = load_dense_geometry_preview(path, max_preview_points=4096)
    assert loaded.point_count == count
    assert len(loaded.preview.point_samples) == 4096
    assert loaded.preview.point_samples[0].sample_index == 0
    assert loaded.preview.point_samples[-1].sample_index == count - 1
