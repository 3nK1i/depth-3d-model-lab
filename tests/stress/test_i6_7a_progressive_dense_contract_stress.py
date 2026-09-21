from __future__ import annotations

from core.reconstruction.dense_provider_architecture import DenseProviderImplementation
from core.reconstruction.reconstruction_progressive_dense_contract import (
    ProgressiveDenseSnapshot,
    ProgressiveDenseStage,
    ProgressiveDenseStatistics,
    validate_progressive_dense_snapshot_transition,
)
from core.reconstruction.reconstruction_provider_selection_contract import PRODUCTION_PROVIDER_ID_BY_IMPLEMENTATION


def _snapshot(revision: int, points: int) -> ProgressiveDenseSnapshot:
    implementation = DenseProviderImplementation.COLMAP_MVS
    return ProgressiveDenseSnapshot(
        source_sfm_session_id="11111111-2222-4333-8444-555555555555",
        dense_run_id="aaaaaaaa-bbbb-4ccc-8ddd-eeeeeeeeeeee",
        revision=revision,
        timestamp_utc="2026-09-18T09:00:00+00:00",
        elapsed_ns=revision,
        provider_id=PRODUCTION_PROVIDER_ID_BY_IMPLEMENTATION[implementation],
        implementation=implementation,
        stage=ProgressiveDenseStage.FUSION_FILTERING,
        stage_detail="stress",
        request_fingerprint_sha256="a" * 64,
        source_sparse_result_sha256="b" * 64,
        reconstruction_frame_id="pycolmap-reconstruction-frame",
        statistics=ProgressiveDenseStatistics(
            input_views=128,
            selected_views=96,
            processed_views=96,
            current_dense_points=points,
        ),
        progress_fraction=0.9,
    )


def test_thousands_of_monotonic_revisions_do_not_require_monotonic_geometry_counts() -> None:
    previous = _snapshot(1, 100000)
    for revision in range(2, 2002):
        points = 100000 + ((revision % 7) - 3) * 500
        current = _snapshot(revision, points)
        validate_progressive_dense_snapshot_transition(previous, current)
        previous = current
