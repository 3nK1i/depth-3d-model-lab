from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta, timezone

import numpy as np

from core.reconstruction.dense_provider_architecture import DenseProviderImplementation
from core.reconstruction.reconstruction_provider_selection_contract import (
    PRODUCTION_PROVIDER_ID_BY_IMPLEMENTATION,
)
from core.reconstruction.reconstruction_progressive_dense_contract import (
    ProgressiveDenseGeometryPreview,
    ProgressiveDensePointSample,
    ProgressiveDenseSnapshot,
    ProgressiveDenseStage,
    ProgressiveDenseStatistics,
)

SESSION_ID = "11111111-2222-4333-8444-555555555555"
DENSE_RUN_ID = "aaaaaaaa-bbbb-4ccc-8ddd-eeeeeeeeeeee"
REQUEST_SHA = "a" * 64
SPARSE_SHA = "b" * 64
FRAME_ID = "pycolmap-reconstruction-frame"


def make_snapshot(
    *,
    revision: int = 1,
    stage: ProgressiveDenseStage = ProgressiveDenseStage.PROVIDER_EXECUTION,
    geometry_points: int = 0,
    dense_points: int = 100,
    warnings: tuple[str, ...] = (),
    dense_run_id: str = DENSE_RUN_ID,
    session_id: str = SESSION_ID,
    request_sha: str = REQUEST_SHA,
    sparse_sha: str = SPARSE_SHA,
    frame_id: str = FRAME_ID,
    implementation: DenseProviderImplementation = DenseProviderImplementation.COLMAP_MVS,
) -> ProgressiveDenseSnapshot:
    geometry = None
    if geometry_points:
        geometry = ProgressiveDenseGeometryPreview(
            point_samples=tuple(
                ProgressiveDensePointSample(
                    sample_index=index,
                    position_reconstruction=np.asarray(
                        [float(index), float(index) + 0.25, float(index) - 0.5]
                    ),
                    color_rgb=(index % 256, (index * 3) % 256, (index * 7) % 256),
                    confidence=0.75,
                    view_support=3,
                )
                for index in range(geometry_points)
            )
        )
        dense_points = max(dense_points, geometry_points)
    timestamp = datetime(2026, 9, 18, 9, 0, tzinfo=timezone.utc) + timedelta(
        milliseconds=revision
    )
    return ProgressiveDenseSnapshot(
        source_sfm_session_id=session_id,
        dense_run_id=dense_run_id,
        revision=revision,
        timestamp_utc=timestamp.isoformat(),
        elapsed_ns=revision * 1_000_000,
        provider_id=PRODUCTION_PROVIDER_ID_BY_IMPLEMENTATION[implementation],
        implementation=implementation,
        stage=stage,
        stage_detail=f"i6_7c_probe_revision_{revision}",
        request_fingerprint_sha256=request_sha,
        source_sparse_result_sha256=sparse_sha,
        reconstruction_frame_id=frame_id,
        statistics=ProgressiveDenseStatistics(
            input_views=24,
            selected_views=24,
            processed_views=min(revision, 24),
            current_dense_points=dense_points,
            mean_confidence=0.75,
            mean_view_support=3.0,
        ),
        progress_fraction=min(0.95, revision / 100.0),
        geometry=geometry,
        warnings=warnings,
    )


def replace_snapshot(snapshot: ProgressiveDenseSnapshot, **changes) -> ProgressiveDenseSnapshot:
    return replace(snapshot, **changes)
