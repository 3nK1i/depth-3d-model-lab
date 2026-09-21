from __future__ import annotations

from datetime import datetime, timedelta, timezone

from core.reconstruction.reconstruction_progressive_sfm_contract import (
    I66A_PYCOLMAP_BACKEND_ID,
    ProgressiveSfmSnapshot,
    ProgressiveSfmStage,
    ProgressiveSfmStatistics,
)

SESSION_ID = "11111111-2222-4333-8444-555555555555"
REQUEST_SHA = "d" * 64


def make_snapshot(
    *,
    revision: int = 1,
    session_id: str = SESSION_ID,
    request_sha: str = REQUEST_SHA,
    stage: ProgressiveSfmStage = ProgressiveSfmStage.MAPPING,
    elapsed_ns: int | None = None,
    timestamp_utc: str | None = None,
    registered_cameras: int = 3,
) -> ProgressiveSfmSnapshot:
    elapsed = revision * 1_000_000 if elapsed_ns is None else elapsed_ns
    timestamp = timestamp_utc or (
        datetime(2026, 9, 17, 12, 0, 0, tzinfo=timezone.utc)
        + timedelta(milliseconds=revision)
    ).isoformat()
    return ProgressiveSfmSnapshot(
        session_id=session_id,
        revision=revision,
        timestamp_utc=timestamp,
        elapsed_ns=elapsed,
        stage=stage,
        stage_detail=(
            "selected_model_before_final_result_admission"
            if stage is ProgressiveSfmStage.FINALIZING
            else "registered_image_snapshot"
        ),
        backend_id=I66A_PYCOLMAP_BACKEND_ID,
        backend_version="1",
        request_fingerprint_sha256=request_sha,
        statistics=ProgressiveSfmStatistics(
            input_views=24,
            registered_cameras=registered_cameras,
            sparse_landmarks=100,
            observations=300,
            mean_reprojection_error_px=0.6,
        ),
        progress_fraction=(0.94 if stage is ProgressiveSfmStage.FINALIZING else 0.7),
    )
