from __future__ import annotations

from dataclasses import replace

import pytest

from core.reconstruction.dense_provider_architecture import (
    DenseProviderImplementation,
    DenseProviderProgressStage,
)
from core.reconstruction.errors import ReconstructionValidationError
from core.reconstruction.reconstruction_progressive_dense_contract import (
    ProgressiveDenseSnapshot,
    ProgressiveDenseStage,
    ProgressiveDenseStatistics,
    validate_progressive_dense_snapshot_transition,
)
from core.reconstruction.reconstruction_provider_selection_contract import (
    PRODUCTION_PROVIDER_ID_BY_IMPLEMENTATION,
)


def _snapshot(*, revision: int, stage: ProgressiveDenseStage, points: int, elapsed_ns: int) -> ProgressiveDenseSnapshot:
    implementation = DenseProviderImplementation.COLMAP_MVS
    return ProgressiveDenseSnapshot(
        source_sfm_session_id="11111111-2222-4333-8444-555555555555",
        dense_run_id="aaaaaaaa-bbbb-4ccc-8ddd-eeeeeeeeeeee",
        revision=revision,
        timestamp_utc=f"2026-09-18T09:00:{revision:02d}+00:00",
        elapsed_ns=elapsed_ns,
        provider_id=PRODUCTION_PROVIDER_ID_BY_IMPLEMENTATION[implementation],
        implementation=implementation,
        stage=stage,
        stage_detail="integration",
        request_fingerprint_sha256="a" * 64,
        source_sparse_result_sha256="b" * 64,
        reconstruction_frame_id="pycolmap-reconstruction-frame",
        statistics=ProgressiveDenseStatistics(
            input_views=24,
            selected_views=18,
            processed_views=min(18, revision),
            current_dense_points=points,
        ),
        progress_fraction=min(0.95, revision / 10.0),
    )


def test_progressive_dense_stage_surface_matches_existing_provider_progress_stages() -> None:
    assert {item.value for item in ProgressiveDenseStage} == {item.value for item in DenseProviderProgressStage}


def test_revision_gaps_and_filtering_point_count_reduction_are_allowed() -> None:
    first = _snapshot(revision=1, stage=ProgressiveDenseStage.PROVIDER_EXECUTION, points=10000, elapsed_ns=1)
    second = _snapshot(revision=5, stage=ProgressiveDenseStage.FUSION_FILTERING, points=7000, elapsed_ns=2)
    validate_progressive_dense_snapshot_transition(first, second)


def test_stale_duplicate_and_stage_regression_are_rejected() -> None:
    first = _snapshot(revision=5, stage=ProgressiveDenseStage.ALIGNMENT, points=100, elapsed_ns=5)
    with pytest.raises(ReconstructionValidationError):
        validate_progressive_dense_snapshot_transition(first, replace(first, revision=5))
    with pytest.raises(ReconstructionValidationError):
        validate_progressive_dense_snapshot_transition(
            first,
            _snapshot(revision=6, stage=ProgressiveDenseStage.PROVIDER_EXECUTION, points=100, elapsed_ns=6),
        )


def test_cross_identity_transitions_are_rejected() -> None:
    first = _snapshot(revision=1, stage=ProgressiveDenseStage.PROVIDER_EXECUTION, points=100, elapsed_ns=1)
    second = _snapshot(revision=2, stage=ProgressiveDenseStage.ALIGNMENT, points=100, elapsed_ns=2)
    mutations = (
        {"source_sfm_session_id": "99999999-2222-4333-8444-555555555555"},
        {"dense_run_id": "99999999-bbbb-4ccc-8ddd-eeeeeeeeeeee"},
        {"request_fingerprint_sha256": "c" * 64},
        {"source_sparse_result_sha256": "d" * 64},
        {"reconstruction_frame_id": "other-frame"},
        {
            "implementation": DenseProviderImplementation.DA3,
            "provider_id": PRODUCTION_PROVIDER_ID_BY_IMPLEMENTATION[DenseProviderImplementation.DA3],
        },
    )
    for values in mutations:
        with pytest.raises(ReconstructionValidationError):
            validate_progressive_dense_snapshot_transition(first, replace(second, **values))
