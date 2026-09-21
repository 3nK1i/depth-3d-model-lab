from __future__ import annotations

import pytest

from core.reconstruction.errors import ReconstructionValidationError
from core.reconstruction.reconstruction_progressive_sfm_contract import (
    I66A_PYCOLMAP_BACKEND_ID,
    ProgressiveSfmSnapshot,
    ProgressiveSfmStage,
    ProgressiveSfmStatistics,
    validate_progressive_sfm_snapshot_transition,
)

SESSION_ID = "11111111-2222-4333-8444-555555555555"
REQUEST_SHA = "b" * 64


def _snapshot(*, revision: int, elapsed_ns: int, timestamp: str, stage=ProgressiveSfmStage.MAPPING, cameras=3, landmarks=10):
    return ProgressiveSfmSnapshot(
        session_id=SESSION_ID,
        revision=revision,
        timestamp_utc=timestamp,
        elapsed_ns=elapsed_ns,
        stage=stage,
        stage_detail="integration",
        backend_id=I66A_PYCOLMAP_BACKEND_ID,
        backend_version="1",
        request_fingerprint_sha256=REQUEST_SHA,
        statistics=ProgressiveSfmStatistics(
            input_views=20,
            registered_cameras=cameras,
            sparse_landmarks=landmarks,
            observations=0 if landmarks == 0 else landmarks * 2,
        ),
        progress_fraction=0.7 if stage is ProgressiveSfmStage.MAPPING else 0.95,
    )


def test_transition_allows_revision_gaps_for_coalescing() -> None:
    previous = _snapshot(revision=1, elapsed_ns=10, timestamp="2026-09-17T12:00:00+00:00")
    current = _snapshot(revision=4, elapsed_ns=20, timestamp="2026-09-17T12:00:01+00:00")
    assert validate_progressive_sfm_snapshot_transition(previous, current) is current


def test_transition_rejects_stale_or_duplicate_revision() -> None:
    previous = _snapshot(revision=4, elapsed_ns=20, timestamp="2026-09-17T12:00:01+00:00")
    for revision in (4, 3):
        with pytest.raises(ReconstructionValidationError):
            validate_progressive_sfm_snapshot_transition(
                previous,
                _snapshot(revision=revision, elapsed_ns=21, timestamp="2026-09-17T12:00:02+00:00"),
            )


def test_transition_rejects_cross_session_or_request_identity() -> None:
    previous = _snapshot(revision=1, elapsed_ns=10, timestamp="2026-09-17T12:00:00+00:00")
    other_session_payload = previous.as_dict()
    other_session_payload.pop("snapshot_sha256", None)
    other_session = ProgressiveSfmSnapshot.from_payload({
        **other_session_payload,
        "session_id": "aaaaaaaa-bbbb-4ccc-8ddd-eeeeeeeeeeee",
        "revision": 2,
        "timestamp_utc": "2026-09-17T12:00:01+00:00",
        "elapsed_ns": 20,
    })
    with pytest.raises(ReconstructionValidationError):
        validate_progressive_sfm_snapshot_transition(previous, other_session)

    other_request_payload = previous.as_dict()
    other_request_payload.pop("snapshot_sha256", None)
    other_request = ProgressiveSfmSnapshot.from_payload({
        **other_request_payload,
        "request_fingerprint_sha256": "c" * 64,
        "revision": 2,
        "timestamp_utc": "2026-09-17T12:00:01+00:00",
        "elapsed_ns": 20,
    })
    with pytest.raises(ReconstructionValidationError):
        validate_progressive_sfm_snapshot_transition(previous, other_request)


def test_transition_allows_geometry_count_decrease_but_not_stage_regression() -> None:
    # Multi-model mapping / BA cleanup may change the current visualization candidate;
    # revision ordering is authoritative, not monotonic geometry counts.
    previous = _snapshot(revision=1, elapsed_ns=10, timestamp="2026-09-17T12:00:00+00:00", cameras=8, landmarks=100)
    current = _snapshot(revision=2, elapsed_ns=20, timestamp="2026-09-17T12:00:01+00:00", cameras=7, landmarks=90)
    assert validate_progressive_sfm_snapshot_transition(previous, current) is current

    finalizing = _snapshot(
        revision=3,
        elapsed_ns=30,
        timestamp="2026-09-17T12:00:02+00:00",
        stage=ProgressiveSfmStage.FINALIZING,
        cameras=7,
        landmarks=90,
    )
    assert validate_progressive_sfm_snapshot_transition(current, finalizing) is finalizing
    with pytest.raises(ReconstructionValidationError):
        validate_progressive_sfm_snapshot_transition(
            finalizing,
            _snapshot(revision=4, elapsed_ns=40, timestamp="2026-09-17T12:00:03+00:00"),
        )
