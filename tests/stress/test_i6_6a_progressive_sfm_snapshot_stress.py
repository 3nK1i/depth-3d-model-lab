from __future__ import annotations

from core.reconstruction.reconstruction_progressive_sfm_contract import (
    I66A_PYCOLMAP_BACKEND_ID,
    ProgressiveSfmLandmarkSample,
    ProgressiveSfmGeometryPreview,
    ProgressiveSfmSnapshot,
    ProgressiveSfmStage,
    ProgressiveSfmStatistics,
    validate_progressive_sfm_snapshot_transition,
)

SESSION_ID = "11111111-2222-4333-8444-555555555555"
REQUEST_SHA = "d" * 64


def _snapshot(revision: int) -> ProgressiveSfmSnapshot:
    return ProgressiveSfmSnapshot(
        session_id=SESSION_ID,
        revision=revision,
        timestamp_utc=f"2026-09-17T12:{revision // 60:02d}:{revision % 60:02d}+00:00",
        elapsed_ns=revision * 1_000_000,
        stage=ProgressiveSfmStage.MAPPING,
        stage_detail="stress",
        backend_id=I66A_PYCOLMAP_BACKEND_ID,
        backend_version="1",
        request_fingerprint_sha256=REQUEST_SHA,
        statistics=ProgressiveSfmStatistics(
            input_views=2000,
            registered_cameras=min(2000, revision + 1),
            sparse_landmarks=revision * 10,
            observations=revision * 20,
        ),
        progress_fraction=min(0.99, 0.64 + revision / 10_000.0),
    )


def test_many_monotonic_revisions_validate_without_identity_collision() -> None:
    previous = _snapshot(1)
    digests = {previous.snapshot_sha256}
    for revision in range(2, 1001):
        current = _snapshot(revision)
        validate_progressive_sfm_snapshot_transition(previous, current)
        digests.add(current.snapshot_sha256)
        previous = current
    assert len(digests) == 1000


def test_large_landmark_preview_serialization_is_deterministic() -> None:
    points = tuple(
        ProgressiveSfmLandmarkSample(
            backend_point3d_id=index,
            position_reconstruction=(float(index), float(index % 17), float(index % 31)),
            color_rgb=(index % 256, (index * 3) % 256, (index * 7) % 256),
            track_length=2 + (index % 5),
        )
        for index in range(5000)
    )
    preview = ProgressiveSfmGeometryPreview(landmark_samples=points, sampling_profile="stress_bounded_preview")
    snapshot = ProgressiveSfmSnapshot(
        session_id=SESSION_ID,
        revision=1,
        timestamp_utc="2026-09-17T12:00:00+00:00",
        elapsed_ns=1,
        stage=ProgressiveSfmStage.MAPPING,
        stage_detail="stress-large-preview",
        backend_id=I66A_PYCOLMAP_BACKEND_ID,
        backend_version="1",
        request_fingerprint_sha256=REQUEST_SHA,
        statistics=ProgressiveSfmStatistics(
            input_views=100,
            registered_cameras=20,
            sparse_landmarks=10_000,
            observations=25_000,
        ),
        progress_fraction=0.8,
        geometry=preview,
    )
    assert snapshot.snapshot_sha256 == ProgressiveSfmSnapshot.from_payload(snapshot.as_dict()).snapshot_sha256
