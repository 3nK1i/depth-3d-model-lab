from __future__ import annotations

import json

import numpy as np
import pytest

from core.reconstruction import (
    BackendProvenance,
    CheckpointArtifact,
    CheckpointMetadata,
    IncompatibleCheckpointError,
    ReconstructionMetadataBundle,
    ReconstructionOperationControl,
    ReconstructionProvenance,
    ReconstructionQualityProfile,
    ReconstructionStage,
    ReconstructionTrajectory,
    ReconstructionValidationError,
    SparseLandmark,
    SparseObservation,
    check_resume_compatibility,
    metadata_bundle_from_json,
    metadata_bundle_to_json,
)
from core.version import __version__
from tests.reconstruction_helpers import SESSION_ID, make_camera, make_pose, make_session


def _landmark() -> SparseLandmark:
    observations = (
        SparseObservation("obs-0", "landmark-1", "keyframe-0", (100.0, 200.0), 0, 0.2, 0.9),
        SparseObservation("obs-1", "landmark-1", "keyframe-1", (110.0, 205.0), 1, 0.3, 0.8),
    )
    return SparseLandmark(
        "landmark-1",
        "track-1",
        np.array([0.5, 0.1, 2.0]),
        observations,
        color_rgb=(100, 120, 140),
    )


def _bundle() -> ReconstructionMetadataBundle:
    session = make_session(frame_count=3)
    snapshot = session.snapshot()
    checkpoint = CheckpointMetadata.create(
        session_id=SESSION_ID,
        application_version=__version__,
        session_state=snapshot.state,
        completed_stage=ReconstructionStage.KEYFRAME_SELECTION,
        profile=ReconstructionQualityProfile.BALANCED,
        source_fingerprint="abc123",
        backend_versions=(("mock-sfm", "1.0"),),
        artifacts=(
            CheckpointArtifact(
                "feature_database",
                "checkpoints/features.json",
                "a" * 64,
            ),
        ),
        last_frame_id="frame-2",
        last_keyframe_id="keyframe-1",
    )
    provenance = ReconstructionProvenance(
        application_version=__version__,
        source_fingerprint="abc123",
        hardware_label="Synthetic RTX profile",
        backends=(BackendProvenance("mock-sfm", "1.0", "1"),),
        created_at_utc="2026-08-06T15:00:00+00:00",
        metadata={"test": True},
    )
    return ReconstructionMetadataBundle(
        session=snapshot,
        cameras=(make_camera(),),
        trajectory=ReconstructionTrajectory((make_pose(0), make_pose(1))),
        sparse_landmarks=(_landmark(),),
        checkpoint=checkpoint,
        provenance=provenance,
    )


def test_checkpoint_resume_compatibility() -> None:
    checkpoint = _bundle().checkpoint
    result = check_resume_compatibility(
        checkpoint,
        session_id=SESSION_ID,
        source_fingerprint="abc123",
        available_backend_versions={"mock-sfm": "1.0"},
    )
    assert result.compatible
    result.require_compatible()


def test_checkpoint_rejects_backend_or_source_mismatch() -> None:
    checkpoint = _bundle().checkpoint
    result = check_resume_compatibility(
        checkpoint,
        session_id=SESSION_ID,
        source_fingerprint="different",
        available_backend_versions={"mock-sfm": "2.0"},
    )
    assert not result.compatible
    assert len(result.reasons) == 2
    with pytest.raises(IncompatibleCheckpointError):
        result.require_compatible()


def test_unsupported_checkpoint_schema_version_fails_clearly() -> None:
    checkpoint = _bundle().checkpoint
    payload = checkpoint.as_dict()
    payload["schema_version"] = 99
    with pytest.raises(ReconstructionValidationError, match="Unsupported"):
        CheckpointMetadata(
            checkpoint_id=payload["checkpoint_id"],
            session_id=payload["session_id"],
            application_version=payload["application_version"],
            session_state=checkpoint.session_state,
            completed_stage=checkpoint.completed_stage,
            profile=checkpoint.profile,
            source_fingerprint=payload["source_fingerprint"],
            schema_version=99,
        )


def test_metadata_serialization_round_trip_is_stable() -> None:
    bundle = _bundle()
    first = metadata_bundle_to_json(bundle)
    restored = metadata_bundle_from_json(first)
    second = metadata_bundle_to_json(restored)
    assert json.loads(first) == json.loads(second)
    assert restored.session.session_id == SESSION_ID
    assert restored.trajectory.samples[1].frame_id == "frame-1"
    assert restored.sparse_landmarks[0].track_id == "track-1"


def test_serialization_rejects_unsupported_root_schema() -> None:
    payload = json.loads(metadata_bundle_to_json(_bundle()))
    payload["schema_version"] = 2
    with pytest.raises(ReconstructionValidationError, match="Unsupported"):
        metadata_bundle_from_json(json.dumps(payload))


def test_operation_control_cancellation_is_cooperative() -> None:
    control = ReconstructionOperationControl()
    assert control.request_pause()
    assert control.paused
    assert control.request_resume()
    assert not control.paused
    assert control.request_cancel()
    assert not control.request_cancel()
    with pytest.raises(Exception, match="cancellation"):
        control.checkpoint()


def test_invalid_serialized_enum_is_reported_as_reconstruction_validation_error() -> None:
    payload = json.loads(metadata_bundle_to_json(_bundle()))
    payload["session"]["state"] = "not-a-state"
    with pytest.raises(ReconstructionValidationError, match="invalid value"):
        metadata_bundle_from_json(json.dumps(payload))
