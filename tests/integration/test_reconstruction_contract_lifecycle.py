from __future__ import annotations

import json

import numpy as np

from core.reconstruction import (
    BackendProvenance,
    CheckpointMetadata,
    ReconstructionEventType,
    ReconstructionMetadataBundle,
    ReconstructionProvenance,
    ReconstructionQualityProfile,
    ReconstructionStage,
    ReconstructionTrajectory,
    SparseLandmark,
    SparseObservation,
    metadata_bundle_from_json,
    metadata_bundle_to_json,
)
from core.version import __version__
from tests.reconstruction_helpers import SESSION_ID, make_camera, make_pose, make_session


def test_contract_lifecycle_from_session_to_checkpoint_restore() -> None:
    session = make_session(frame_count=3)
    session.start()
    session.events.emit(
        event_type=ReconstructionEventType.PROGRESS,
        stage=ReconstructionStage.KEYFRAME_SELECTION,
        message="Keyframes selected",
        elapsed_ns=10,
        progress_fraction=1.0,
    )
    session.pause()
    snapshot = session.snapshot()
    checkpoint = CheckpointMetadata.create(
        session_id=SESSION_ID,
        application_version=__version__,
        session_state=snapshot.state,
        completed_stage=ReconstructionStage.KEYFRAME_SELECTION,
        profile=ReconstructionQualityProfile.BALANCED,
        source_fingerprint="abc123",
        last_frame_id="frame-2",
        last_keyframe_id="keyframe-1",
    )
    landmark = SparseLandmark(
        "landmark-1",
        "track-1",
        np.array([0.0, 0.0, 2.0]),
        (
            SparseObservation("obs-0", "landmark-1", "keyframe-0", (100.0, 100.0)),
            SparseObservation("obs-1", "landmark-1", "keyframe-1", (110.0, 100.0)),
        ),
    )
    bundle = ReconstructionMetadataBundle(
        session=snapshot,
        cameras=(make_camera(),),
        trajectory=ReconstructionTrajectory((make_pose(0), make_pose(1))),
        sparse_landmarks=(landmark,),
        checkpoint=checkpoint,
        provenance=ReconstructionProvenance(
            application_version=__version__,
            source_fingerprint="abc123",
            hardware_label="integration-test",
            backends=(BackendProvenance("mock", "1"),),
            created_at_utc="2026-08-06T15:00:00+00:00",
        ),
    )
    payload = metadata_bundle_to_json(bundle)
    restored = metadata_bundle_from_json(payload)
    assert restored.session.state.value == "paused"
    assert restored.checkpoint.last_keyframe_id == "keyframe-1"
    assert json.loads(metadata_bundle_to_json(restored)) == json.loads(payload)


def test_running_session_cancel_flow_reaches_terminal_state() -> None:
    session = make_session()
    session.start()
    session.request_cancel()
    assert session.state.value == "cancelling"
    session.acknowledge_cancelled()
    assert session.state.value == "cancelled"
    assert session.control.cancellation_requested
