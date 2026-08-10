from __future__ import annotations

import numpy as np
import pytest

from core.reconstruction import LiveMapGeometrySnapshot, LiveMapKeyframeSample
from core.reconstruction.errors import ReconstructionValidationError


def _sample(index: int, x: float) -> LiveMapKeyframeSample:
    pose = np.eye(4, dtype=np.float64)
    pose[0, 3] = x
    return LiveMapKeyframeSample(
        keyframe_id=f"stella-keyframe-{index}",
        timestamp_ns=1_000_000_000 + index,
        T_reconstruction_from_camera=pose,
    )


def test_c4_keyframe_sample_is_arbitrary_frame_pose_without_metric_fields() -> None:
    sample = _sample(3, 0.25)
    payload = sample.as_dict()
    assert payload["keyframe_id"] == "stella-keyframe-3"
    assert payload["pose_convention"] == "T_reconstruction_from_camera"
    assert np.allclose(payload["T_reconstruction_from_camera"], sample.T_reconstruction_from_camera)
    assert "unit_to_metre" not in payload
    assert "metric" not in payload


def test_c4_geometry_snapshot_extends_c3_without_rewriting_base_counts() -> None:
    samples = (_sample(0, 0.0), _sample(2, 0.2))
    snapshot = LiveMapGeometrySnapshot(
        revision=4,
        keyframe_count=3,
        landmark_count=10,
        landmarks_reconstruction=((0.0, 0.0, 1.0),),
        keyframes_reconstruction=samples,
    )
    assert snapshot.keyframe_count == 3
    assert snapshot.landmark_count == 10
    assert snapshot.keyframes_reconstruction == samples
    payload = snapshot.as_dict()
    assert len(payload["keyframes_reconstruction"]) == 2
    assert payload["landmarks_reconstruction"] == [[0.0, 0.0, 1.0]]


def test_c4_geometry_snapshot_rejects_duplicate_or_excess_keyframes() -> None:
    with pytest.raises(ReconstructionValidationError, match="Duplicate"):
        LiveMapGeometrySnapshot(
            revision=1,
            keyframe_count=2,
            landmark_count=0,
            keyframes_reconstruction=(_sample(1, 0.0), _sample(1, 0.1)),
        )
    with pytest.raises(ReconstructionValidationError, match="cannot exceed"):
        LiveMapGeometrySnapshot(
            revision=1,
            keyframe_count=1,
            landmark_count=0,
            keyframes_reconstruction=(_sample(1, 0.0), _sample(2, 0.1)),
        )
