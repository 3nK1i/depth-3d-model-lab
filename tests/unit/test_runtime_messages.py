"""IPC contract tests: messages must stay immutable and spawn-picklable."""

from __future__ import annotations

import pickle

import numpy as np
import pytest

from core.runtime.messages import (
    Ack,
    Command,
    CommandType,
    FrameRequest,
    InferenceResult,
    ModelStage,
    ModelStageEvent,
)


def test_frame_request_pickle_round_trip_preserves_array_and_metadata() -> None:
    image = np.arange(3 * 4 * 3, dtype=np.uint8).reshape(3, 4, 3)
    request = FrameRequest(
        8,
        12,
        image,
        point_budget=100_000,
        submitted_at=1.25,
        source_timestamp=1.0,
    )

    restored = pickle.loads(pickle.dumps(request))

    assert restored.session_id == 8
    assert restored.frame_id == 12
    assert restored.point_budget == 100_000
    assert restored.submitted_at == 1.25
    np.testing.assert_array_equal(restored.image, image)


def test_result_pickle_round_trip_preserves_optional_geometry() -> None:
    depth = np.ones((2, 3), dtype=np.float32)
    points = np.ones((2, 3, 3), dtype=np.float32)
    result = InferenceResult(
        session_id=2,
        frame_id=4,
        model_id="moge-2",
        depth=depth,
        latency_ms=12.5,
        points=points,
        precision="FP16",
        vram_reserved_gib=0.75,
    )

    restored = pickle.loads(pickle.dumps(result))
    np.testing.assert_array_equal(restored.depth, depth)
    np.testing.assert_array_equal(restored.points, points)
    assert restored.precision == "FP16"
    assert restored.vram_reserved_gib == 0.75


def test_control_messages_are_frozen() -> None:
    command = Command(CommandType.LOAD_MODEL, model_id="moge-2")
    ack = Ack(CommandType.LOAD_MODEL, True)

    with pytest.raises((AttributeError, TypeError)):
        command.model_id = "other"  # type: ignore[misc]
    with pytest.raises((AttributeError, TypeError)):
        ack.success = False  # type: ignore[misc]


def test_model_stage_event_is_picklable_and_frozen() -> None:
    event = ModelStageEvent(
        stage=ModelStage.WARMING_UP,
        model_id="moge-2",
        message="Priming kernels",
    )
    restored = pickle.loads(pickle.dumps(event))
    assert restored.stage is ModelStage.WARMING_UP
    assert restored.model_id == "moge-2"
    with pytest.raises((AttributeError, TypeError)):
        event.message = "changed"  # type: ignore[misc]
