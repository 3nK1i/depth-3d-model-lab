"""Worker registry and latest-result queue helper tests."""

from __future__ import annotations

import queue

import pytest

from core.inference.worker import _build_model, _drain, _put_latest, _reset_model_sequence
from core.models.adapters.mock_adapter import MockGeometryModel
from core.models.catalog import MOCK_MODEL_ID
from core.runtime.messages import InferenceResult


def _result(frame_id: int) -> InferenceResult:
    return InferenceResult(1, frame_id, MOCK_MODEL_ID, None, 0.0)


def test_worker_builds_registered_mock_and_rejects_unknown_model() -> None:
    assert isinstance(_build_model(MOCK_MODEL_ID), MockGeometryModel)
    with pytest.raises(ValueError, match="Unknown model id"):
        _build_model("does-not-exist")


def test_put_latest_replaces_old_result() -> None:
    results = queue.Queue(maxsize=1)
    results.put(_result(1))
    _put_latest(results, _result(2))
    assert results.get_nowait().frame_id == 2


def test_drain_empties_queue_without_blocking() -> None:
    values = queue.Queue()
    values.put(1)
    values.put(2)
    _drain(values)
    assert values.empty()
    _drain(values)


class _StatefulModel:
    def reset_sequence(self):
        return {"sequence_reset": True}


def test_reset_model_sequence_is_generic_and_metadata_safe() -> None:
    assert _reset_model_sequence(_StatefulModel()) == {"sequence_reset": True}
    assert _reset_model_sequence(MockGeometryModel()) == {}
