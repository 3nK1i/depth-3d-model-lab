from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pytest

from core.reconstruction import (
    ReconstructionEventRecorder,
    ReconstructionEventType,
    ReconstructionStage,
)
from tests.reconstruction_helpers import SESSION_ID


ROOT = Path(__file__).resolve().parents[2]
WORKER_PATH = ROOT / "scripts" / "reconstruction_pycolmap_worker.py"


def _load_worker_module():
    spec = importlib.util.spec_from_file_location("phase9_6e21_worker_test_module", WORKER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _record_progress(recorder: ReconstructionEventRecorder, fraction: float) -> None:
    recorder.emit(
        event_type=ReconstructionEventType.PROGRESS,
        stage=ReconstructionStage.SPARSE_RECONSTRUCTION,
        message="Synthetic PyCOLMAP mapping progress.",
        elapsed_ns=1,
        progress_fraction=fraction,
    )


def test_multimodel_mapping_seed_callbacks_do_not_restart_progress() -> None:
    worker = _load_worker_module()
    tracker = worker._MappingProgressTracker(12)
    recorder = ReconstructionEventRecorder(SESSION_ID)

    _record_progress(recorder, 0.64)
    event_type, first_fraction, _ = tracker.initial_pair()
    assert event_type == "progress"
    assert first_fraction == pytest.approx(0.68)
    _record_progress(recorder, first_fraction)

    advanced = [tracker.next_image() for _ in range(6)]
    for fraction in advanced:
        _record_progress(recorder, fraction)
    assert advanced[-1] > 0.68

    event_type, second_fraction, message = tracker.initial_pair()
    assert event_type == "status"
    assert second_fraction is None
    assert "Additional sparse-model" in message

    continued = tracker.next_image()
    assert continued >= advanced[-1]
    _record_progress(recorder, continued)

    fractions = [
        event.progress_fraction
        for event in recorder.events
        if event.progress_fraction is not None
    ]
    assert fractions == sorted(fractions)


def test_mapping_progress_is_bounded_and_never_decreases_across_many_models() -> None:
    worker = _load_worker_module()
    tracker = worker._MappingProgressTracker(5)
    fractions = [0.64]

    for model_index in range(4):
        event_type, fraction, _ = tracker.initial_pair()
        if fraction is not None:
            assert event_type == "progress"
            fractions.append(fraction)
        else:
            assert model_index > 0
            assert event_type == "status"
        fractions.extend(tracker.next_image() for _ in range(5))

    assert fractions == sorted(fractions)
    assert fractions[-1] == pytest.approx(0.92)
    assert all(0.0 <= value <= 1.0 for value in fractions)


def test_mapping_progress_tracker_rejects_invalid_image_count() -> None:
    worker = _load_worker_module()
    with pytest.raises(ValueError, match="at least one image"):
        worker._MappingProgressTracker(0)
