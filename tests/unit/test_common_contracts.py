from __future__ import annotations

import json
import pickle

import pytest

from core.contracts import FrameReference, ImageSize, ModelProvenance, ProcessingStep
from core.contracts.validation import ContractValidationError


def test_shared_contracts_are_pickle_safe_and_json_compatible() -> None:
    frame = FrameReference(3, 9, 12.5, ImageSize(640, 480))
    provenance = ModelProvenance(
        model_id="test-model",
        display_name="Test Model",
        task="instance_segmentation",
        model_version="1",
        metadata={"threshold": 0.4, "labels": ["part", "tool"]},
    )
    step = ProcessingStep(
        step_id="normalize",
        method="dense-mask-normalization",
        input_ids=("mask-raw",),
        output_ids=("mask-normalized",),
        parameters={"strict": True},
    )

    restored = pickle.loads(pickle.dumps((frame, provenance, step)))
    assert restored == (frame, provenance, step)
    json.dumps(
        {
            "frame": frame.as_dict(),
            "provenance": provenance.as_dict(),
            "step": step.as_dict(),
        }
    )


def test_provenance_rejects_live_or_non_json_resources() -> None:
    with pytest.raises(ContractValidationError, match="JSON-compatible"):
        ModelProvenance(
            model_id="bad",
            display_name="Bad",
            task="detection",
            metadata={"resource": object()},
        )


def test_image_and_frame_identity_validation() -> None:
    with pytest.raises(ValueError, match="positive"):
        ImageSize(0, 10)
    with pytest.raises(ValueError, match="non-negative"):
        FrameReference(-1, 0, 1.0, ImageSize(10, 10))
