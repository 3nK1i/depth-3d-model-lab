from __future__ import annotations
import pickle
import numpy as np
import pytest
from core.contracts.common import FrameReference, ImageSize, ModelProvenance
from core.contracts.validation import ContractValidationError
from core.models.metadata import ConfidenceKind
from core.perception.raw import RawPerceptionBatch, RawPerceptionInstance, RawPromptedMask

def frame():
    return FrameReference(1, 2, 3.0, ImageSize(10, 8))

def prov():
    return ModelProvenance('m', 'M', 'instance_segmentation')

def test_raw_instances_are_pickle_safe_and_arrays_readonly():
    mask = np.ones((8, 10), dtype=bool)
    item = RawPerceptionInstance('i', 0, 'part', 0.8, (0, 0, 10, 8), mask, 0.7, ConfidenceKind.CONFIDENCE)
    batch = RawPerceptionBatch(frame(), 'rgb', (item,), prov(), 1.2)
    restored = pickle.loads(pickle.dumps(batch))
    assert restored.instances[0].mask.shape == (8, 10)
    assert not restored.instances[0].mask.flags.writeable

def test_raw_ids_unique_and_confidence_semantics_enforced():
    item = RawPerceptionInstance('i', 0, 'part', 0.8, (0, 0, 10, 8))
    with pytest.raises(ContractValidationError, match='unique'):
        RawPerceptionBatch(frame(), 'rgb', (item, item), prov())
    with pytest.raises(ContractValidationError, match='must be None'):
        RawPerceptionInstance('i', 0, 'part', 0.8, (0, 0, 10, 8), mask_confidence=0.4)

def test_prompted_raw_requires_parent():
    with pytest.raises(ContractValidationError, match='parent'):
        RawPromptedMask(frame(), '', '', np.ones((8, 10), bool), prov())


def test_prompted_raw_uncertainty_must_be_non_negative() -> None:
    with pytest.raises(ContractValidationError, match="must not be negative"):
        RawPromptedMask(
            frame(),
            "det",
            "",
            np.ones((8, 10), bool),
            prov(),
            -0.1,
            ConfidenceKind.UNCERTAINTY,
        )

    value = RawPromptedMask(
        frame(),
        "det",
        "",
        np.ones((8, 10), bool),
        prov(),
        0.2,
        ConfidenceKind.UNCERTAINTY,
    )
    assert value.confidence == 0.2
