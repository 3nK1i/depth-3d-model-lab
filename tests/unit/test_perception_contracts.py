from __future__ import annotations

import pickle

import numpy as np
import pytest

from core.contracts import FrameReference, ImageSize, ModelProvenance, PixelCoordinateSpace, PixelSpaceRole
from core.contracts.validation import ContractValidationError
from core.models.metadata import ConfidenceKind
from core.perception import (
    ApprovalStatus,
    DenseMaskPayload,
    Detection2D,
    InstanceMask,
    MaskRepresentation,
    MaskSource,
    ObjectObservation,
    PolygonMaskPayload,
    RLEMaskPayload,
    SharedMemoryMaskPayload,
    UnsupportedMaskRepresentationError,
    normalize_detection_box,
    normalize_mask_payload,
    select_authoritative_mask,
)


def _frame() -> FrameReference:
    return FrameReference(7, 11, 2.5, ImageSize(10, 8))


def _space() -> PixelCoordinateSpace:
    return PixelCoordinateSpace("mask-space", _frame(), ImageSize(10, 8), PixelSpaceRole.MASK)


def _provenance() -> ModelProvenance:
    return ModelProvenance("seg-test", "Seg Test", "instance_segmentation")


def _dense(mask_id: str, source: MaskSource, **overrides) -> InstanceMask:
    values = dict(
        mask_id=mask_id,
        frame_reference=_frame(),
        pixel_space=_space(),
        payload=DenseMaskPayload(np.ones((8, 10), dtype=bool)),
        source=source,
        model_provenance=_provenance(),
        confidence=0.8,
        confidence_kind=ConfidenceKind.CONFIDENCE,
    )
    values.update(overrides)
    return InstanceMask(**values)


def test_approved_manual_mask_overrides_all_automatic_masks() -> None:
    native = _dense("native", MaskSource.NATIVE_INSTANCE)
    prompted = _dense("prompted", MaskSource.PROMPTED_REFINEMENT)
    manual = _dense(
        "manual-v2",
        MaskSource.MANUAL_CORRECTION,
        parent_mask_id="native",
        revision=2,
        approval_status=ApprovalStatus.APPROVED,
        approved_by="reviewer",
        approved_timestamp=3.0,
    )
    selected = select_authoritative_mask((native, prompted, manual))
    assert selected is manual

    observation = ObjectObservation(
        observation_id="object-1",
        frame_reference=_frame(),
        class_id=1,
        class_name="part",
        masks=(native, prompted, manual),
    )
    assert observation.selected_mask_id == "manual-v2"
    assert observation.selected_mask is manual


def test_unapproved_manual_mask_does_not_override_native_mask() -> None:
    native = _dense("native", MaskSource.NATIVE_INSTANCE)
    pending = _dense(
        "manual-pending",
        MaskSource.MANUAL_CORRECTION,
        parent_mask_id="native",
        revision=1,
        approval_status=ApprovalStatus.PENDING,
    )
    assert select_authoritative_mask((native, pending)) is native


def test_manual_approval_and_revision_provenance_validation() -> None:
    with pytest.raises(ContractValidationError, match="parent_mask_id"):
        _dense(
            "manual",
            MaskSource.MANUAL_CORRECTION,
            revision=1,
            approval_status=ApprovalStatus.PENDING,
        )
    with pytest.raises(ContractValidationError, match="approver"):
        _dense(
            "manual",
            MaskSource.MANUAL_CORRECTION,
            parent_mask_id="native",
            revision=1,
            approval_status=ApprovalStatus.APPROVED,
        )


def test_dense_mask_normalization_and_pickle_safety() -> None:
    mask = _dense("native", MaskSource.NATIVE_INSTANCE)
    normalized = normalize_mask_payload(mask.payload, mask.pixel_space)
    assert normalized.dtype == np.bool_
    assert normalized.shape == (8, 10)
    assert not normalized.flags.writeable
    assert pickle.loads(pickle.dumps(mask)).mask_id == "native"


def test_rle_polygon_and_shared_memory_payloads_are_schema_ready() -> None:
    payloads = (
        RLEMaskPayload(10, 8, (0, 80)),
        PolygonMaskPayload(10, 8, ((1.0, 1.0, 8.0, 1.0, 8.0, 6.0),)),
        SharedMemoryMaskPayload(10, 8, "mask-shm", nbytes=80),
    )
    assert [payload.representation for payload in payloads] == [
        MaskRepresentation.RLE,
        MaskRepresentation.POLYGON,
        MaskRepresentation.SHARED_MEMORY_REFERENCE,
    ]
    for payload in payloads:
        with pytest.raises(UnsupportedMaskRepresentationError, match="deferred"):
            normalize_mask_payload(payload, _space())


def test_raw_detection_box_is_clipped_and_provenance_preserved() -> None:
    rgb_space = PixelCoordinateSpace("rgb", _frame(), ImageSize(10, 8), PixelSpaceRole.RGB)
    box = normalize_detection_box((-2.0, 1.0, 12.0, 9.0), rgb_space)
    assert box.xyxy == (0.0, 1.0, 10.0, 8.0)
    assert box.was_clipped is True
    assert box.raw_xyxy == (-2.0, 1.0, 12.0, 9.0)
    detection = Detection2D(
        "det-1", _frame(), 0, "part", 0.9, box, _provenance()
    )
    assert detection.box.x_max == 10.0

    with pytest.raises(ContractValidationError, match="after clipping"):
        normalize_detection_box((-5.0, 0.0, -1.0, 2.0), rgb_space)
