from __future__ import annotations

import pickle

import numpy as np
import pytest

from core.contracts.spatial import AxisConvention, CoordinateFrame, CoordinateFrameKind
from core.contracts.validation import ContractValidationError
from core.measurement.contracts import RejectionReason
from core.object_geometry.contracts import (
    ExtractionStageCounts,
    ObjectPointExtractionBatchResult,
    ObjectPointExtractionResult,
    ObjectPointExtractionStatus,
    PointCoordinateSemantics,
)
from tests.object_geometry_helpers import make_bundle


def _frame(bundle):
    return CoordinateFrame(
        "camera", CoordinateFrameKind.CAMERA_DEPTH,
        AxisConvention.X_RIGHT_Y_DOWN_Z_FORWARD,
        source_frame=bundle.frame_reference,
    )


def test_stage_counts_are_monotonic_and_serializable():
    counts = ExtractionStageCounts(10, 9, 8, 7, 6, 5, 4, 4, 3, 3, 2, 2)
    assert counts.as_dict()["output_points"] == 2
    assert pickle.loads(pickle.dumps(counts)) == counts
    with pytest.raises(ContractValidationError):
        ExtractionStageCounts(1, 2)


def test_rejected_result_requires_reason_and_no_filtered_cloud():
    bundle = make_bundle()
    frame = _frame(bundle)
    result = ObjectPointExtractionResult(
        "object-0", bundle.frame_reference, ObjectPointExtractionStatus.REJECTED,
        frame, PointCoordinateSemantics.METRIC_METRES,
        rejection_reasons=(RejectionReason.MASK_UNAVAILABLE,),
    )
    assert result.filtered_cloud is None
    with pytest.raises(ContractValidationError):
        ObjectPointExtractionResult(
            "object-0", bundle.frame_reference, ObjectPointExtractionStatus.REJECTED,
            frame, PointCoordinateSemantics.METRIC_METRES,
        )


def test_batch_status_must_match_members():
    bundle = make_bundle()
    frame = _frame(bundle)
    rejected = ObjectPointExtractionResult(
        "object-0", bundle.frame_reference, ObjectPointExtractionStatus.REJECTED,
        frame, PointCoordinateSemantics.METRIC_METRES,
        rejection_reasons=(RejectionReason.MASK_UNAVAILABLE,),
    )
    batch = ObjectPointExtractionBatchResult(
        bundle.frame_reference, frame, (rejected,), ObjectPointExtractionStatus.REJECTED
    )
    assert batch.status is ObjectPointExtractionStatus.REJECTED
    with pytest.raises(ContractValidationError):
        ObjectPointExtractionBatchResult(
            bundle.frame_reference, frame, (rejected,), ObjectPointExtractionStatus.SUCCESS
        )
