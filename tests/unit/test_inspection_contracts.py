from pathlib import Path
import pytest

from core.contracts.validation import ContractValidationError
from core.inspection import (
    InspectionBoxMethod,
    InspectionBundleSummary,
    InspectionObjectSummary,
    InspectionRunStatus,
)


def _object() -> InspectionObjectSummary:
    return InspectionObjectSummary(
        observation_id="object-0", object_index=0, class_id=1, class_name="box",
        detection_confidence=0.9, detection_box_xyxy=(1.0, 2.0, 9.0, 10.0),
        mask_key="mask", raw_points_key="raw", filtered_points_key="filtered",
        preferred_box_id="obb", boxes=(("oriented", {"box_id": "obb"}),),
        measurements=(
            {"kind": "width", "status": "available", "reliability": "high"},
            {"kind": "depth", "status": "available", "reliability": "medium"},
        ),
    )


def test_inspection_object_exposes_preferred_box_and_conservative_reliability():
    obj = _object()
    assert obj.box_payload(InspectionBoxMethod.AUTO)["box_id"] == "obb"
    assert obj.box_payload(InspectionBoxMethod.ORIENTED)["box_id"] == "obb"
    assert obj.reliability_level.value == "medium"


def test_inspection_object_rejects_invalid_box_and_confidence():
    with pytest.raises(ContractValidationError):
        InspectionObjectSummary("x", 0, 0, "x", 1.2, (0, 0, 1, 1), "m", "r", "f")
    with pytest.raises(ContractValidationError):
        InspectionObjectSummary("x", 0, 0, "x", 0.5, (1, 1, 1, 2), "m", "r", "f")


def test_bundle_paths_are_confined_to_bundle_directory(tmp_path: Path):
    obj = _object()
    summary = InspectionBundleSummary(
        tmp_path, 1, InspectionRunStatus.PARTIAL, "arrays.npz", "state.pkl", 64, 48,
        "depth-pro", "Depth Pro", "yolo", "YOLO", "metric", "estimated",
        "not_calibrated", "none", InspectionBoxMethod.AUTO, (obj,), elapsed_seconds=0.2,
    )
    assert summary.arrays_path == tmp_path / "arrays.npz"
    bad = InspectionBundleSummary(
        tmp_path, 1, InspectionRunStatus.PARTIAL, "../arrays.npz", "state.pkl", 64, 48,
        "depth-pro", "Depth Pro", "yolo", "YOLO", "metric", "estimated",
        "not_calibrated", "none", InspectionBoxMethod.AUTO, (obj,),
    )
    with pytest.raises(ContractValidationError):
        _ = bad.arrays_path
