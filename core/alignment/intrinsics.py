"""Camera-intrinsics propagation through declared pixel transforms."""

from __future__ import annotations

import numpy as np

from core.contracts.spatial import CameraIntrinsics, ImageTransform2D
from core.contracts.validation import ContractValidationError


def transform_intrinsics(
    intrinsics: CameraIntrinsics,
    transform: ImageTransform2D,
) -> CameraIntrinsics:
    """Transform pinhole intrinsics with the authoritative source->destination matrix."""

    if intrinsics.pixel_space_id != transform.source_space.space_id:
        raise ContractValidationError(
            "Camera intrinsics do not reference the transform source space."
        )
    matrix = transform.matrix @ intrinsics.matrix
    if not np.allclose(matrix[2], np.array([0.0, 0.0, 1.0]), atol=1e-9):
        raise ContractValidationError(
            "Pixel transform is not compatible with a pinhole intrinsics matrix."
        )
    return CameraIntrinsics(
        matrix=matrix,
        pixel_space_id=transform.destination_space.space_id,
        source=intrinsics.source,
        calibration_status=intrinsics.calibration_status,
        calibration_record_id=intrinsics.calibration_record_id,
    )


def preserve_native_intrinsics(
    intrinsics: CameraIntrinsics,
    target_space_id: str,
) -> CameraIntrinsics:
    if intrinsics.pixel_space_id != target_space_id:
        raise ContractValidationError(
            "Native depth intrinsics must already reference the target depth grid."
        )
    return intrinsics
