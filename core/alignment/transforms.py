"""Pixel-space transform lookup and geometry helpers."""

from __future__ import annotations

import numpy as np

from core.alignment.factories import declare_full_frame_resize, declare_identity_transform
from core.contracts.spatial import ImageTransform2D, PixelCoordinateSpace
from core.contracts.validation import ContractValidationError
from core.perception.result import BoundingBox2D, normalize_detection_box


def resolve_direct_transform(
    source_space: PixelCoordinateSpace,
    destination_space: PixelCoordinateSpace,
    transforms: tuple[ImageTransform2D, ...],
    *,
    allow_implicit_resize: bool = False,
    interpolation: str = "auto",
) -> tuple[ImageTransform2D, tuple[str, ...]]:
    """Resolve one direct transform; transform chaining is intentionally deferred."""

    if source_space.frame_reference != destination_space.frame_reference:
        raise ContractValidationError("Cannot align pixel spaces from different captured frames.")
    if source_space.space_id == destination_space.space_id:
        if source_space != destination_space:
            raise ContractValidationError(
                "Equal pixel-space IDs must describe the same coordinate space."
            )
        return declare_identity_transform(source_space, destination_space), ()

    matches = [
        value
        for value in transforms
        if value.source_space.space_id == source_space.space_id
        and value.destination_space.space_id == destination_space.space_id
    ]
    if len(matches) > 1:
        raise ContractValidationError(
            f"Multiple transforms declared for {source_space.space_id!r} -> "
            f"{destination_space.space_id!r}."
        )
    if matches:
        transform = matches[0]
        if transform.source_space != source_space or transform.destination_space != destination_space:
            raise ContractValidationError(
                "Transform pixel-space declarations disagree with the requested spaces."
            )
        return transform, ()
    if not allow_implicit_resize:
        raise ContractValidationError(
            f"No explicit transform declared for {source_space.space_id!r} -> "
            f"{destination_space.space_id!r}."
        )
    return (
        declare_full_frame_resize(
            source_space,
            destination_space,
            interpolation=interpolation,
        ),
        ("implicit_full_frame_resize_assumption",),
    )


def transform_points_xy(points_xy: np.ndarray, transform: ImageTransform2D) -> np.ndarray:
    points = np.asarray(points_xy, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ContractValidationError("points_xy must have shape Nx2.")
    if not np.isfinite(points).all():
        raise ContractValidationError("points_xy must contain finite coordinates.")
    homogeneous = np.concatenate(
        [points, np.ones((points.shape[0], 1), dtype=np.float64)], axis=1
    )
    mapped = homogeneous @ transform.matrix.T
    if np.any(np.isclose(mapped[:, 2], 0.0)):
        raise ContractValidationError("Pixel transform produced a zero homogeneous scale.")
    return mapped[:, :2] / mapped[:, 2:3]


def transform_detection_box(
    box: BoundingBox2D,
    source_space: PixelCoordinateSpace,
    destination_space: PixelCoordinateSpace,
    transform: ImageTransform2D,
) -> BoundingBox2D:
    if box.pixel_space_id != source_space.space_id:
        raise ContractValidationError("Detection box does not reference source_space.")
    x0, y0, x1, y1 = box.xyxy
    corners = np.asarray(
        [[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=np.float64
    )
    mapped = transform_points_xy(corners, transform)
    raw = (
        float(np.min(mapped[:, 0])),
        float(np.min(mapped[:, 1])),
        float(np.max(mapped[:, 0])),
        float(np.max(mapped[:, 1])),
    )
    return normalize_detection_box(raw, destination_space)


def linear_area_scale(transform: ImageTransform2D) -> float:
    return abs(float(np.linalg.det(transform.matrix[:2, :2])))


def is_nonuniform_scale(transform: ImageTransform2D, *, tolerance: float = 1e-9) -> bool:
    return not np.isclose(
        abs(float(transform.matrix[0, 0])),
        abs(float(transform.matrix[1, 1])),
        rtol=tolerance,
        atol=tolerance,
    )


def is_full_frame_resize(transform: ImageTransform2D) -> bool:
    source = transform.source_space.size
    crop = transform.crop_xyxy
    return (
        crop is not None
        and np.allclose(crop, (0.0, 0.0, float(source.width), float(source.height)))
        and transform.padding_ltrb == (0, 0, 0, 0)
        and transform.resized_size == transform.destination_space.size
    )
