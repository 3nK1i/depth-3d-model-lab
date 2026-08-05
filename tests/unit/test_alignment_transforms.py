from __future__ import annotations

import numpy as np
import pytest

from core.alignment.factories import (
    declare_crop_resize_padding,
    declare_full_frame_resize,
    declare_identity_transform,
)
from core.alignment.transforms import (
    is_nonuniform_scale,
    resolve_direct_transform,
    transform_detection_box,
    transform_points_xy,
)
from core.contracts.common import FrameReference, ImageSize
from core.contracts.spatial import PixelCoordinateSpace, PixelSpaceRole
from core.contracts.validation import ContractValidationError
from core.perception.result import BoundingBox2D


def _spaces(source_size=(8, 6), destination_size=(4, 3)):
    frame = FrameReference(1, 1, 0.0, ImageSize(*source_size))
    source = PixelCoordinateSpace(
        "rgb", frame, ImageSize(*source_size), PixelSpaceRole.RGB
    )
    destination = PixelCoordinateSpace(
        "depth", frame, ImageSize(*destination_size), PixelSpaceRole.DEPTH
    )
    return source, destination


def test_identity_requires_equal_sizes_and_maps_points_exactly():
    source, destination = _spaces((8, 6), (8, 6))
    transform = declare_identity_transform(source, destination)
    points = np.array([[0.0, 0.0], [8.0, 6.0], [3.5, 2.0]])
    assert np.allclose(transform_points_xy(points, transform), points)
    _, smaller = _spaces((8, 6), (4, 3))
    with pytest.raises(ContractValidationError):
        declare_identity_transform(source, smaller)


def test_full_frame_resize_and_detection_box_clipping():
    source, destination = _spaces()
    transform = declare_full_frame_resize(source, destination)
    box = BoundingBox2D(-1.0, 1.0, 9.0, 7.0, "rgb", raw_xyxy=(-1.0, 1.0, 9.0, 7.0))
    aligned = transform_detection_box(box, source, destination, transform)
    assert aligned.xyxy == pytest.approx((0.0, 0.5, 4.0, 3.0))
    assert aligned.was_clipped
    assert aligned.pixel_space_id == "depth"


def test_crop_resize_padding_uses_authoritative_matrix():
    source, destination = _spaces((10, 8), (8, 8))
    transform = declare_crop_resize_padding(
        source,
        destination,
        crop_xyxy=(2.0, 1.0, 8.0, 7.0),
        resized_size=ImageSize(6, 6),
        padding_ltrb=(1, 1, 1, 1),
    )
    mapped = transform_points_xy(np.array([[2.0, 1.0], [8.0, 7.0]]), transform)
    assert mapped[0] == pytest.approx((1.0, 1.0))
    assert mapped[1] == pytest.approx((7.0, 7.0))


def test_missing_transform_is_rejected_without_explicit_implicit_policy():
    source, destination = _spaces()
    with pytest.raises(ContractValidationError):
        resolve_direct_transform(source, destination, ())
    transform, warnings = resolve_direct_transform(
        source, destination, (), allow_implicit_resize=True
    )
    assert transform.destination_space == destination
    assert warnings == ("implicit_full_frame_resize_assumption",)


def test_nonuniform_resize_is_detected():
    source, destination = _spaces((8, 8), (8, 4))
    assert is_nonuniform_scale(declare_full_frame_resize(source, destination))


def test_authoritative_matrix_disagreement_is_rejected():
    from core.contracts.spatial import ImageTransform2D

    source, destination = _spaces()
    with pytest.raises(ContractValidationError):
        ImageTransform2D(
            source,
            destination,
            np.eye(3),
            crop_xyxy=(0.0, 0.0, 8.0, 6.0),
            resized_size=ImageSize(4, 3),
            padding_ltrb=(0, 0, 0, 0),
        )
