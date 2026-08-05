from __future__ import annotations

import numpy as np

from core.alignment.factories import declare_crop_resize_padding, declare_full_frame_resize
from core.alignment.resampling import (
    resample_boolean_mask,
    resample_continuous_map,
    resample_rgb,
    resample_support_mask,
)
from core.contracts.common import FrameReference, ImageSize
from core.contracts.spatial import PixelCoordinateSpace, PixelSpaceRole


def _spaces(source_size=(8, 6), destination_size=(4, 3)):
    frame = FrameReference(2, 1, 0.0, ImageSize(*source_size))
    source = PixelCoordinateSpace("rgb", frame, ImageSize(*source_size), PixelSpaceRole.RGB)
    destination = PixelCoordinateSpace("depth", frame, ImageSize(*destination_size), PixelSpaceRole.DEPTH)
    return source, destination


def test_rgb_resize_returns_uint8_readonly_without_mutating_source():
    source_space, destination_space = _spaces()
    transform = declare_full_frame_resize(source_space, destination_space)
    source = np.arange(6 * 8 * 3, dtype=np.uint8).reshape(6, 8, 3)
    original = source.copy()
    output = resample_rgb(source, transform)
    assert output.shape == (3, 4, 3)
    assert output.dtype == np.uint8
    assert not output.flags.writeable
    assert np.array_equal(source, original)


def test_boolean_mask_uses_nearest_and_never_contains_fractional_values():
    source_space, destination_space = _spaces((4, 4), (9, 7))
    transform = declare_full_frame_resize(source_space, destination_space)
    mask = np.zeros((4, 4), bool)
    mask[1:3, 1:3] = True
    output = resample_boolean_mask(mask, transform)
    assert output.dtype == np.bool_
    assert set(np.unique(output)).issubset({False, True})
    assert np.any(output)


def test_crop_padding_support_mask_marks_only_supported_pixels():
    source_space, destination_space = _spaces((6, 6), (8, 8))
    transform = declare_crop_resize_padding(
        source_space,
        destination_space,
        crop_xyxy=(1.0, 1.0, 5.0, 5.0),
        resized_size=ImageSize(4, 4),
        padding_ltrb=(2, 2, 2, 2),
    )
    support = resample_support_mask(transform)
    assert support.shape == (8, 8)
    assert not np.any(support[:2])
    assert not np.any(support[:, :2])
    assert np.all(support[2:6, 2:6])


def test_continuous_map_alignment_is_numeric_and_readonly():
    source_space, destination_space = _spaces()
    transform = declare_full_frame_resize(source_space, destination_space)
    values = np.linspace(0.0, 1.0, 48, dtype=np.float32).reshape(6, 8)
    output = resample_continuous_map(values, transform)
    assert output.shape == (3, 4)
    assert np.isfinite(output).all()
    assert not output.flags.writeable
