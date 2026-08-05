from __future__ import annotations
import numpy as np
import pytest
from core.contracts.common import FrameReference, ImageSize, ModelProvenance
from core.contracts.spatial import PixelCoordinateSpace, PixelSpaceRole
from core.contracts.validation import ContractValidationError
from core.models.metadata import ConfidenceKind
from core.perception.normalization import border_contacts, build_box_fallback_mask, normalize_native_mask, normalize_raw_detection
from core.perception.raw import RawPerceptionInstance

def setup():
    f = FrameReference(1, 1, 0, ImageSize(10, 8))
    s = PixelCoordinateSpace('rgb', f, f.source_size, PixelSpaceRole.RGB)
    p = ModelProvenance('m', 'M', 'seg')
    return (f, s, p)

def test_box_clips_and_fallback_is_dense():
    f, s, p = setup()
    raw = RawPerceptionInstance('x', 0, 'part', 0.9, (-2, -1, 12, 9))
    d = normalize_raw_detection(raw, f, s, p, prefix='p')
    assert d.box.was_clipped
    assert border_contacts(d, s) == ('left', 'top', 'right', 'bottom')
    m = build_box_fallback_mask(d, s, p, mask_id='box')
    assert m.payload.data.all()

def test_native_mask_must_match_rgb_space():
    f, s, p = setup()
    raw = RawPerceptionInstance('x', 0, 'part', 0.9, (0, 0, 5, 5), np.ones((4, 5), bool), 0.8, ConfidenceKind.CONFIDENCE)
    with pytest.raises(ContractValidationError, match='shape'):
        normalize_native_mask(raw, f, s, p, prefix='p')

def test_overlapping_masks_remain_independent():
    f, s, p = setup()
    a = np.zeros((8, 10), bool)
    b = np.zeros_like(a)
    a[:, 1:6] = 1
    b[:, 4:9] = 1
    ma = normalize_native_mask(RawPerceptionInstance('a', 0, 'a', 0.9, (1, 0, 6, 8), a), f, s, p, prefix='p')
    mb = normalize_native_mask(RawPerceptionInstance('b', 1, 'b', 0.8, (4, 0, 9, 8), b), f, s, p, prefix='p')
    assert np.any(ma.payload.data & mb.payload.data)

def test_ultralytics_adapter_converts_public_rgb_to_bgr():
    from core.perception.adapters.ultralytics_yolo import _ultralytics_source
    rgb = np.asarray([[[1, 2, 3]]], dtype=np.uint8)
    bgr = _ultralytics_source(rgb)
    assert bgr.tolist() == [[[3, 2, 1]]]
    assert bgr.flags.c_contiguous
