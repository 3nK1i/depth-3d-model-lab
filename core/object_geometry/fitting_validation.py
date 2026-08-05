"""Validation helpers for Step 5 fitting results."""
from __future__ import annotations
import numpy as np

def validate_box_contains_centre(box):
    if box is None: return
    if not np.isfinite(box.centre_xyz_m).all() or np.any(box.extents_xyz_m<=0): raise ValueError("invalid fitted box")

def validate_fit_result(result):
    for box in (result.axis_aligned_box,result.camera_aligned_box,result.oriented_box): validate_box_contains_centre(box)
    return result
