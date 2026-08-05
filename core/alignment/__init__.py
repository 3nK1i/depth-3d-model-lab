"""Public Phase 7.5 RGB/mask/depth/intrinsics alignment API."""

from core.alignment.contracts import (
    AlignmentPolicy,
    AlignmentRequest,
    AlignmentStatus,
    AlignmentTargetGrid,
    AlignedFrameBundle,
    AlignedObjectObservation,
    DepthAlignmentInput,
    ResamplingMethod,
)
from core.alignment.factories import (
    declare_crop_resize_padding,
    declare_full_frame_resize,
    declare_identity_transform,
    depth_input_from_inference_result,
    make_depth_space,
)
from core.alignment.intrinsics import preserve_native_intrinsics, transform_intrinsics
from core.alignment.pipeline import align_frame
from core.alignment.resampling import (
    resample_boolean_mask,
    resample_continuous_map,
    resample_rgb,
    resample_support_mask,
)
from core.alignment.transforms import (
    is_full_frame_resize,
    is_nonuniform_scale,
    linear_area_scale,
    resolve_direct_transform,
    transform_detection_box,
    transform_points_xy,
)
from core.alignment.validation import (
    compose_confidence_valid_mask,
    compose_depth_valid_mask,
    validate_aligned_bundle,
    validate_request_identity,
)

__all__ = [name for name in globals() if not name.startswith("_")]
