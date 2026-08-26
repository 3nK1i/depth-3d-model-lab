"""Bridge offline SfM observations to dense-depth reconstruction-gauge alignment.

B1 deliberately stops before RGB-D projection or multi-view fusion.  It takes one
registered offline-SfM keyframe, samples a dense depth map at the exact sparse
feature observations retained by the SfM adapter, converts the corresponding
landmarks into camera-space Z, and delegates the scale-only fit to the existing
dense-fusion alignment estimator.

No metric scale is inferred here.  The returned scale maps dense-depth source
units into the reconstruction gauge carried by ``OfflineSfmResult``; that gauge
may remain arbitrary.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import isfinite
from typing import Any

import numpy as np

from core.contracts import ImageSize
from core.contracts.validation import require_nonempty
from core.reconstruction.dense_fusion import (
    DenseDepthAlignment,
    DenseDepthAlignmentFit,
    DenseDepthAlignmentMode,
    fit_depth_alignment_from_correspondences,
)
from core.reconstruction.errors import ReconstructionValidationError
from core.reconstruction.sfm_contracts import (
    OfflineSfmResult,
    SfmCameraEstimate,
    SfmCameraModelEstimate,
    SfmScaleStatus,
    SfmWorldAlignmentStatus,
)
from core.reconstruction.sfm_coordinates import invert_rigid_matrix
from core.reconstruction.validation import require_nonnegative_int

SFM_DENSE_ALIGNMENT_SCHEMA_VERSION = 1


class SfmDenseDepthAlignmentStatus(str, Enum):
    ACCEPTED = "accepted"
    REJECTED = "rejected"


class SfmDenseDepthAlignmentRejectionCode(str, Enum):
    KEYFRAME_NOT_REGISTERED = "keyframe_not_registered"
    CAMERA_MODEL_UNAVAILABLE = "camera_model_unavailable"
    DEPTH_PIXEL_SPACE_MISMATCH = "depth_pixel_space_mismatch"
    INSUFFICIENT_CORRESPONDENCES = "insufficient_correspondences"
    ROBUST_SCALE_FIT_REJECTED = "robust_scale_fit_rejected"


@dataclass(frozen=True, slots=True)
class SfmDenseDepthAlignmentEvidence:
    """Accepted/rejected evidence for one SfM-keyframe dense-depth alignment."""

    session_id: str
    keyframe_id: str
    depth_provenance: str
    status: SfmDenseDepthAlignmentStatus
    sfm_scale_status: SfmScaleStatus
    sfm_world_alignment_status: SfmWorldAlignmentStatus
    depth_image_size: ImageSize
    minimum_correspondences: int
    maximum_relative_residual: float
    frame_id: str = ""
    camera_id: str = ""
    camera_model_name: str = ""
    candidate_observations: int = 0
    behind_camera_observations: int = 0
    out_of_bounds_observations: int = 0
    invalid_depth_observations: int = 0
    valid_correspondences: int = 0
    inlier_correspondences: int = 0
    robust_relative_residual_limit: float | None = None
    median_relative_residual: float | None = None
    p90_relative_residual: float | None = None
    alignment: DenseDepthAlignment | None = None
    rejection_code: SfmDenseDepthAlignmentRejectionCode | None = None
    message: str = ""
    schema_version: int = SFM_DENSE_ALIGNMENT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "session_id", require_nonempty(self.session_id, field_name="session_id")
        )
        object.__setattr__(
            self, "keyframe_id", require_nonempty(self.keyframe_id, field_name="keyframe_id")
        )
        object.__setattr__(
            self,
            "depth_provenance",
            require_nonempty(self.depth_provenance, field_name="depth_provenance"),
        )
        if not isinstance(self.status, SfmDenseDepthAlignmentStatus):
            raise TypeError("status must be SfmDenseDepthAlignmentStatus.")
        if not isinstance(self.sfm_scale_status, SfmScaleStatus):
            raise TypeError("sfm_scale_status must be SfmScaleStatus.")
        if not isinstance(self.sfm_world_alignment_status, SfmWorldAlignmentStatus):
            raise TypeError("sfm_world_alignment_status must be SfmWorldAlignmentStatus.")
        if not isinstance(self.depth_image_size, ImageSize):
            raise TypeError("depth_image_size must be ImageSize.")

        minimum = require_nonnegative_int(
            self.minimum_correspondences, field_name="minimum_correspondences"
        )
        if minimum < 3:
            raise ReconstructionValidationError("minimum_correspondences must be at least 3.")
        maximum = float(self.maximum_relative_residual)
        if not isfinite(maximum) or not 0.0 < maximum < 1.0:
            raise ReconstructionValidationError(
                "maximum_relative_residual must be finite and in (0, 1)."
            )
        object.__setattr__(self, "minimum_correspondences", minimum)
        object.__setattr__(self, "maximum_relative_residual", maximum)

        for name in (
            "candidate_observations",
            "behind_camera_observations",
            "out_of_bounds_observations",
            "invalid_depth_observations",
            "valid_correspondences",
            "inlier_correspondences",
        ):
            object.__setattr__(
                self, name, require_nonnegative_int(getattr(self, name), field_name=name)
            )
        classified = (
            self.behind_camera_observations
            + self.out_of_bounds_observations
            + self.invalid_depth_observations
            + self.valid_correspondences
        )
        if classified != self.candidate_observations:
            raise ReconstructionValidationError(
                "Every candidate SfM observation must have exactly one B1 classification."
            )
        if self.inlier_correspondences > self.valid_correspondences:
            raise ReconstructionValidationError(
                "inlier_correspondences cannot exceed valid_correspondences."
            )

        for name in (
            "robust_relative_residual_limit",
            "median_relative_residual",
            "p90_relative_residual",
        ):
            value = getattr(self, name)
            if value is None:
                continue
            numeric = float(value)
            if not isfinite(numeric) or numeric < 0.0:
                raise ReconstructionValidationError(f"{name} must be finite and non-negative.")
            object.__setattr__(self, name, numeric)

        object.__setattr__(self, "frame_id", str(self.frame_id).strip())
        object.__setattr__(self, "camera_id", str(self.camera_id).strip())
        object.__setattr__(self, "camera_model_name", str(self.camera_model_name).strip())
        object.__setattr__(self, "message", require_nonempty(self.message, field_name="message"))

        if self.status is SfmDenseDepthAlignmentStatus.ACCEPTED:
            if not isinstance(self.alignment, DenseDepthAlignment):
                raise ReconstructionValidationError(
                    "Accepted SfM/dense alignment evidence requires a DenseDepthAlignment."
                )
            if self.alignment.mode is not DenseDepthAlignmentMode.SPARSE_CORRESPONDENCE_SCALE:
                raise ReconstructionValidationError(
                    "B1 accepted evidence requires sparse-correspondence scale alignment."
                )
            if self.rejection_code is not None:
                raise ReconstructionValidationError(
                    "Accepted SfM/dense alignment evidence cannot carry a rejection code."
                )
            if self.valid_correspondences < self.minimum_correspondences:
                raise ReconstructionValidationError(
                    "Accepted evidence requires the configured minimum correspondences."
                )
            if self.inlier_correspondences < self.minimum_correspondences:
                raise ReconstructionValidationError(
                    "Accepted evidence requires the configured minimum inlier correspondences."
                )
            if any(
                value is None
                for value in (
                    self.robust_relative_residual_limit,
                    self.median_relative_residual,
                    self.p90_relative_residual,
                )
            ):
                raise ReconstructionValidationError(
                    "Accepted evidence requires robust-fit residual diagnostics."
                )
        else:
            if self.alignment is not None:
                raise ReconstructionValidationError(
                    "Rejected SfM/dense alignment evidence must not carry an alignment."
                )
            if not isinstance(self.rejection_code, SfmDenseDepthAlignmentRejectionCode):
                raise ReconstructionValidationError(
                    "Rejected SfM/dense alignment evidence requires a rejection code."
                )

        if (
            isinstance(self.schema_version, bool)
            or int(self.schema_version) != SFM_DENSE_ALIGNMENT_SCHEMA_VERSION
        ):
            raise ReconstructionValidationError(
                f"Unsupported SfM/dense alignment schema version: {self.schema_version}."
            )
        object.__setattr__(self, "schema_version", SFM_DENSE_ALIGNMENT_SCHEMA_VERSION)

    @property
    def is_accepted(self) -> bool:
        return self.status is SfmDenseDepthAlignmentStatus.ACCEPTED

    @property
    def preserves_arbitrary_gauge(self) -> bool:
        return self.sfm_scale_status is SfmScaleStatus.ARBITRARY

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "session_id": self.session_id,
            "keyframe_id": self.keyframe_id,
            "frame_id": self.frame_id,
            "camera_id": self.camera_id,
            "camera_model_name": self.camera_model_name,
            "depth_provenance": self.depth_provenance,
            "status": self.status.value,
            "rejection_code": self.rejection_code.value if self.rejection_code else None,
            "message": self.message,
            "sfm_scale_status": self.sfm_scale_status.value,
            "sfm_world_alignment_status": self.sfm_world_alignment_status.value,
            "depth_image_size": self.depth_image_size.as_dict(),
            "minimum_correspondences": self.minimum_correspondences,
            "maximum_relative_residual": self.maximum_relative_residual,
            "candidate_observations": self.candidate_observations,
            "behind_camera_observations": self.behind_camera_observations,
            "out_of_bounds_observations": self.out_of_bounds_observations,
            "invalid_depth_observations": self.invalid_depth_observations,
            "valid_correspondences": self.valid_correspondences,
            "inlier_correspondences": self.inlier_correspondences,
            "robust_relative_residual_limit": self.robust_relative_residual_limit,
            "median_relative_residual": self.median_relative_residual,
            "p90_relative_residual": self.p90_relative_residual,
            "alignment": self.alignment.as_dict() if self.alignment is not None else None,
            "metric_scale_claimed": False,
        }


def align_sfm_keyframe_dense_depth(
    sfm_result: OfflineSfmResult,
    *,
    keyframe_id: str,
    dense_depth_source_units: object,
    depth_provenance: str,
    minimum_correspondences: int = 8,
    maximum_relative_residual: float = 0.35,
) -> SfmDenseDepthAlignmentEvidence:
    """Estimate one keyframe's dense-depth scale into the offline-SfM gauge.

    Expected evidence failures are returned as explicit rejected evidence rather
    than exceptions.  Contract/programming errors (wrong argument types, invalid
    estimator configuration, non-2D depth input) remain exceptions.
    """

    if not isinstance(sfm_result, OfflineSfmResult):
        raise TypeError("sfm_result must be OfflineSfmResult.")
    keyframe_id = require_nonempty(keyframe_id, field_name="keyframe_id")
    depth_provenance = require_nonempty(depth_provenance, field_name="depth_provenance")
    if isinstance(minimum_correspondences, bool) or int(minimum_correspondences) < 3:
        raise ReconstructionValidationError("minimum_correspondences must be at least 3.")
    minimum_correspondences = int(minimum_correspondences)
    maximum_relative_residual = float(maximum_relative_residual)
    if not isfinite(maximum_relative_residual) or not 0.0 < maximum_relative_residual < 1.0:
        raise ReconstructionValidationError(
            "maximum_relative_residual must be finite and in (0, 1)."
        )

    try:
        depth = np.asarray(dense_depth_source_units, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ReconstructionValidationError("dense_depth_source_units must be numeric.") from exc
    if depth.ndim != 2 or depth.shape[0] <= 0 or depth.shape[1] <= 0:
        raise ReconstructionValidationError(
            "dense_depth_source_units must be a non-empty HxW depth map."
        )
    depth_size = ImageSize(width=int(depth.shape[1]), height=int(depth.shape[0]))

    camera = next((item for item in sfm_result.cameras if item.keyframe_id == keyframe_id), None)
    if camera is None:
        return _rejected_evidence(
            sfm_result,
            keyframe_id=keyframe_id,
            depth_provenance=depth_provenance,
            depth_size=depth_size,
            minimum_correspondences=minimum_correspondences,
            maximum_relative_residual=maximum_relative_residual,
            rejection_code=SfmDenseDepthAlignmentRejectionCode.KEYFRAME_NOT_REGISTERED,
            message="The selected keyframe has no registered offline-SfM camera pose.",
        )

    camera_model = next(
        (item for item in sfm_result.camera_models if item.camera_id == camera.camera_id), None
    )
    if camera_model is None:
        return _rejected_evidence(
            sfm_result,
            keyframe_id=keyframe_id,
            depth_provenance=depth_provenance,
            depth_size=depth_size,
            minimum_correspondences=minimum_correspondences,
            maximum_relative_residual=maximum_relative_residual,
            camera=camera,
            rejection_code=SfmDenseDepthAlignmentRejectionCode.CAMERA_MODEL_UNAVAILABLE,
            message=(
                "The registered SfM camera has no returned camera-model estimate, so the dense "
                "depth pixel space cannot be proven compatible."
            ),
        )

    workspace_image = next(
        (item for item in sfm_result.workspace.images if item.keyframe_id == keyframe_id), None
    )
    # OfflineSfmResult already guarantees the camera references a workspace keyframe.
    if workspace_image is None:  # defensive against corrupted/deserialized external values
        return _rejected_evidence(
            sfm_result,
            keyframe_id=keyframe_id,
            depth_provenance=depth_provenance,
            depth_size=depth_size,
            minimum_correspondences=minimum_correspondences,
            maximum_relative_residual=maximum_relative_residual,
            camera=camera,
            camera_model=camera_model,
            rejection_code=SfmDenseDepthAlignmentRejectionCode.KEYFRAME_NOT_REGISTERED,
            message="The registered SfM keyframe is missing from its workspace manifest.",
        )

    if depth_size != camera_model.image_size or depth_size != workspace_image.image_size:
        return _rejected_evidence(
            sfm_result,
            keyframe_id=keyframe_id,
            depth_provenance=depth_provenance,
            depth_size=depth_size,
            minimum_correspondences=minimum_correspondences,
            maximum_relative_residual=maximum_relative_residual,
            camera=camera,
            camera_model=camera_model,
            rejection_code=SfmDenseDepthAlignmentRejectionCode.DEPTH_PIXEL_SPACE_MISMATCH,
            message=(
                "Dense depth, SfM camera model, and staged keyframe must share the same pixel "
                "dimensions before sparse/depth correspondence sampling."
            ),
        )

    T_camera_from_reconstruction = invert_rigid_matrix(camera.T_reconstruction_from_camera)
    source_depths: list[float] = []
    camera_depths: list[float] = []
    candidate = 0
    behind = 0
    out_of_bounds = 0
    invalid_depth = 0

    for landmark in sfm_result.landmarks:
        observation = next(
            (item for item in landmark.observations if item.keyframe_id == keyframe_id), None
        )
        if observation is None:
            continue
        candidate += 1
        position_camera = (
            T_camera_from_reconstruction[:3, :3] @ landmark.position_reconstruction
            + T_camera_from_reconstruction[:3, 3]
        )
        camera_z = float(position_camera[2])
        if not isfinite(camera_z) or camera_z <= 0.0:
            behind += 1
            continue

        x, y = observation.feature_xy
        sampled = _bilinear_positive_depth(depth, x=float(x), y=float(y))
        if sampled is _OUT_OF_BOUNDS:
            out_of_bounds += 1
            continue
        if sampled is None:
            invalid_depth += 1
            continue
        source_depths.append(float(sampled))
        camera_depths.append(camera_z)

    valid = len(source_depths)
    classified = {
        "candidate_observations": candidate,
        "behind_camera_observations": behind,
        "out_of_bounds_observations": out_of_bounds,
        "invalid_depth_observations": invalid_depth,
        "valid_correspondences": valid,
    }
    if valid < minimum_correspondences:
        return _rejected_evidence(
            sfm_result,
            keyframe_id=keyframe_id,
            depth_provenance=depth_provenance,
            depth_size=depth_size,
            minimum_correspondences=minimum_correspondences,
            maximum_relative_residual=maximum_relative_residual,
            camera=camera,
            camera_model=camera_model,
            rejection_code=SfmDenseDepthAlignmentRejectionCode.INSUFFICIENT_CORRESPONDENCES,
            message=(
                f"Only {valid} valid SfM/depth correspondences remain; "
                f"{minimum_correspondences} are required."
            ),
            **classified,
        )

    provenance = (
        f"{depth_provenance};sfm_backend={sfm_result.backend_id};"
        f"sfm_keyframe={keyframe_id};reconstruction_gauge={sfm_result.gauge.scale_status.value}"
    )
    try:
        fit = fit_depth_alignment_from_correspondences(
            source_depths,
            camera_depths,
            provenance=provenance,
            minimum_correspondences=minimum_correspondences,
            maximum_relative_residual=maximum_relative_residual,
        )
    except ReconstructionValidationError as exc:
        return _rejected_evidence(
            sfm_result,
            keyframe_id=keyframe_id,
            depth_provenance=depth_provenance,
            depth_size=depth_size,
            minimum_correspondences=minimum_correspondences,
            maximum_relative_residual=maximum_relative_residual,
            camera=camera,
            camera_model=camera_model,
            rejection_code=SfmDenseDepthAlignmentRejectionCode.ROBUST_SCALE_FIT_REJECTED,
            message=f"Robust sparse/depth scale fit rejected the keyframe: {exc}",
            **classified,
        )

    return _accepted_evidence(
        sfm_result,
        keyframe_id=keyframe_id,
        depth_provenance=depth_provenance,
        depth_size=depth_size,
        minimum_correspondences=minimum_correspondences,
        maximum_relative_residual=maximum_relative_residual,
        camera=camera,
        camera_model=camera_model,
        fit=fit,
        **classified,
    )


_OUT_OF_BOUNDS = object()


def _bilinear_positive_depth(depth: np.ndarray, *, x: float, y: float) -> float | None | object:
    height, width = depth.shape
    if x < 0.0 or y < 0.0 or x > width - 1 or y > height - 1:
        return _OUT_OF_BOUNDS

    x0 = int(np.floor(x))
    y0 = int(np.floor(y))
    x1 = min(x0 + 1, width - 1)
    y1 = min(y0 + 1, height - 1)
    values = np.asarray(
        [depth[y0, x0], depth[y0, x1], depth[y1, x0], depth[y1, x1]],
        dtype=np.float64,
    )
    if not np.isfinite(values).all() or np.any(values <= 0.0):
        return None

    wx = x - x0
    wy = y - y0
    top = values[0] * (1.0 - wx) + values[1] * wx
    bottom = values[2] * (1.0 - wx) + values[3] * wx
    sampled = float(top * (1.0 - wy) + bottom * wy)
    return sampled if isfinite(sampled) and sampled > 0.0 else None


def _base_evidence_kwargs(
    sfm_result: OfflineSfmResult,
    *,
    keyframe_id: str,
    depth_provenance: str,
    depth_size: ImageSize,
    minimum_correspondences: int,
    maximum_relative_residual: float,
    camera: SfmCameraEstimate | None = None,
    camera_model: SfmCameraModelEstimate | None = None,
    candidate_observations: int = 0,
    behind_camera_observations: int = 0,
    out_of_bounds_observations: int = 0,
    invalid_depth_observations: int = 0,
    valid_correspondences: int = 0,
) -> dict[str, Any]:
    return {
        "session_id": sfm_result.session_id,
        "keyframe_id": keyframe_id,
        "frame_id": camera.frame_id if camera is not None else "",
        "camera_id": camera.camera_id if camera is not None else "",
        "camera_model_name": camera_model.backend_model_name if camera_model is not None else "",
        "depth_provenance": depth_provenance,
        "sfm_scale_status": sfm_result.gauge.scale_status,
        "sfm_world_alignment_status": sfm_result.gauge.alignment_status,
        "depth_image_size": depth_size,
        "minimum_correspondences": minimum_correspondences,
        "maximum_relative_residual": maximum_relative_residual,
        "candidate_observations": candidate_observations,
        "behind_camera_observations": behind_camera_observations,
        "out_of_bounds_observations": out_of_bounds_observations,
        "invalid_depth_observations": invalid_depth_observations,
        "valid_correspondences": valid_correspondences,
    }


def _rejected_evidence(
    sfm_result: OfflineSfmResult,
    *,
    rejection_code: SfmDenseDepthAlignmentRejectionCode,
    message: str,
    **kwargs: Any,
) -> SfmDenseDepthAlignmentEvidence:
    return SfmDenseDepthAlignmentEvidence(
        **_base_evidence_kwargs(sfm_result, **kwargs),
        status=SfmDenseDepthAlignmentStatus.REJECTED,
        rejection_code=rejection_code,
        message=message,
    )


def _accepted_evidence(
    sfm_result: OfflineSfmResult,
    *,
    fit: DenseDepthAlignmentFit,
    **kwargs: Any,
) -> SfmDenseDepthAlignmentEvidence:
    return SfmDenseDepthAlignmentEvidence(
        **_base_evidence_kwargs(sfm_result, **kwargs),
        status=SfmDenseDepthAlignmentStatus.ACCEPTED,
        alignment=fit.alignment,
        inlier_correspondences=fit.inlier_correspondences,
        robust_relative_residual_limit=fit.robust_relative_residual_limit,
        median_relative_residual=fit.median_relative_residual,
        p90_relative_residual=fit.p90_relative_residual,
        message=(
            "Dense depth accepted in the offline-SfM reconstruction gauge; this alignment does "
            "not by itself establish physical metric scale."
        ),
    )
