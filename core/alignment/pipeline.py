"""Authoritative Step 3 alignment pipeline using the native depth grid."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from core.alignment.contracts import (
    AlignmentRequest,
    AlignmentStatus,
    AlignedFrameBundle,
    AlignedObjectObservation,
    unique_model_provenance,
)
from core.alignment.intrinsics import preserve_native_intrinsics
from core.alignment.resampling import resample_boolean_mask, resample_rgb, resample_support_mask
from core.alignment.transforms import (
    is_nonuniform_scale,
    linear_area_scale,
    resolve_direct_transform,
    transform_detection_box,
)
from core.alignment.validation import (
    compose_confidence_valid_mask,
    compose_depth_valid_mask,
    validate_aligned_bundle,
    validate_request_identity,
)
from core.contracts.common import ProcessingStep
from core.contracts.spatial import ImageTransform2D, PixelCoordinateSpace
from core.contracts.validation import ContractValidationError
from core.perception.result import (
    ObjectObservation,
    UnsupportedMaskRepresentationError,
    normalize_mask_payload,
)


@dataclass(slots=True)
class _AlignmentContext:
    request: AlignmentRequest
    spaces: dict[str, PixelCoordinateSpace]
    transforms_used: list[ImageTransform2D]
    warnings: list[str]

    def resolve(self, source: PixelCoordinateSpace) -> ImageTransform2D:
        target = self.request.depth_input.depth_space
        transform, warnings = resolve_direct_transform(
            source,
            target,
            self.request.transforms,
            allow_implicit_resize=self.request.policy.allow_implicit_resize,
            interpolation=self.request.policy.rgb_interpolation.value,
        )
        self.warnings.extend(warnings)
        key = (transform.source_space.space_id, transform.destination_space.space_id)
        if not any(
            (value.source_space.space_id, value.destination_space.space_id) == key
            for value in self.transforms_used
        ):
            self.transforms_used.append(transform)
        return transform


def _collect_spaces(request: AlignmentRequest) -> dict[str, PixelCoordinateSpace]:
    spaces: dict[str, PixelCoordinateSpace] = {
        request.rgb_space.space_id: request.rgb_space,
        request.depth_input.depth_space.space_id: request.depth_input.depth_space,
    }
    for transform in request.transforms:
        for space in (transform.source_space, transform.destination_space):
            existing = spaces.get(space.space_id)
            if existing is not None and existing != space:
                raise ContractValidationError(
                    f"Pixel space ID {space.space_id!r} has conflicting declarations."
                )
            spaces[space.space_id] = space
    for observation in request.perception_result.observations:
        for mask in observation.masks:
            existing = spaces.get(mask.pixel_space.space_id)
            if existing is not None and existing != mask.pixel_space:
                raise ContractValidationError(
                    f"Pixel space ID {mask.pixel_space.space_id!r} has conflicting declarations."
                )
            spaces[mask.pixel_space.space_id] = mask.pixel_space
    return spaces


def _mask_border_contacts(mask: np.ndarray | None) -> tuple[str, ...]:
    if mask is None or not np.any(mask):
        return ()
    contacts: list[str] = []
    if np.any(mask[:, 0]):
        contacts.append("left")
    if np.any(mask[0, :]):
        contacts.append("top")
    if np.any(mask[:, -1]):
        contacts.append("right")
    if np.any(mask[-1, :]):
        contacts.append("bottom")
    return tuple(contacts)


def _align_observation(
    observation: ObjectObservation,
    context: _AlignmentContext,
) -> AlignedObjectObservation:
    target = context.request.depth_input.depth_space
    warnings: list[str] = []
    steps: list[ProcessingStep] = []
    aligned_box = None

    if observation.detection is not None:
        source_space = context.spaces.get(observation.detection.box.pixel_space_id)
        if source_space is None:
            raise ContractValidationError(
                f"Detection pixel space {observation.detection.box.pixel_space_id!r} is undeclared."
            )
        transform = context.resolve(source_space)
        try:
            aligned_box = transform_detection_box(
                observation.detection.box,
                source_space,
                target,
                transform,
            )
            if aligned_box.was_clipped:
                warnings.append("aligned_detection_box_clipped")
        except ContractValidationError:
            warnings.append("aligned_detection_box_empty")
        steps.append(
            ProcessingStep(
                step_id=f"{observation.observation_id}:detection-transform",
                method="homogeneous_box_corner_transform",
                input_ids=(observation.detection.detection_id,),
                output_ids=(f"{observation.observation_id}:aligned-box",),
                parameters={
                    "source_space_id": source_space.space_id,
                    "target_space_id": target.space_id,
                    "box_convention": "xyxy_exclusive",
                },
                warnings=tuple(warnings),
            )
        )

    selected_mask = observation.selected_mask
    source_mask_id = ""
    source_mask_space_id = ""
    aligned_mask = None
    source_pixels = 0
    aligned_pixels = 0
    retention = 0.0
    mask_source = None
    mask_revision = 0
    from core.perception.result import ApprovalStatus

    approval_status = (
        observation.masks[0].approval_status
        if observation.masks
        else ApprovalStatus.NOT_REQUIRED
    )
    mask_provenance = None

    if selected_mask is None:
        warnings.append("selected_mask_unavailable")
    else:
        source_mask_id = selected_mask.mask_id
        source_mask_space_id = selected_mask.pixel_space.space_id
        mask_source = selected_mask.source
        mask_revision = selected_mask.revision
        approval_status = selected_mask.approval_status
        mask_provenance = selected_mask.model_provenance
        try:
            dense = normalize_mask_payload(selected_mask.payload, selected_mask.pixel_space)
        except UnsupportedMaskRepresentationError:
            warnings.append(
                f"mask_representation_not_normalized:{selected_mask.representation.value}"
            )
        else:
            source_pixels = int(np.count_nonzero(dense))
            transform = context.resolve(selected_mask.pixel_space)
            if context.request.policy.warn_on_nonuniform_resize and is_nonuniform_scale(transform):
                warnings.append("nonuniform_mask_resize")
            aligned_mask = resample_boolean_mask(dense, transform)
            aligned_pixels = int(np.count_nonzero(aligned_mask))
            expected_pixels = source_pixels * linear_area_scale(transform)
            if expected_pixels > 0.0:
                retention = min(1.0, aligned_pixels / expected_pixels)
            if aligned_pixels == 0:
                warnings.append("aligned_mask_empty")
            steps.append(
                ProcessingStep(
                    step_id=f"{observation.observation_id}:mask-transform",
                    method="nearest_neighbour_boolean_warp",
                    input_ids=(selected_mask.mask_id,),
                    output_ids=(f"{observation.observation_id}:aligned-mask",),
                    parameters={
                        "source_space_id": selected_mask.pixel_space.space_id,
                        "target_space_id": target.space_id,
                        "source_foreground_pixels": source_pixels,
                        "aligned_foreground_pixels": aligned_pixels,
                        "mask_source": selected_mask.source.value,
                        "mask_revision": selected_mask.revision,
                    },
                    warnings=tuple(warnings),
                )
            )

    contacts = set(_mask_border_contacts(aligned_mask))
    if aligned_box is not None:
        if aligned_box.x_min <= 0.0:
            contacts.add("left")
        if aligned_box.y_min <= 0.0:
            contacts.add("top")
        if aligned_box.x_max >= target.size.width:
            contacts.add("right")
        if aligned_box.y_max >= target.size.height:
            contacts.add("bottom")

    return AlignedObjectObservation(
        observation_id=observation.observation_id,
        class_id=observation.class_id,
        class_name=observation.class_name,
        target_space=target,
        source_detection_id=(
            observation.detection.detection_id if observation.detection is not None else ""
        ),
        aligned_detection_box=aligned_box,
        source_mask_id=source_mask_id,
        source_mask_space_id=source_mask_space_id,
        aligned_mask=aligned_mask,
        source_foreground_pixels=source_pixels,
        aligned_foreground_pixels=aligned_pixels,
        foreground_retention_ratio=retention,
        border_contacts=tuple(value for value in ("left", "top", "right", "bottom") if value in contacts),
        mask_source=mask_source,
        mask_revision=mask_revision,
        approval_status=approval_status,
        mask_model_provenance=mask_provenance,
        detection_model_provenance=(
            observation.detection.model_provenance if observation.detection is not None else None
        ),
        processing_steps=tuple(steps),
        warnings=tuple(warnings),
    )


def align_frame(request: AlignmentRequest) -> AlignedFrameBundle:
    """Align one matching RGB/perception/depth frame onto native depth pixels."""

    validate_request_identity(request)
    context = _AlignmentContext(
        request=request,
        spaces=_collect_spaces(request),
        transforms_used=[],
        warnings=[],
    )
    target = request.depth_input.depth_space
    rgb_transform = context.resolve(request.rgb_space)
    if request.policy.warn_on_nonuniform_resize and is_nonuniform_scale(rgb_transform):
        context.warnings.append("nonuniform_rgb_resize")

    aligned_rgb = resample_rgb(
        request.rgb,
        rgb_transform,
        method=request.policy.rgb_interpolation,
    )
    rgb_support = resample_support_mask(rgb_transform)
    depth_valid = compose_depth_valid_mask(
        request.depth_input.depth,
        request.depth_input.valid_mask,
    )
    confidence_valid = compose_confidence_valid_mask(
        request.depth_input.confidence,
        request.depth_input.confidence_kind,
        request.depth_input.depth.shape,
    )
    aligned_observations = tuple(
        _align_observation(value, context)
        for value in request.perception_result.observations
    )
    if any(value.warnings for value in aligned_observations):
        status = AlignmentStatus.PARTIAL
    else:
        status = AlignmentStatus.SUCCESS

    processing_steps = (
        ProcessingStep(
            step_id=f"frame:{request.frame_reference.frame_id}:rgb-transform",
            method="rgb_to_native_depth_grid",
            input_ids=(request.rgb_space.space_id,),
            output_ids=(target.space_id,),
            parameters={
                "source_space_id": request.rgb_space.space_id,
                "target_space_id": target.space_id,
                "depth_resampled": False,
                "rgb_interpolation": request.policy.rgb_interpolation.value,
            },
            warnings=tuple(context.warnings),
        ),
        ProcessingStep(
            step_id=f"frame:{request.frame_reference.frame_id}:validity",
            method="finite_positive_depth_and_model_validity",
            input_ids=(target.space_id,),
            output_ids=(f"{target.space_id}:valid",),
            parameters={
                "valid_depth_pixels": int(np.count_nonzero(depth_valid)),
                "confidence_valid_pixels": int(np.count_nonzero(confidence_valid)),
            },
        ),
    )

    bundle = AlignedFrameBundle(
        frame_reference=request.frame_reference,
        target_depth_space=target,
        aligned_rgb=aligned_rgb,
        depth=request.depth_input.depth,
        depth_valid_mask=depth_valid,
        rgb_support_mask=rgb_support,
        confidence=request.depth_input.confidence,
        confidence_valid_mask=confidence_valid,
        confidence_kind=request.depth_input.confidence_kind,
        intrinsics=preserve_native_intrinsics(
            request.depth_input.intrinsics,
            target.space_id,
        ),
        depth_scale=request.depth_input.depth_scale,
        point_stride=request.depth_input.point_stride,
        depth_model_provenance=request.depth_input.depth_model_provenance,
        perception_model_provenance=unique_model_provenance(
            request.perception_result.model_provenance
        ),
        aligned_observations=aligned_observations,
        transforms=tuple(context.transforms_used),
        status=status,
        warnings=tuple(context.warnings),
        processing_steps=processing_steps,
    )
    validate_aligned_bundle(bundle)
    return bundle
