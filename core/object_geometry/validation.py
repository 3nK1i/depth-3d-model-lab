"""Authoritative validation helpers for Step 4 extraction requests."""

from __future__ import annotations

from core.alignment.contracts import AlignedFrameBundle
from core.contracts.validation import ContractValidationError


def validate_observation_selection(
    bundle: AlignedFrameBundle,
    observation_ids: tuple[str, ...],
) -> tuple[str, ...]:
    requested = tuple(str(value).strip() for value in observation_ids)
    if any(not value for value in requested):
        raise ContractValidationError("observation_ids must not contain empty values.")
    if len(requested) != len(set(requested)):
        raise ContractValidationError("observation_ids must be unique.")
    available = {value.observation_id for value in bundle.aligned_observations}
    missing = [value for value in requested if value not in available]
    if missing:
        raise ContractValidationError(f"Unknown observation IDs: {missing!r}.")
    return requested


def assert_native_depth_alignment(bundle: AlignedFrameBundle) -> None:
    shape = (bundle.target_depth_space.size.height, bundle.target_depth_space.size.width)
    if bundle.depth.shape != shape or bundle.depth_valid_mask.shape != shape:
        raise ContractValidationError("Aligned bundle depth arrays do not match target_depth_space.")
    if bundle.intrinsics.pixel_space_id != bundle.target_depth_space.space_id:
        raise ContractValidationError("Aligned bundle intrinsics do not target the depth space.")
