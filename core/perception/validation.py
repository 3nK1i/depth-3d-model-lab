"""Authoritative perception-contract validation entry points."""

from __future__ import annotations

from core.perception.result import (
    InstanceMask,
    MaskPayload,
    ObjectObservation,
    normalize_mask_payload,
    select_authoritative_mask,
)


def validate_instance_mask(mask: InstanceMask) -> InstanceMask:
    """Return a constructed mask after its immutable contract has validated."""

    if not isinstance(mask, InstanceMask):
        raise TypeError("mask must be an InstanceMask.")
    return mask


def validate_object_observation(observation: ObjectObservation) -> ObjectObservation:
    if not isinstance(observation, ObjectObservation):
        raise TypeError("observation must be an ObjectObservation.")
    select_authoritative_mask(observation.masks)
    return observation


__all__ = [
    "MaskPayload",
    "normalize_mask_payload",
    "select_authoritative_mask",
    "validate_instance_mask",
    "validate_object_observation",
]
