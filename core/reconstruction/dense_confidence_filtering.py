"""Confidence, consistency and artifact filtering for fused dense geometry.

Phase F formalizes the evidence model first proven by the E2 validation stage
and applies it to an expanded E3 fused scene.  This module is deliberately
independent of Qt, DA3 and PyCOLMAP.  It consumes per-voxel evidence already
expressed in one reconstruction gauge and returns deterministic masks/scores.

It does not infer metric scale, certify measurements, persist canonical scene
state, or generate a surface mesh.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import IntFlag
from math import isfinite
from typing import Any

import numpy as np

from core.reconstruction.errors import ReconstructionValidationError

DENSE_CONFIDENCE_FILTER_SCHEMA_VERSION = 1


class DenseArtifactReason(IntFlag):
    """Bit flags explaining why a supported voxel was excluded from Extended."""

    NONE = 0
    SMALL_COMPONENT = 1
    EXCESSIVE_DISPERSION = 2
    WEAK_LOCAL_CONNECTIVITY = 4
    WEAK_HIGH_SUPPORT_ANCHOR = 8
    SINGLE_VIEW_DETACHED = 16
    WEAK_MANIFOLD_ATTACHMENT = 32


@dataclass(frozen=True, slots=True)
class DenseConfidenceFilterPolicy:
    """Deterministic filtering policy in units of the fused voxel lattice."""

    maximum_views: int
    minimum_component_size: int = 64
    core_max_dispersion_voxels: float = 0.70
    support2_core_max_dispersion_voxels: float = 0.45
    extended_max_dispersion_voxels: float = 0.95
    support2_core_min_high_neighbors: int = 2
    support2_core_min_radius2_neighbors: int = 8
    extended_min_radius2_neighbors: int = 6
    support2_extended_min_high_neighbors: int = 1
    support2_extended_min_radius2_neighbors: int = 8

    def __post_init__(self) -> None:
        maximum = int(self.maximum_views)
        if maximum < 2:
            raise ReconstructionValidationError("maximum_views must be >= 2.")
        object.__setattr__(self, "maximum_views", maximum)
        component = int(self.minimum_component_size)
        if component <= 0:
            raise ReconstructionValidationError("minimum_component_size must be positive.")
        object.__setattr__(self, "minimum_component_size", component)
        for name in (
            "core_max_dispersion_voxels",
            "support2_core_max_dispersion_voxels",
            "extended_max_dispersion_voxels",
        ):
            value = float(getattr(self, name))
            if not isfinite(value) or value <= 0.0 or value > 2.0:
                raise ReconstructionValidationError(f"{name} must be finite in (0, 2].")
            object.__setattr__(self, name, value)
        if self.support2_core_max_dispersion_voxels > self.core_max_dispersion_voxels:
            raise ReconstructionValidationError(
                "support2_core_max_dispersion_voxels cannot exceed core_max_dispersion_voxels."
            )
        if self.core_max_dispersion_voxels > self.extended_max_dispersion_voxels:
            raise ReconstructionValidationError(
                "core_max_dispersion_voxels cannot exceed extended_max_dispersion_voxels."
            )
        for name in (
            "support2_core_min_high_neighbors",
            "support2_core_min_radius2_neighbors",
            "extended_min_radius2_neighbors",
            "support2_extended_min_high_neighbors",
            "support2_extended_min_radius2_neighbors",
        ):
            value = int(getattr(self, name))
            if value < 0:
                raise ReconstructionValidationError(f"{name} must be non-negative.")
            object.__setattr__(self, name, value)

    def as_dict(self) -> dict[str, Any]:
        return {
            "maximum_views": self.maximum_views,
            "minimum_component_size": self.minimum_component_size,
            "core_max_dispersion_voxels": self.core_max_dispersion_voxels,
            "support2_core_max_dispersion_voxels": self.support2_core_max_dispersion_voxels,
            "extended_max_dispersion_voxels": self.extended_max_dispersion_voxels,
            "support2_core_min_high_neighbors": self.support2_core_min_high_neighbors,
            "support2_core_min_radius2_neighbors": self.support2_core_min_radius2_neighbors,
            "extended_min_radius2_neighbors": self.extended_min_radius2_neighbors,
            "support2_extended_min_high_neighbors": self.support2_extended_min_high_neighbors,
            "support2_extended_min_radius2_neighbors": self.support2_extended_min_radius2_neighbors,
        }


@dataclass(frozen=True, slots=True)
class DenseConfidenceFilterResult:
    """Per-supported-voxel confidence, class masks and artifact reasons."""

    confidence: np.ndarray
    core_mask: np.ndarray
    extended_mask: np.ndarray
    rejected_mask: np.ndarray
    artifact_reason_code: np.ndarray

    def __post_init__(self) -> None:
        confidence = np.asarray(self.confidence, dtype=np.float32)
        core = np.asarray(self.core_mask, dtype=bool)
        extended = np.asarray(self.extended_mask, dtype=bool)
        rejected = np.asarray(self.rejected_mask, dtype=bool)
        reasons = np.asarray(self.artifact_reason_code, dtype=np.uint16)
        shape = confidence.shape
        if confidence.ndim != 1 or not (
            core.shape == extended.shape == rejected.shape == reasons.shape == shape
        ):
            raise ReconstructionValidationError("Dense confidence filter result arrays must be aligned 1D vectors.")
        if np.any(~np.isfinite(confidence)) or np.any((confidence < 0.0) | (confidence > 1.0)):
            raise ReconstructionValidationError("Dense confidence values must be finite in [0, 1].")
        if np.any(core & ~extended):
            raise ReconstructionValidationError("Core geometry must be a subset of Extended geometry.")
        if np.any(rejected != ~extended):
            raise ReconstructionValidationError("Rejected mask must be the complement of Extended.")
        if np.any((reasons == 0) & rejected):
            raise ReconstructionValidationError("Every rejected supported voxel must carry an artifact reason.")
        if np.any((reasons != 0) & ~rejected):
            raise ReconstructionValidationError("Retained supported voxels must not carry rejection reasons.")
        object.__setattr__(self, "confidence", confidence)
        object.__setattr__(self, "core_mask", core)
        object.__setattr__(self, "extended_mask", extended)
        object.__setattr__(self, "rejected_mask", rejected)
        object.__setattr__(self, "artifact_reason_code", reasons)


def _validate_evidence(
    support_counts: object,
    p90_dispersion_voxels: object,
    radius2_neighbors: object,
    high_support_neighbors: object,
    component_sizes: object,
    *,
    maximum_views: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    support = np.asarray(support_counts, dtype=np.int64)
    dispersion = np.asarray(p90_dispersion_voxels, dtype=np.float64)
    radius2 = np.asarray(radius2_neighbors, dtype=np.int64)
    high = np.asarray(high_support_neighbors, dtype=np.int64)
    components = np.asarray(component_sizes, dtype=np.int64)
    if support.ndim != 1 or not (
        support.shape == dispersion.shape == radius2.shape == high.shape == components.shape
    ):
        raise ReconstructionValidationError("Dense confidence evidence must be aligned 1D vectors.")
    if len(support) == 0:
        raise ReconstructionValidationError("Dense confidence filtering requires supported voxels.")
    if np.any((support < 2) | (support > int(maximum_views))):
        raise ReconstructionValidationError("Supported voxel counts must be in [2, maximum_views].")
    if np.any(~np.isfinite(dispersion)) or np.any(dispersion < 0.0):
        raise ReconstructionValidationError("Dispersion must be finite and non-negative.")
    if np.any(radius2 < 0) or np.any(high < 0) or np.any(components <= 0):
        raise ReconstructionValidationError("Topology evidence is invalid.")
    return support, dispersion, radius2, high, components


def dense_filter_confidence(
    support_counts: object,
    p90_dispersion_voxels: object,
    radius2_neighbors: object,
    high_support_neighbors: object,
    *,
    policy: DenseConfidenceFilterPolicy,
) -> np.ndarray:
    """Return a diagnostic confidence score without using it as the only hard gate."""

    if not isinstance(policy, DenseConfidenceFilterPolicy):
        raise TypeError("policy must be DenseConfidenceFilterPolicy.")
    support = np.asarray(support_counts, dtype=np.float64)
    dispersion = np.asarray(p90_dispersion_voxels, dtype=np.float64)
    radius2 = np.asarray(radius2_neighbors, dtype=np.float64)
    high = np.asarray(high_support_neighbors, dtype=np.float64)
    if support.ndim != 1 or not (support.shape == dispersion.shape == radius2.shape == high.shape):
        raise ReconstructionValidationError("Dense confidence inputs must be aligned 1D vectors.")
    if len(support) == 0:
        return np.empty((0,), dtype=np.float32)
    if np.any((support < 2) | (support > policy.maximum_views)):
        raise ReconstructionValidationError("Dense confidence support is outside the policy view range.")
    if np.any(~np.isfinite(dispersion)) or np.any(dispersion < 0.0):
        raise ReconstructionValidationError("Dense confidence dispersion is invalid.")

    # Unlike the E2 proof, support does not saturate at five observations.  The
    # score stays meaningful when E3 expands to tens of qualified views.
    denominator = max(float(policy.maximum_views - 1), 1.0)
    support_score = np.sqrt(np.clip((support - 1.0) / denominator, 0.0, 1.0))
    dispersion_score = np.clip(1.0 - dispersion / policy.extended_max_dispersion_voxels, 0.0, 1.0)
    density_score = np.clip(radius2 / 24.0, 0.0, 1.0)
    strong_neighbor_score = np.clip(high / 6.0, 0.0, 1.0)
    return np.clip(
        0.35 * support_score
        + 0.35 * dispersion_score
        + 0.20 * density_score
        + 0.10 * strong_neighbor_score,
        0.0,
        1.0,
    ).astype(np.float32)


def filter_dense_supported_geometry(
    support_counts: object,
    p90_dispersion_voxels: object,
    radius2_neighbors: object,
    high_support_neighbors: object,
    component_sizes: object,
    *,
    policy: DenseConfidenceFilterPolicy,
) -> DenseConfidenceFilterResult:
    """Classify supported fused voxels into Core, Extended, or rejected artifact."""

    if not isinstance(policy, DenseConfidenceFilterPolicy):
        raise TypeError("policy must be DenseConfidenceFilterPolicy.")
    support, dispersion, radius2, high, components = _validate_evidence(
        support_counts,
        p90_dispersion_voxels,
        radius2_neighbors,
        high_support_neighbors,
        component_sizes,
        maximum_views=policy.maximum_views,
    )

    connected = components >= policy.minimum_component_size
    core_high_support = (
        (support >= 3)
        & (dispersion <= policy.core_max_dispersion_voxels)
        & connected
        & (radius2 >= max(2, policy.extended_min_radius2_neighbors // 2))
    )
    core_support2 = (
        (support == 2)
        & (dispersion <= policy.support2_core_max_dispersion_voxels)
        & connected
        & (high >= policy.support2_core_min_high_neighbors)
        & (radius2 >= policy.support2_core_min_radius2_neighbors)
    )
    core = core_high_support | core_support2

    general_extended = (
        (support >= 3)
        & (dispersion <= policy.extended_max_dispersion_voxels)
        & connected
        & (radius2 >= policy.extended_min_radius2_neighbors)
    )
    support2_extended = (
        (support == 2)
        & (dispersion <= policy.extended_max_dispersion_voxels)
        & connected
        & (radius2 >= policy.support2_extended_min_radius2_neighbors)
        & (high >= policy.support2_extended_min_high_neighbors)
    )
    extended = general_extended | support2_extended | core
    rejected = ~extended

    reasons = np.zeros(len(support), dtype=np.uint16)
    reasons[rejected & ~connected] |= int(DenseArtifactReason.SMALL_COMPONENT)
    reasons[rejected & (dispersion > policy.extended_max_dispersion_voxels)] |= int(
        DenseArtifactReason.EXCESSIVE_DISPERSION
    )
    required_radius2 = np.where(
        support == 2,
        policy.support2_extended_min_radius2_neighbors,
        policy.extended_min_radius2_neighbors,
    )
    reasons[rejected & (radius2 < required_radius2)] |= int(DenseArtifactReason.WEAK_LOCAL_CONNECTIVITY)
    reasons[
        rejected
        & (support == 2)
        & (high < policy.support2_extended_min_high_neighbors)
    ] |= int(DenseArtifactReason.WEAK_HIGH_SUPPORT_ANCHOR)

    # Defensive fallback: if a future policy creates a rejection not represented
    # by one of the explicit bits above, retain a deterministic reason rather than
    # allowing an unexplained artifact classification.
    unexplained = rejected & (reasons == 0)
    reasons[unexplained] = int(DenseArtifactReason.WEAK_LOCAL_CONNECTIVITY)

    confidence = dense_filter_confidence(
        support,
        dispersion,
        radius2,
        high,
        policy=policy,
    )
    return DenseConfidenceFilterResult(
        confidence=confidence,
        core_mask=core,
        extended_mask=extended,
        rejected_mask=rejected,
        artifact_reason_code=reasons,
    )


def artifact_reason_histogram(reason_codes: object) -> dict[str, int]:
    """Return per-bit rejection counts for diagnostics and persisted provenance."""

    values = np.asarray(reason_codes, dtype=np.uint16)
    if values.ndim != 1:
        raise ReconstructionValidationError("Artifact reason codes must be a 1D vector.")
    return {
        "small_component": int(np.count_nonzero(values & int(DenseArtifactReason.SMALL_COMPONENT))),
        "excessive_dispersion": int(np.count_nonzero(values & int(DenseArtifactReason.EXCESSIVE_DISPERSION))),
        "weak_local_connectivity": int(np.count_nonzero(values & int(DenseArtifactReason.WEAK_LOCAL_CONNECTIVITY))),
        "weak_high_support_anchor": int(np.count_nonzero(values & int(DenseArtifactReason.WEAK_HIGH_SUPPORT_ANCHOR))),
        "single_view_detached": int(np.count_nonzero(values & int(DenseArtifactReason.SINGLE_VIEW_DETACHED))),
        "weak_manifold_attachment": int(np.count_nonzero(values & int(DenseArtifactReason.WEAK_MANIFOLD_ATTACHMENT))),
    }
