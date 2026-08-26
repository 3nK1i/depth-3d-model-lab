"""I2.7.1 detached-component qualification helpers.

This module is diagnostic-only.  It compares the original I2 Extended cloud
with the promoted I2.6/I2.7 Extended cloud using the exact same 26-neighbour
voxel connectivity.  It does not change geometry, filtering, persistence, or
any authority state.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Any

import numpy as np
from scipy.spatial import cKDTree

from core.reconstruction.errors import ReconstructionValidationError

I2_7_1_SCHEMA_VERSION = 1

_OFFSETS_26 = tuple(
    (dx, dy, dz)
    for dx in (-1, 0, 1)
    for dy in (-1, 0, 1)
    for dz in (-1, 0, 1)
    if not (dx == 0 and dy == 0 and dz == 0)
)


@dataclass(frozen=True, slots=True)
class ComponentLineage:
    """Aligned component labels/profile for one retained voxel cloud."""

    labels: np.ndarray
    component_sizes: np.ndarray
    dominant_label: int
    profile: dict[str, Any]


def _validate_points(points: object, voxel_size: float, *, name: str) -> tuple[np.ndarray, float, np.ndarray, list[tuple[int, int, int]]]:
    xyz = np.asarray(points, dtype=np.float64)
    voxel = float(voxel_size)
    if xyz.ndim != 2 or xyz.shape[1] != 3 or len(xyz) == 0:
        raise ReconstructionValidationError(f"{name} requires non-empty Nx3 geometry.")
    if not isfinite(voxel) or voxel <= 0.0 or not np.isfinite(xyz).all():
        raise ReconstructionValidationError(f"{name} received invalid geometry/voxel size.")
    key_array = np.floor(xyz / voxel).astype(np.int64)
    keys = [tuple(map(int, row)) for row in key_array]
    if len(set(keys)) != len(keys):
        raise ReconstructionValidationError(f"{name} expects exactly one retained point per voxel key.")
    return xyz, voxel, key_array, keys


def component_lineage(points: object, voxel_size: float) -> ComponentLineage:
    """Compute 26-neighbour component labels aligned to input points."""

    xyz, _, _, keys = _validate_points(points, voxel_size, name="I2.7.1 component analysis")
    key_to_index = {key: index for index, key in enumerate(keys)}
    unseen = set(keys)
    labels = np.full(len(keys), -1, dtype=np.int32)
    sizes: list[int] = []

    while unseen:
        seed = unseen.pop()
        stack = [seed]
        members: list[int] = []
        while stack:
            key = stack.pop()
            members.append(key_to_index[key])
            x, y, z = key
            for dx, dy, dz in _OFFSETS_26:
                candidate = (x + dx, y + dy, z + dz)
                if candidate in unseen:
                    unseen.remove(candidate)
                    stack.append(candidate)
        label = len(sizes)
        labels[np.asarray(members, dtype=np.int64)] = label
        sizes.append(len(members))

    size_array = np.asarray(sizes, dtype=np.int64)
    dominant = int(np.argmax(size_array))
    sorted_sizes = np.sort(size_array)[::-1]
    largest = int(size_array[dominant])
    profile = {
        "component_count": int(len(size_array)),
        "largest_component_voxels": largest,
        "largest_component_fraction": float(largest / len(xyz)),
        "detached_voxels": int(len(xyz) - largest),
        "top_component_sizes": [int(v) for v in sorted_sizes[:10]],
        "retained_voxels": int(len(xyz)),
        "connectivity": 26,
    }
    return ComponentLineage(labels=labels, component_sizes=size_array, dominant_label=dominant, profile=profile)


def mask_statistics(
    mask: object,
    *,
    is_core: object,
    peripheral: object,
    far: object,
    weak: object,
    support: object,
    confidence: object,
    dispersion: object,
) -> dict[str, Any]:
    """Compact evidence summary for a component subset."""

    m = np.asarray(mask, dtype=bool)
    arrays = {
        "is_core": np.asarray(is_core, dtype=bool),
        "peripheral": np.asarray(peripheral, dtype=bool),
        "far": np.asarray(far, dtype=bool),
        "weak": np.asarray(weak, dtype=bool),
        "support": np.asarray(support, dtype=np.float64),
        "confidence": np.asarray(confidence, dtype=np.float64),
        "dispersion": np.asarray(dispersion, dtype=np.float64),
    }
    if any(value.shape != m.shape for value in arrays.values()):
        raise ReconstructionValidationError("I2.7.1 mask-statistic fields are not aligned.")
    count = int(np.count_nonzero(m))
    if count == 0:
        return {
            "count": 0,
            "core_fraction": 0.0,
            "peripheral_fraction": 0.0,
            "far_fraction": 0.0,
            "weak_fraction": 0.0,
            "support_quantiles": {"p50": None, "p90": None},
            "confidence_quantiles": {"p50": None, "p90": None},
            "dispersion_quantiles": {"p50": None, "p90": None},
        }

    def q(values: np.ndarray) -> dict[str, float]:
        selected = values[m]
        return {"p50": float(np.quantile(selected, 0.50)), "p90": float(np.quantile(selected, 0.90))}

    return {
        "count": count,
        "core_fraction": float(np.mean(arrays["is_core"][m])),
        "peripheral_fraction": float(np.mean(arrays["peripheral"][m])),
        "far_fraction": float(np.mean(arrays["far"][m])),
        "weak_fraction": float(np.mean(arrays["weak"][m])),
        "support_quantiles": q(arrays["support"]),
        "confidence_quantiles": q(arrays["confidence"]),
        "dispersion_quantiles": q(arrays["dispersion"]),
    }


def detached_distance_statistics(points: object, dominant_mask: object, detached_mask: object, voxel_size: float) -> dict[str, Any]:
    """Nearest-dominant distance for detached points, reported in voxel units."""

    xyz, voxel, _, _ = _validate_points(points, voxel_size, name="I2.7.1 detached-distance analysis")
    dominant = np.asarray(dominant_mask, dtype=bool)
    detached = np.asarray(detached_mask, dtype=bool)
    if dominant.shape != (len(xyz),) or detached.shape != (len(xyz),):
        raise ReconstructionValidationError("I2.7.1 distance masks are not aligned.")
    if not np.any(dominant):
        raise ReconstructionValidationError("I2.7.1 requires a non-empty dominant component.")
    count = int(np.count_nonzero(detached))
    if count == 0:
        return {"count": 0, "p50_vox": None, "p90_vox": None, "max_vox": None}
    tree = cKDTree(xyz[dominant])
    distances, _ = tree.query(xyz[detached], k=1, workers=-1)
    values = np.asarray(distances, dtype=np.float64) / voxel
    return {
        "count": count,
        "p50_vox": float(np.quantile(values, 0.50)),
        "p90_vox": float(np.quantile(values, 0.90)),
        "max_vox": float(np.max(values)),
    }


def classify_component_gate_failure(*, baseline_fraction: float, promoted_fraction: float, threshold: float, newly_detached_voxels: int) -> str:
    """Describe the observed origin of the absolute component-gate failure."""

    for name, value in (("baseline_fraction", baseline_fraction), ("promoted_fraction", promoted_fraction), ("threshold", threshold)):
        if not isfinite(float(value)) or float(value) < 0.0 or float(value) > 1.0:
            raise ReconstructionValidationError(f"I2.7.1 {name} must be finite in [0, 1].")
    if int(newly_detached_voxels) < 0:
        raise ReconstructionValidationError("I2.7.1 newly_detached_voxels cannot be negative.")
    baseline_pass = float(baseline_fraction) >= float(threshold)
    promoted_pass = float(promoted_fraction) >= float(threshold)
    if baseline_pass and not promoted_pass:
        return "promotion_induced_threshold_crossing"
    if not baseline_pass and not promoted_pass:
        return "preexisting_baseline_fragmentation" if int(newly_detached_voxels) == 0 else "preexisting_plus_promotion_fragmentation"
    if baseline_pass and promoted_pass:
        return "absolute_gate_not_failed"
    return "promoted_recovers_baseline_failure"


def compare_component_lineage(
    *,
    baseline_points: object,
    promoted_points: object,
    voxel_size: float,
) -> dict[str, Any]:
    """Exact baseline→promoted component lineage using retained voxel keys."""

    base_xyz, voxel, _, base_keys = _validate_points(baseline_points, voxel_size, name="I2.7.1 baseline")
    promoted_xyz, _, _, promoted_keys = _validate_points(promoted_points, voxel_size, name="I2.7.1 promoted")
    base_lookup = {key: index for index, key in enumerate(base_keys)}
    promoted_lookup = {key: index for index, key in enumerate(promoted_keys)}
    base_set, promoted_set = set(base_keys), set(promoted_keys)
    if not promoted_set.issubset(base_set):
        raise ReconstructionValidationError("I2.7.1 promoted geometry is not an exact voxel-key subset of baseline I2 geometry.")

    base_lineage = component_lineage(base_xyz, voxel)
    promoted_lineage = component_lineage(promoted_xyz, voxel)
    base_dom = base_lineage.labels == base_lineage.dominant_label
    promoted_dom = promoted_lineage.labels == promoted_lineage.dominant_label

    promoted_baseline_indices = np.asarray([base_lookup[key] for key in promoted_keys], dtype=np.int64)
    baseline_dom_on_promoted = base_dom[promoted_baseline_indices]
    newly_detached = baseline_dom_on_promoted & ~promoted_dom
    surviving_preexisting_detached = ~baseline_dom_on_promoted & ~promoted_dom
    promoted_detached = ~promoted_dom
    removed_keys = base_set - promoted_set
    removed_indices = np.asarray([base_lookup[key] for key in removed_keys], dtype=np.int64)
    removed_from_baseline_dominant = int(np.count_nonzero(base_dom[removed_indices])) if len(removed_indices) else 0
    removed_from_preexisting_detached = int(len(removed_indices) - removed_from_baseline_dominant)

    # A deletion cannot connect voxel components.  Therefore the promoted dominant
    # component must lie inside one baseline component; record its dominant-lineage overlap.
    promoted_dom_baseline_labels = base_lineage.labels[promoted_baseline_indices[promoted_dom]]
    unique_labels, counts = np.unique(promoted_dom_baseline_labels, return_counts=True)
    lineage_label = int(unique_labels[int(np.argmax(counts))])
    dominant_lineage_fraction = float(np.max(counts) / np.sum(counts))

    # Component-level bridge evidence: newly detached promoted components that
    # contain voxels formerly in the baseline dominant component and touch at
    # least one d12-removed voxel under 26-neighbour connectivity.
    removed_set = removed_keys
    promoted_component_records: list[dict[str, Any]] = []
    for label, size in enumerate(promoted_lineage.component_sizes.tolist()):
        if label == promoted_lineage.dominant_label:
            continue
        members = np.flatnonzero(promoted_lineage.labels == label)
        was_base_dom = baseline_dom_on_promoted[members]
        new_count = int(np.count_nonzero(was_base_dom))
        if new_count == 0:
            origin = "preexisting_detached"
        elif new_count == len(members):
            origin = "newly_detached_from_baseline_dominant"
        else:
            origin = "mixed_lineage"
        touching_removed = 0
        if removed_set and new_count:
            for member in members:
                if not baseline_dom_on_promoted[member]:
                    continue
                x, y, z = promoted_keys[int(member)]
                if any((x + dx, y + dy, z + dz) in removed_set for dx, dy, dz in _OFFSETS_26):
                    touching_removed += 1
        promoted_component_records.append({
            "label": int(label),
            "size": int(size),
            "origin": origin,
            "newly_detached_voxels": new_count,
            "touching_removed_voxels": int(touching_removed),
        })
    promoted_component_records.sort(key=lambda item: item["size"], reverse=True)

    expected_promoted_detached = int(np.count_nonzero(promoted_detached))
    decomposition = int(np.count_nonzero(newly_detached)) + int(np.count_nonzero(surviving_preexisting_detached))
    if decomposition != expected_promoted_detached:
        raise ReconstructionValidationError("I2.7.1 promoted detached lineage accounting is not exact.")

    return {
        "baseline": base_lineage.profile,
        "promoted": promoted_lineage.profile,
        "removed_voxels": int(len(removed_set)),
        "removed_from_baseline_dominant": removed_from_baseline_dominant,
        "removed_from_preexisting_detached": removed_from_preexisting_detached,
        "newly_detached_voxels": int(np.count_nonzero(newly_detached)),
        "surviving_preexisting_detached_voxels": int(np.count_nonzero(surviving_preexisting_detached)),
        "promoted_detached_voxels": expected_promoted_detached,
        "largest_component_fraction_delta": float(
            promoted_lineage.profile["largest_component_fraction"] - base_lineage.profile["largest_component_fraction"]
        ),
        "detached_voxel_delta": int(promoted_lineage.profile["detached_voxels"] - base_lineage.profile["detached_voxels"]),
        "dominant_component_lineage_label": lineage_label,
        "promoted_dominant_from_single_baseline_component_fraction": dominant_lineage_fraction,
        "promoted_component_records": promoted_component_records,
        "masks": {
            "baseline_dominant": base_dom,
            "promoted_dominant": promoted_dom,
            "newly_detached": newly_detached,
            "surviving_preexisting_detached": surviving_preexisting_detached,
            "promoted_baseline_indices": promoted_baseline_indices,
            "removed_baseline_indices": removed_indices,
        },
    }
