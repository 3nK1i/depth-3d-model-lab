"""I4.2E TempleRing thin-structure / detail-preservation qualification.

I4.2E is a diagnostic counterfactual.  It consumes the immutable raw COLMAP
MVS result from I4.2D and rebuilds scene-relative multi-view support evidence
from COLMAP's native ``fused.ply.vis`` visibility sidecar emitted with the raw
``fused.ply``.  The exact locked production ``d12_sparse`` predicate is then
evaluated on a production-scale voxelized Extended analogue.

The transfer deliberately separates two questions:

* does the existing d12 predicate reject any MVS-supported object detail; and
* does that rejection look like legitimate thin structure under matched visual
  review?

The first can be described automatically.  The second remains a manual visual
review decision because TempleRing does not ship a benchmark laser surface in
this project and the d12 rule was originally qualified on room-scale F geometry.
No production policy is changed here.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import hashlib
from math import isfinite
from typing import Any, Mapping

import numpy as np

from core.reconstruction.dense_confidence_filtering import (
    DenseConfidenceFilterPolicy,
    filter_dense_supported_geometry,
)
from core.reconstruction.dense_manifold_refinement import DenseManifoldRefinementPolicy
from core.reconstruction.errors import ReconstructionValidationError
from core.reconstruction.offline_dense_extremum_local_geometry import (
    I252LocalGeometryPolicy,
    local_shape_metrics,
)
from core.reconstruction.offline_dense_i3_transfer import build_scene_relative_manifold_evidence
from core.reconstruction.offline_dense_quality import I2QualityPolicy, spatial_envelope

I42E_SCHEMA_VERSION = 1
I42E_PHASE = "I4.2E"
PRODUCTION_VOXEL_SCENE_FRACTION = 0.004

_NEIGHBOR26 = tuple(
    (dx, dy, dz)
    for dx in (-1, 0, 1)
    for dy in (-1, 0, 1)
    for dz in (-1, 0, 1)
    if (dx, dy, dz) != (0, 0, 0)
)
_NEIGHBOR_R2 = tuple(
    (dx, dy, dz)
    for dx in range(-2, 3)
    for dy in range(-2, 3)
    for dz in range(-2, 3)
    if (dx, dy, dz) != (0, 0, 0)
)


@dataclass(frozen=True, slots=True)
class TempleRingDetailTransferPolicy:
    """Frozen production-scale evidence reconstruction used by I4.2E.

    These values mirror the established I1/E2 evidence scale and classification
    defaults.  I4.2E does not expose them as tuning controls.
    """

    voxel_scene_fraction: float = PRODUCTION_VOXEL_SCENE_FRACTION
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
        locked = TempleRingDetailTransferPolicy.__dataclass_fields__
        defaults = {name: field.default for name, field in locked.items()}
        for name, expected in defaults.items():
            actual = getattr(self, name)
            if isinstance(expected, float):
                if not isfinite(float(actual)) or not np.isclose(float(actual), float(expected), rtol=0.0, atol=1e-12):
                    raise ReconstructionValidationError(f"I4.2E {name} is locked to {expected!r}.")
            elif int(actual) != int(expected):
                raise ReconstructionValidationError(f"I4.2E {name} is locked to {expected!r}.")

    def as_dict(self) -> dict[str, Any]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}


def _xyz(points: object, *, label: str) -> np.ndarray:
    xyz = np.asarray(points, dtype=np.float64)
    if xyz.ndim != 2 or xyz.shape[1] != 3 or len(xyz) == 0 or not np.isfinite(xyz).all():
        raise ReconstructionValidationError(f"I4.2E {label} must be finite non-empty Nx3 geometry.")
    return xyz


def _rgb(colors: object, length: int) -> np.ndarray:
    rgb = np.asarray(colors, dtype=np.uint8)
    if rgb.shape != (int(length), 3):
        raise ReconstructionValidationError("I4.2E RGB evidence must align Nx3 with geometry.")
    return rgb


def canonical_cloud_digest(points: object, colors: object) -> str:
    """Order-independent digest of float32 XYZ + uint8 RGB point records."""
    xyz = _xyz(points, label="cloud digest points").astype(np.float32)
    rgb = _rgb(colors, len(xyz))
    xyz[xyz == 0.0] = 0.0  # canonicalise signed zero
    order = np.lexsort((rgb[:, 2], rgb[:, 1], rgb[:, 0], xyz[:, 2], xyz[:, 1], xyz[:, 0]))
    dtype = np.dtype([
        ("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
        ("r", "u1"), ("g", "u1"), ("b", "u1"),
    ])
    rows = np.empty(len(xyz), dtype=dtype)
    sorted_xyz = xyz[order]
    sorted_rgb = rgb[order]
    rows["x"], rows["y"], rows["z"] = sorted_xyz[:, 0], sorted_xyz[:, 1], sorted_xyz[:, 2]
    rows["r"], rows["g"], rows["b"] = sorted_rgb[:, 0], sorted_rgb[:, 1], sorted_rgb[:, 2]
    return hashlib.sha256(rows.tobytes()).hexdigest()


def validate_fusion_visibility_evidence(
    points: object,
    colors: object,
    visibility_offsets: object,
    visibility_image_indices: object,
    *,
    expected_views: int = 47,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Validate native COLMAP fused-point visibility provenance.

    ``fused.ply.vis`` stores zero-based MVS image *indices*, not COLMAP image
    IDs.  I4.2E uses those indices only as stable per-view identities for
    support aggregation; no pose or image-specific semantic is inferred.
    The observed global index set may be any non-empty subset of the campaign
    view range; full-set coverage is not a native visibility validity rule.
    """
    xyz = _xyz(points, label="fusion-visibility points")
    rgb = _rgb(colors, len(xyz))
    offsets = np.asarray(visibility_offsets, dtype=np.int64).reshape(-1)
    image_indices = np.asarray(visibility_image_indices, dtype=np.int64).reshape(-1)
    if offsets.shape != (len(xyz) + 1,) or offsets[0] != 0 or offsets[-1] != len(image_indices):
        raise ReconstructionValidationError("I4.2E visibility offsets are malformed or misaligned.")
    if np.any(np.diff(offsets) <= 0):
        raise ReconstructionValidationError("I4.2E every fused point must retain at least one native visibility image.")
    if np.any(image_indices < 0) or np.any(image_indices >= int(expected_views)):
        raise ReconstructionValidationError(
            f"I4.2E native visibility image indices must lie in [0, {int(expected_views) - 1}]."
        )
    # Global coverage of all campaign views is deliberately not required.
    # ``fused.ply.vis`` records which views support each surviving fused point;
    # an MVS reference image can legitimately contribute no surviving support
    # entry after geometric-consistency/fusion filtering.
    for index in range(len(xyz)):
        start, stop = int(offsets[index]), int(offsets[index + 1])
        segment = image_indices[start:stop]
        if len(np.unique(segment)) != len(segment):
            raise ReconstructionValidationError(
                "I4.2E native visibility contains duplicate image indices for one fused point."
            )
    return xyz, rgb, offsets, image_indices


def validate_fusion_track_evidence(
    points: object,
    colors: object,
    track_offsets: object,
    track_image_ids: object,
    *,
    expected_views: int = 47,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compatibility wrapper for the pre-I4.2E.2 helper name.

    The payload is now native COLMAP visibility, so callers must pass zero-based
    MVS image indices despite the historical ``track_image_ids`` parameter name.
    """
    return validate_fusion_visibility_evidence(
        points, colors, track_offsets, track_image_ids, expected_views=expected_views
    )


def _topology_metrics(voxel_keys: np.ndarray, support_counts: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    keys = np.asarray(voxel_keys, dtype=np.int64)
    support = np.asarray(support_counts, dtype=np.int64).reshape(-1)
    if keys.ndim != 2 or keys.shape != (len(support), 3) or len(keys) == 0:
        raise ReconstructionValidationError("I4.2E topology requires aligned non-empty voxel keys/support.")
    key_to_index = {tuple(map(int, key)): index for index, key in enumerate(keys)}
    n26 = np.zeros(len(keys), dtype=np.int32)
    nr2 = np.zeros(len(keys), dtype=np.int32)
    high26 = np.zeros(len(keys), dtype=np.int32)
    for index, key_arr in enumerate(keys):
        key = tuple(map(int, key_arr))
        for delta in _NEIGHBOR26:
            other = key_to_index.get((key[0] + delta[0], key[1] + delta[1], key[2] + delta[2]))
            if other is not None:
                n26[index] += 1
                if support[other] >= 3:
                    high26[index] += 1
        for delta in _NEIGHBOR_R2:
            if (key[0] + delta[0], key[1] + delta[1], key[2] + delta[2]) in key_to_index:
                nr2[index] += 1

    labels = np.full(len(keys), -1, dtype=np.int32)
    component_sizes: list[int] = []
    component_id = 0
    for start in range(len(keys)):
        if labels[start] >= 0:
            continue
        labels[start] = component_id
        queue: deque[int] = deque([start])
        members = 0
        while queue:
            current = queue.popleft()
            members += 1
            key = tuple(map(int, keys[current]))
            for delta in _NEIGHBOR26:
                other = key_to_index.get((key[0] + delta[0], key[1] + delta[1], key[2] + delta[2]))
                if other is not None and labels[other] < 0:
                    labels[other] = component_id
                    queue.append(other)
        component_sizes.append(members)
        component_id += 1
    sizes = np.asarray(component_sizes, dtype=np.int32)
    per_point = sizes[labels]
    return n26, nr2, high26, per_point


def build_production_scale_mvs_payload(
    points: object,
    colors: object,
    track_offsets: object,
    track_image_ids: object,
    *,
    policy: TempleRingDetailTransferPolicy = TempleRingDetailTransferPolicy(),
) -> dict[str, Any]:
    """Build a production-scale Extended analogue from native MVS visibility.

    The evidence reconstruction intentionally mirrors the established I1/E2
    scale and support/topology definitions, while recording that source support
    comes from COLMAP's native fused-point visibility rather than the DA3 F
    pipeline.
    """
    xyz, rgb, offsets, image_ids = validate_fusion_visibility_evidence(
        points, colors, track_offsets, track_image_ids
    )
    envelope = spatial_envelope(xyz, lower_quantile=0.02, upper_quantile=0.98)
    scene_scale = float(envelope.robust_diagonal)
    if not isfinite(scene_scale) or scene_scale <= 0.0:
        raise ReconstructionValidationError("I4.2E raw MVS robust scene diagonal is invalid.")
    voxel_size = scene_scale * float(policy.voxel_scene_fraction)
    keys_per_point = np.floor(xyz / voxel_size).astype(np.int64)
    unique_keys, inverse = np.unique(keys_per_point, axis=0, return_inverse=True)

    # Mean RGB for display only. Geometry/support decisions do not use colour.
    raw_count = np.bincount(inverse, minlength=len(unique_keys)).astype(np.float64)
    voxel_colors = np.column_stack([
        np.bincount(inverse, weights=rgb[:, axis].astype(np.float64), minlength=len(unique_keys)) / raw_count
        for axis in range(3)
    ])
    voxel_colors = np.clip(np.rint(voxel_colors), 0, 255).astype(np.uint8)

    track_lengths = np.diff(offsets)
    point_index = np.repeat(np.arange(len(xyz), dtype=np.int64), track_lengths)
    observation_voxel = inverse[point_index]
    pair = np.column_stack((observation_voxel, image_ids))
    unique_pair, pair_inverse = np.unique(pair, axis=0, return_inverse=True)
    pair_count = np.bincount(pair_inverse, minlength=len(unique_pair)).astype(np.float64)
    pair_centroids = np.column_stack([
        np.bincount(
            pair_inverse,
            weights=xyz[point_index, axis],
            minlength=len(unique_pair),
        ) / pair_count
        for axis in range(3)
    ])

    support_all = np.bincount(unique_pair[:, 0], minlength=len(unique_keys)).astype(np.int64)
    supported_global = np.flatnonzero(support_all >= 2)
    if len(supported_global) < 256:
        raise ReconstructionValidationError(
            f"I4.2E native MVS visibility produced too few 2+ view voxels: {len(supported_global)}."
        )
    global_to_local = np.full(len(unique_keys), -1, dtype=np.int64)
    global_to_local[supported_global] = np.arange(len(supported_global), dtype=np.int64)
    pair_local = global_to_local[unique_pair[:, 0]]
    supported_pairs = pair_local >= 0
    pair_local = pair_local[supported_pairs]
    pair_centroids = pair_centroids[supported_pairs]

    order = np.argsort(pair_local, kind="stable")
    pair_local = pair_local[order]
    pair_centroids = pair_centroids[order]
    split = np.r_[0, np.flatnonzero(np.diff(pair_local)) + 1, len(pair_local)]
    refined_points = np.empty((len(supported_global), 3), dtype=np.float64)
    p90_dispersion = np.empty(len(supported_global), dtype=np.float64)
    median_dispersion = np.empty(len(supported_global), dtype=np.float64)
    for group in range(len(split) - 1):
        start, stop = int(split[group]), int(split[group + 1])
        local_index = int(pair_local[start])
        observations = pair_centroids[start:stop]
        center = np.median(observations, axis=0)
        distances = np.linalg.norm(observations - center[None, :], axis=1) / voxel_size
        refined_points[local_index] = center
        median_dispersion[local_index] = float(np.median(distances))
        p90_dispersion[local_index] = float(np.quantile(distances, 0.90))

    supported_keys = unique_keys[supported_global]
    support = support_all[supported_global]
    colors_supported = voxel_colors[supported_global]
    _, radius2, high26, component_size = _topology_metrics(supported_keys, support)
    production_filter = DenseConfidenceFilterPolicy(
        maximum_views=47,
        minimum_component_size=int(policy.minimum_component_size),
        core_max_dispersion_voxels=float(policy.core_max_dispersion_voxels),
        support2_core_max_dispersion_voxels=float(policy.support2_core_max_dispersion_voxels),
        extended_max_dispersion_voxels=float(policy.extended_max_dispersion_voxels),
        support2_core_min_high_neighbors=int(policy.support2_core_min_high_neighbors),
        support2_core_min_radius2_neighbors=int(policy.support2_core_min_radius2_neighbors),
        extended_min_radius2_neighbors=int(policy.extended_min_radius2_neighbors),
        support2_extended_min_high_neighbors=int(policy.support2_extended_min_high_neighbors),
        support2_extended_min_radius2_neighbors=int(policy.support2_extended_min_radius2_neighbors),
    )
    classification = filter_dense_supported_geometry(
        support,
        p90_dispersion,
        radius2,
        high26,
        component_size,
        policy=production_filter,
    )
    core = np.asarray(classification.core_mask, dtype=bool)
    extended = np.asarray(classification.extended_mask, dtype=bool)
    if int(np.count_nonzero(core)) < 128:
        raise ReconstructionValidationError(
            f"I4.2E MVS-derived Core analogue is too small: {int(np.count_nonzero(core))}."
        )
    if int(np.count_nonzero(extended)) < 256:
        raise ReconstructionValidationError(
            f"I4.2E MVS-derived Extended analogue is too small: {int(np.count_nonzero(extended))}."
        )
    class_code = np.where(core, 3, np.where(extended, 2, 0)).astype(np.uint8)
    ext = np.flatnonzero(extended)
    return {
        "points_reconstruction": refined_points[ext],
        "colors_rgb": colors_supported[ext],
        "view_observation_counts": support[ext].astype(np.uint32),
        "high_support_neighbor_count": high26[ext].astype(np.int32),
        "class_code": class_code[ext],
        "p90_dispersion_voxels": p90_dispersion[ext].astype(np.float32),
        "median_dispersion_voxels": median_dispersion[ext].astype(np.float32),
        "radius2_neighbor_count": radius2[ext].astype(np.int32),
        "component_size": component_size[ext].astype(np.int32),
        "voxel_size_reconstruction_units": np.asarray(voxel_size, dtype=np.float64),
        "diagnostics": {
            "raw_fused_points": int(len(xyz)),
            "raw_unique_voxels": int(len(unique_keys)),
            "supported_2plus_voxels": int(len(supported_global)),
            "core_analogue_voxels": int(np.count_nonzero(core)),
            "extended_analogue_voxels": int(np.count_nonzero(extended)),
            "robust_scene_diagonal_reconstruction_units": scene_scale,
            "voxel_size_reconstruction_units": float(voxel_size),
            "voxel_scene_fraction": float(policy.voxel_scene_fraction),
            "source_evidence": "COLMAP native fused.ply.vis visibility emitted with unchanged I4.2D fused.ply",
            "native_fusion_visibility_consumed": True,
            "stereo_fusion_replay_required": False,
            "visibility_indices_are_mvs_indices_not_colmap_image_ids": True,
            "production_i1_voxel_scale_mirrored": True,
            "production_i1_e2_scale_and_classification_defaults_mirrored": True,
            "production_dense_confidence_classifier_reused_exactly": True,
            "production_dense_confidence_policy": production_filter.as_dict(),
            "exact_da3_f_upstream_evidence_reproduction": False,
        },
    }


def _count_fraction(mask: np.ndarray, region: np.ndarray) -> dict[str, Any]:
    denominator = int(np.count_nonzero(region))
    count = int(np.count_nonzero(mask & region))
    return {
        "population": denominator,
        "rejected": count,
        "rejected_fraction": (count / float(denominator) if denominator else 0.0),
    }


def qualify_d12_detail_transfer(
    payload: Mapping[str, object],
    *,
    quality_policy: I2QualityPolicy | None = None,
    local_policy: I252LocalGeometryPolicy | None = None,
) -> dict[str, Any]:
    """Evaluate exact d12 predicate and descriptive detail-risk populations."""
    points = _xyz(payload.get("points_reconstruction"), label="Extended analogue")
    colors = _rgb(payload.get("colors_rgb"), len(points))
    support = np.asarray(payload.get("view_observation_counts"), dtype=np.int64).reshape(-1)
    high = np.asarray(payload.get("high_support_neighbor_count"), dtype=np.int64).reshape(-1)
    classes = np.asarray(payload.get("class_code"), dtype=np.uint8).reshape(-1)
    voxel = float(np.asarray(payload.get("voxel_size_reconstruction_units")).item())
    if not (support.shape == high.shape == classes.shape == (len(points),)):
        raise ReconstructionValidationError("I4.2E Extended analogue evidence arrays are misaligned.")
    if np.count_nonzero(classes == 3) < 128:
        raise ReconstructionValidationError("I4.2E requires at least 128 Core analogue voxels.")

    refinement = DenseManifoldRefinementPolicy(
        reference_source="I4.2E MVS-derived scene-relative central Core non-driver q90; exact d12 predicate",
    )
    evidence = build_scene_relative_manifold_evidence(
        {
            "points_reconstruction": points,
            "colors_rgb": colors,
            "class_code": classes,
            "view_observation_counts": support,
            "high_support_neighbor_count": high,
            "voxel_size_reconstruction_units": np.asarray(voxel, dtype=np.float64),
        },
        quality_policy=quality_policy or I2QualityPolicy(),
        local_policy=local_policy or I252LocalGeometryPolicy(),
        refinement_policy=refinement,
    )
    reject = np.asarray(evidence["reject"], dtype=bool)
    if reject.shape != (len(points),):
        raise ReconstructionValidationError("I4.2E d12 rejection mask is misaligned.")

    shape = local_shape_metrics(points, voxel_size=voxel, knn_size=16)
    reference = np.asarray(evidence["reference"], dtype=bool)
    line_q90 = float(np.quantile(np.asarray(shape["local_linearity"])[reference], 0.90))
    knn_q90 = float(evidence["reference_thresholds"]["knn_spacing_q90_voxels"])
    high_support = support >= 3
    core = classes == 3
    line_like = high_support & (np.asarray(shape["local_linearity"]) >= line_q90)
    sparse_supported = high_support & (np.asarray(shape["knn_median_distance_voxels"]) >= knn_q90)
    line_like_sparse = line_like & sparse_supported

    kept = ~reject
    if int(np.count_nonzero(kept)) == 0:
        raise ReconstructionValidationError("I4.2E d12 counterfactual rejected all Extended geometry.")
    before = spatial_envelope(points, lower_quantile=0.02, upper_quantile=0.98)
    after = spatial_envelope(points[kept], lower_quantile=0.02, upper_quantile=0.98)
    rejected_count = int(np.count_nonzero(reject))
    result = {
        "schema_version": I42E_SCHEMA_VERSION,
        "phase": I42E_PHASE,
        "policy": {
            **refinement.as_dict(),
            "exact_production_predicate": True,
            "upstream_evidence_source": "MVS-derived production-scale Extended analogue",
            "production_policy_changed": False,
        },
        "counts": {
            "pre_d12_extended_voxels": int(len(points)),
            "post_d12_extended_voxels": int(np.count_nonzero(kept)),
            "d12_rejected_voxels": rejected_count,
            "d12_rejected_fraction": rejected_count / float(len(points)),
        },
        "populations": {
            "core_analogue": _count_fraction(reject, core),
            "high_support_3plus": _count_fraction(reject, high_support),
            "supported_line_like_proxy": _count_fraction(reject, line_like),
            "supported_sparse_proxy": _count_fraction(reject, sparse_supported),
            "supported_line_like_sparse_proxy": _count_fraction(reject, line_like_sparse),
        },
        "detail_proxy_definition": {
            "supported_line_like_proxy": "support>=3 AND local_linearity>=central-reference q90",
            "supported_sparse_proxy": "support>=3 AND KNN spacing>=central-reference q90",
            "supported_line_like_sparse_proxy": "intersection of the two descriptive proxies",
            "legitimate_detail_label_claimed": False,
        },
        "potential_detail_loss_signal": bool(
            np.any(reject & core) or np.any(reject & line_like) or np.any(reject & line_like_sparse)
        ),
        "envelope": {
            "pre_d12": before.as_dict(),
            "post_d12": after.as_dict(),
            "robust_diagonal_shrink_fraction": 1.0 - after.robust_diagonal / before.robust_diagonal,
            "full_diagonal_shrink_fraction": 1.0 - after.full_diagonal / before.full_diagonal,
        },
        "reference_thresholds": dict(evidence["reference_thresholds"]),
        "resolved_rule": dict(evidence["resolved_rule"]),
        "manual_visual_review_required": True,
        "automatic_detail_winner_selected": False,
        "production_policy_change_authorized": False,
        "i4_3_provider_architecture_authorized": False,
        "metric_scale_claimed": False,
        "measurement_authoritative": False,
        "certified_metrology": False,
        "surface_mesh_generated": False,
        "arrays": {
            "pre_points": points,
            "colors": colors,
            "reject": reject,
            "core": core,
            "high_support": high_support,
            "line_like_proxy": line_like,
            "line_like_sparse_proxy": line_like_sparse,
        },
    }
    return result


def json_safe_qualification(result: Mapping[str, Any]) -> dict[str, Any]:
    """Drop in-memory arrays before persisting a qualification summary."""
    return {key: value for key, value in result.items() if key != "arrays"}
