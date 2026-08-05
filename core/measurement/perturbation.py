"""Deterministic perturbation and repeated fitting for honest heuristic uncertainty."""
from __future__ import annotations
from collections import defaultdict
import numpy as np
from core.alignment.contracts import AlignedFrameBundle, AlignedObjectObservation
from core.measurement.contracts import BoundingBox3D, BoundingBoxKind, MeasurementKind
from core.measurement.geometry import values_from_box
from core.measurement.policy import PerturbationPolicy
from core.object_geometry.boxes import fit_camera_aligned, fit_oriented
from core.object_geometry.fitting_contracts import ObjectGeometryFitResult
from core.object_geometry.principal_axes import fit_principal_axes
from core.object_geometry.fitting_policy import RobustPcaPolicy
from core.object_geometry.contracts import ObjectPointExtractionResult


def _fit_like(points: np.ndarray, baseline: BoundingBox3D, low: float, high: float, orientation: np.ndarray | None = None) -> BoundingBox3D:
    if baseline.kind is BoundingBoxKind.ORIENTED:
        R = baseline.orientation_matrix if orientation is None else orientation
        return fit_oriented(points, baseline.observation_id, baseline.coordinate_frame_id, R, low, high)[0]
    return fit_camera_aligned(points, baseline.observation_id, baseline.coordinate_frame_id, low, high)[0]

def _append(samples, box: BoundingBox3D) -> None:
    for kind, value in values_from_box(box).items():
        samples[kind].append(float(value))

def _reproject(points: np.ndarray, pixels: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    z = points[:, 2]
    fx, fy, cx, cy = float(matrix[0,0]), float(matrix[1,1]), float(matrix[0,2]), float(matrix[1,2])
    u, v = pixels[:,0].astype(float), pixels[:,1].astype(float)
    return np.column_stack(((u-cx)*z/fx, (v-cy)*z/fy, z))

def generate_perturbation_samples(
    *, baseline_box: BoundingBox3D, fit_result: ObjectGeometryFitResult,
    extraction_result: ObjectPointExtractionResult, bundle: AlignedFrameBundle,
    observation: AlignedObjectObservation, policy: PerturbationPolicy,
) -> dict[MeasurementKind, tuple[float, ...]]:
    samples: dict[MeasurementKind, list[float]] = defaultdict(list)
    _append(samples, baseline_box)
    cloud = extraction_result.filtered_cloud
    if cloud is None:
        return {k: tuple(v) for k,v in samples.items()}
    points = np.asarray(cloud.points_xyz_m, dtype=np.float64)
    for low, high in policy.percentile_pairs:
        try: _append(samples, _fit_like(points, baseline_box, low, high))
        except ValueError: pass
    for stride in policy.point_subset_strides:
        for offset in range(min(stride, 2)):
            rng = np.random.default_rng(stride * 1009 + offset)
            order = rng.permutation(len(points))
            subset = points[order[::stride]]
            if len(subset) < 32: continue
            try:
                orientation = baseline_box.orientation_matrix
                if baseline_box.kind is BoundingBoxKind.ORIENTED:
                    orientation = fit_principal_axes(cloud=type('C', (), {'points_xyz_m': subset, 'point_weights': None, 'observation_id': cloud.observation_id, 'coordinate_frame_id': cloud.coordinate_frame_id})(), policy=RobustPcaPolicy()).orientation_matrix
                _append(samples, _fit_like(subset, baseline_box, 1.0, 99.0, orientation))
            except (ValueError, np.linalg.LinAlgError): pass
    for factor in (1.0-policy.depth_relative_perturbation, 1.0+policy.depth_relative_perturbation):
        if factor > 0:
            try: _append(samples, _fit_like(points*factor, baseline_box, 1.0, 99.0))
            except ValueError: pass
    if cloud.source_pixel_indices_xy is not None and len(cloud.source_pixel_indices_xy) == len(points):
        K = np.array(bundle.intrinsics.matrix, copy=True)
        variants=[]
        for sign in (-1.0,1.0):
            k=np.array(K,copy=True); k[0,0]*=1+sign*policy.focal_length_relative_perturbation; k[1,1]*=1+sign*policy.focal_length_relative_perturbation; variants.append(k)
            k=np.array(K,copy=True); k[0,2]+=sign*policy.principal_point_perturbation_pixels; k[1,2]+=sign*policy.principal_point_perturbation_pixels; variants.append(k)
        for k in variants:
            try: _append(samples, _fit_like(_reproject(points, cloud.source_pixel_indices_xy, k), baseline_box, 1.0, 99.0))
            except ValueError: pass
    raw = extraction_result.raw_cloud
    if raw is not None and len(raw.points_xyz_m) >= 32:
        try: _append(samples, _fit_like(np.asarray(raw.points_xyz_m), baseline_box, 1.0, 99.0))
        except ValueError: pass
    return {kind: tuple(values) for kind, values in samples.items()}
