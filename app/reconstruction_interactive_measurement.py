"""Dependency-light projected-surface picking for interactive reconstruction measurement.

The Reconstruction canvas owns the actual VisPy transform.  This module only
consumes the already-mapped homogeneous vertex coordinates plus one screen click
and resolves a point on the persisted canonical triangle mesh.  It never grants
measurement or metric authority to the viewer itself.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True, slots=True)
class ProjectedSurfacePick:
    face_index: int
    point_viewer: tuple[float, float, float]
    barycentric: tuple[float, float, float]
    projected_depth: float


def _barycentric_2d(point: np.ndarray, triangle: np.ndarray) -> np.ndarray | None:
    a, b, c = triangle
    v0 = b - a
    v1 = c - a
    v2 = point - a
    denominator = float(v0[0] * v1[1] - v1[0] * v0[1])
    if abs(denominator) <= 1e-10:
        return None
    inv = 1.0 / denominator
    u = float((v2[0] * v1[1] - v1[0] * v2[1]) * inv)
    v = float((v0[0] * v2[1] - v2[0] * v0[1]) * inv)
    weights = np.asarray((1.0 - u - v, u, v), dtype=np.float64)
    if np.any(weights < -1e-6) or np.any(weights > 1.0 + 1e-6):
        return None
    return weights


def pick_projected_surface_point(
    *,
    vertices_viewer: np.ndarray,
    faces: np.ndarray,
    mapped_homogeneous: np.ndarray,
    click_xy: Sequence[float],
) -> ProjectedSurfacePick | None:
    """Return the nearest projected canonical triangle hit for one canvas click.

    ``mapped_homogeneous`` is the result of mapping ``vertices_viewer`` through
    the active visual-to-canvas transform.  Perspective-correct interpolation is
    used when homogeneous ``w`` differs across a face.
    """

    vertices = np.asarray(vertices_viewer, dtype=np.float64)
    triangles = np.asarray(faces, dtype=np.int64)
    mapped = np.asarray(mapped_homogeneous, dtype=np.float64)
    point = np.asarray(tuple(click_xy), dtype=np.float64)
    if vertices.ndim != 2 or vertices.shape[1:] != (3,) or not np.isfinite(vertices).all():
        raise ValueError("vertices_viewer must be a finite (N, 3) array.")
    if triangles.ndim != 2 or triangles.shape[1:] != (3,) or np.any(triangles < 0) or np.any(triangles >= vertices.shape[0]):
        raise ValueError("faces must be a valid (F, 3) index array.")
    if mapped.ndim != 2 or mapped.shape[0] != vertices.shape[0] or mapped.shape[1] not in {3, 4}:
        raise ValueError("mapped_homogeneous must be (N, 3) or (N, 4) for the same vertices.")
    if point.shape != (2,) or not np.isfinite(point).all():
        raise ValueError("click_xy must contain two finite coordinates.")

    if mapped.shape[1] == 3:
        w_all = np.ones((mapped.shape[0],), dtype=np.float64)
        normalized = mapped
    else:
        w_all = mapped[:, 3]
        valid_w = np.isfinite(w_all) & (np.abs(w_all) > 1e-10)
        normalized = np.full((mapped.shape[0], 3), np.nan, dtype=np.float64)
        normalized[valid_w] = mapped[valid_w, :3] / w_all[valid_w, None]

    # Fast screen-space bounding-box rejection keeps picking responsive for
    # larger canonical meshes while preserving exact triangle membership.
    face_projected_all = normalized[triangles]
    face_w_all = w_all[triangles]
    finite = np.isfinite(face_projected_all).all(axis=(1, 2)) & np.isfinite(face_w_all).all(axis=1)
    # Standard perspective transforms use positive clip w for points in front
    # of the eye. Affine transforms use w=1. Reject behind-camera faces.
    finite &= np.all(face_w_all > 1e-10, axis=1)
    mins = np.nanmin(face_projected_all[:, :, :2], axis=1)
    maxs = np.nanmax(face_projected_all[:, :, :2], axis=1)
    candidates = np.flatnonzero(
        finite
        & (point[0] >= mins[:, 0] - 1e-6)
        & (point[0] <= maxs[:, 0] + 1e-6)
        & (point[1] >= mins[:, 1] - 1e-6)
        & (point[1] <= maxs[:, 1] + 1e-6)
    )

    best: ProjectedSurfacePick | None = None
    for face_index in candidates:
        face = triangles[face_index]
        face_w = face_w_all[face_index]
        face_projected = face_projected_all[face_index]
        bary_screen = _barycentric_2d(point, face_projected[:, :2])
        if bary_screen is None:
            continue

        # Perspective-correct source-space interpolation.  With an affine
        # transform this reduces to the ordinary screen barycentric weights.
        corrected = bary_screen / face_w
        denominator = float(corrected.sum())
        if abs(denominator) <= 1e-12 or not np.isfinite(denominator):
            continue
        bary_source = corrected / denominator
        if not np.isfinite(bary_source).all():
            continue
        viewer_point = bary_source @ vertices[face]
        projected_depth = float(bary_screen @ face_projected[:, 2])
        candidate = ProjectedSurfacePick(
            face_index=int(face_index),
            point_viewer=tuple(float(value) for value in viewer_point),
            barycentric=tuple(float(value) for value in bary_source),
            projected_depth=projected_depth,
        )
        if best is None or candidate.projected_depth < best.projected_depth:
            best = candidate
    return best
