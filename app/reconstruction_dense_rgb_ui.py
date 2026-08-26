"""Dense RGB Points presentation bridge for Phase H.

Phase H consumes the self-contained Phase-G persistent dense RGB bundle and
presents its canonical point geometry inside the existing Reconstruction
workspace.  This module is dependency-light and deliberately contains no Qt or
backend execution.  It verifies the persisted bundle through the Phase-G
contract, restores only the selected Core/Extended representation, and builds
read-only presentation metadata for the UI.

The H bridge never invents metric scale, measurement authority, certified
metrology, or surface-mesh authority.  DA3/PyCOLMAP are not imported or
executed here.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np

from core.reconstruction.dense_scene_persistence import (
    DENSE_RGB_SCENE_MANIFEST,
    PersistentDenseRgbScene,
    RestoredDenseRgbGeometry,
    read_persistent_dense_rgb_scene,
    restore_persistent_dense_geometry,
)
from core.reconstruction.errors import ReconstructionValidationError


DENSE_RGB_UI_PHASE = "H"
DENSE_RGB_REPRESENTATIONS = ("extended", "core")
DENSE_RGB_FIT_LOWER_QUANTILE = 0.01
DENSE_RGB_FIT_UPPER_QUANTILE = 0.99
DENSE_RGB_FIT_MIN_POINTS = 64


def robust_dense_rgb_fit_bounds(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return robust camera-fit bounds without altering persisted geometry.

    Dense RGB scenes can legitimately retain a small number of distant points.
    Those points stay rendered and persisted, but they should not dominate the
    user's explicit Fit action. For sufficiently large clouds we therefore use
    conservative 1st/99th per-axis quantiles for presentation-only camera
    framing. Small or degenerate clouds fall back to their exact bounds.
    """

    values = np.asarray(points, dtype=np.float64)
    if values.ndim != 2 or values.shape[1:] != (3,) or values.shape[0] < 1:
        raise ValueError("Dense RGB fit points must be a non-empty (N, 3) array.")
    if not np.isfinite(values).all():
        raise ValueError("Dense RGB fit points must be finite.")

    exact_lower = np.min(values, axis=0)
    exact_upper = np.max(values, axis=0)
    if values.shape[0] < DENSE_RGB_FIT_MIN_POINTS:
        return exact_lower, exact_upper

    robust_lower = np.quantile(values, DENSE_RGB_FIT_LOWER_QUANTILE, axis=0)
    robust_upper = np.quantile(values, DENSE_RGB_FIT_UPPER_QUANTILE, axis=0)
    robust_span = robust_upper - robust_lower
    exact_span = exact_upper - exact_lower
    if (
        not np.isfinite(robust_lower).all()
        or not np.isfinite(robust_upper).all()
        or float(np.linalg.norm(robust_span)) <= 1e-9
        or float(np.linalg.norm(exact_span)) <= 1e-9
    ):
        return exact_lower, exact_upper
    return robust_lower, robust_upper


@dataclass(frozen=True, slots=True)
class DenseRgbUiPresentation:
    """Read-only UI projection of one verified persistent Dense RGB scene."""

    scene_id: str
    scene_fingerprint_sha256: str
    manifest_path: str
    representation: str
    representation_label: str
    point_count: int
    primary_point_count: int
    core_point_count: int
    source_view_count: int
    source_supported_point_count: int
    source_all_voxel_count: int
    voxel_size_reconstruction_units: float
    minimum_view_support: int
    median_view_support: float
    p90_view_support: float
    maximum_view_support: int
    median_filter_confidence: float
    scene_text: str
    representation_text: str
    points_text: str
    views_text: str
    support_text: str
    voxel_text: str
    quality_text: str
    scale_text: str
    authority_text: str
    state_text: str
    canonical_geometry_authoritative: bool = True
    canonical_persistence_promoted: bool = True
    reconstruction_ui_enabled: bool = True
    metric_scale_claimed: bool = False
    measurement_authoritative: bool = False
    certified_metrology: bool = False
    surface_mesh_generated: bool = False
    backend_started: bool = False

    def __post_init__(self) -> None:
        representation = str(self.representation).strip().lower()
        if representation not in DENSE_RGB_REPRESENTATIONS:
            raise ReconstructionValidationError("Dense RGB UI representation must be Extended or Core.")
        if not str(self.scene_id).strip() or not str(self.scene_fingerprint_sha256).strip():
            raise ReconstructionValidationError("Dense RGB UI scene identity is incomplete.")
        if int(self.point_count) <= 0:
            raise ReconstructionValidationError("Dense RGB UI representation must be non-empty.")
        if int(self.source_view_count) < 2:
            raise ReconstructionValidationError("Dense RGB UI requires multi-view provenance.")
        if float(self.voxel_size_reconstruction_units) <= 0.0:
            raise ReconstructionValidationError("Dense RGB UI voxel size must be positive.")
        if int(self.minimum_view_support) < 2 or int(self.maximum_view_support) < int(self.minimum_view_support):
            raise ReconstructionValidationError("Dense RGB UI multi-view support range is invalid.")
        if not (float(self.minimum_view_support) <= float(self.median_view_support) <= float(self.maximum_view_support)):
            raise ReconstructionValidationError("Dense RGB UI median view support is outside its range.")
        if not (float(self.minimum_view_support) <= float(self.p90_view_support) <= float(self.maximum_view_support)):
            raise ReconstructionValidationError("Dense RGB UI P90 view support is outside its range.")
        if not 0.0 <= float(self.median_filter_confidence) <= 1.0:
            raise ReconstructionValidationError("Dense RGB UI confidence must be in [0, 1].")
        if not self.canonical_geometry_authoritative or not self.canonical_persistence_promoted:
            raise ReconstructionValidationError("Phase H requires G-promoted persistent canonical geometry.")
        if not self.reconstruction_ui_enabled:
            raise ReconstructionValidationError("Phase H presentation must explicitly enable UI presentation.")
        if any(
            (
                self.metric_scale_claimed,
                self.measurement_authoritative,
                self.certified_metrology,
                self.surface_mesh_generated,
                self.backend_started,
            )
        ):
            raise ReconstructionValidationError("Phase H cannot promote metric/measurement/mesh authority or start a backend.")
        object.__setattr__(self, "representation", representation)


@dataclass(frozen=True, slots=True)
class DenseRgbUiLoadResult:
    scene: PersistentDenseRgbScene
    geometry: RestoredDenseRgbGeometry
    presentation: DenseRgbUiPresentation
    manifest_path: Path

    def __post_init__(self) -> None:
        if self.geometry.scene_id != self.scene.scene_id:
            raise ReconstructionValidationError("Dense RGB UI geometry scene identity changed during restore.")
        if self.presentation.scene_id != self.scene.scene_id:
            raise ReconstructionValidationError("Dense RGB UI presentation scene identity changed during restore.")
        if self.presentation.point_count != self.geometry.point_count:
            raise ReconstructionValidationError("Dense RGB UI presentation point count disagrees with restored geometry.")


def _manifest_path(path: str | Path) -> Path:
    target = Path(path).expanduser().resolve()
    if target.is_dir():
        target = target / DENSE_RGB_SCENE_MANIFEST
    return target


def _mapping(value: object) -> Mapping[str, object]:
    return value if isinstance(value, Mapping) else {}


def _confidence_median(scene: PersistentDenseRgbScene, representation: str) -> float:
    confidence = _mapping(scene.quality_snapshot.get("confidence"))
    key = "extended_quantiles" if representation == "extended" else "core_quantiles"
    quantiles = _mapping(confidence.get(key))
    value = quantiles.get("p50", 0.0)
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return 0.0
    return min(max(numeric, 0.0), 1.0)


def build_dense_rgb_ui_presentation(
    scene: PersistentDenseRgbScene,
    geometry: RestoredDenseRgbGeometry,
    manifest_path: str | Path,
) -> DenseRgbUiPresentation:
    if not isinstance(scene, PersistentDenseRgbScene):
        raise TypeError("scene must be PersistentDenseRgbScene.")
    if not isinstance(geometry, RestoredDenseRgbGeometry):
        raise TypeError("geometry must be RestoredDenseRgbGeometry.")
    if geometry.scene_id != scene.scene_id:
        raise ReconstructionValidationError("Dense RGB UI scene/geometry identity mismatch.")
    if not scene.canonical_geometry_authoritative or not scene.canonical_persistence_promoted:
        raise ReconstructionValidationError("H requires G-promoted canonical persistence.")
    if scene.metric_scale_claimed or scene.measurement_authoritative or scene.certified_metrology:
        raise ReconstructionValidationError("H source scene contains premature metric or measurement authority.")
    if scene.surface_mesh_generated:
        raise ReconstructionValidationError("H Dense RGB Points integration cannot consume a scene claiming a generated surface mesh.")

    representation = geometry.representation
    expected = scene.primary_point_count if representation == "extended" else scene.core_point_count
    if geometry.point_count != expected:
        raise ReconstructionValidationError("H restored geometry count disagrees with G persistent scene.")
    if geometry.selected_workspace_orders != scene.selected_workspace_orders:
        raise ReconstructionValidationError("H restored geometry workspace-order provenance disagrees with G scene.")
    if geometry.selected_keyframe_ids != scene.selected_keyframe_ids:
        raise ReconstructionValidationError("H restored geometry keyframe provenance disagrees with G scene.")

    label = "Extended (primary)" if representation == "extended" else "Core (high confidence)"
    confidence_median = _confidence_median(scene, representation)
    support = geometry.view_observation_counts.astype(float, copy=False)
    minimum_support = int(np.min(support))
    median_support = float(np.median(support))
    p90_support = float(np.quantile(support, 0.90))
    maximum_support = int(np.max(support))
    manifest = _manifest_path(manifest_path)
    return DenseRgbUiPresentation(
        scene_id=scene.scene_id,
        scene_fingerprint_sha256=scene.scene_fingerprint_sha256,
        manifest_path=str(manifest),
        representation=representation,
        representation_label=label,
        point_count=geometry.point_count,
        primary_point_count=scene.primary_point_count,
        core_point_count=scene.core_point_count,
        source_view_count=scene.selected_view_count,
        source_supported_point_count=scene.source_supported_point_count,
        source_all_voxel_count=scene.source_all_voxel_count,
        voxel_size_reconstruction_units=scene.voxel_size_reconstruction_units,
        minimum_view_support=minimum_support,
        median_view_support=median_support,
        p90_view_support=p90_support,
        maximum_view_support=maximum_support,
        median_filter_confidence=confidence_median,
        scene_text=f"{scene.scene_id} • verified persistent bundle",
        representation_text=label,
        points_text=(
            f"{geometry.point_count:,} current • Extended {scene.primary_point_count:,} • "
            f"Core {scene.core_point_count:,} • pre-filter supported {scene.source_supported_point_count:,}"
        ),
        views_text=f"{scene.selected_view_count} qualified source views",
        support_text=(
            f"{minimum_support}–{maximum_support} views/voxel • "
            f"median {median_support:.1f} • P90 {p90_support:.1f}"
        ),
        voxel_text=f"{scene.voxel_size_reconstruction_units:.6f} reconstruction units",
        quality_text=(
            f"Filtered dense geometry qualified • median confidence {confidence_median:.3f}"
            if confidence_median > 0.0
            else "Filtered dense geometry qualified"
        ),
        scale_text="Arbitrary SfM gauge • metric scale NOT claimed",
        authority_text="Canonical geometry • picking/spatial relationships • measurement not qualified",
        state_text="Persistent Dense RGB Points ready in Reconstruction UI",
    )


def load_dense_rgb_ui_representation(
    path: str | Path,
    *,
    representation: str = "extended",
) -> DenseRgbUiLoadResult:
    """Verify one G bundle and restore one Dense RGB representation for H."""

    kind = str(representation).strip().lower()
    if kind not in DENSE_RGB_REPRESENTATIONS:
        raise ReconstructionValidationError("representation must be 'extended' or 'core'.")
    manifest = _manifest_path(path)
    scene = read_persistent_dense_rgb_scene(manifest, verify_files=True)
    geometry = restore_persistent_dense_geometry(manifest, representation=kind)
    presentation = build_dense_rgb_ui_presentation(scene, geometry, manifest)
    return DenseRgbUiLoadResult(
        scene=scene,
        geometry=geometry,
        presentation=presentation,
        manifest_path=manifest,
    )


__all__ = [
    "DENSE_RGB_REPRESENTATIONS",
    "DENSE_RGB_UI_PHASE",
    "DenseRgbUiLoadResult",
    "DenseRgbUiPresentation",
    "robust_dense_rgb_fit_bounds",
    "build_dense_rgb_ui_presentation",
    "load_dense_rgb_ui_representation",
]
