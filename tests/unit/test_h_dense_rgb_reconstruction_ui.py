from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from app.reconstruction_dense_rgb_ui import (
    DENSE_RGB_REPRESENTATIONS,
    build_dense_rgb_ui_presentation,
    load_dense_rgb_ui_representation,
    robust_dense_rgb_fit_bounds,
)
from core.reconstruction.errors import ReconstructionValidationError
from tests.phase_h_dense_rgb_helpers import make_phase_h_dense_rgb_bundle


def test_h_load_bridge_restores_verified_extended_and_core_without_backends(tmp_path: Path):
    bundle, extended_points, core_points, orders = make_phase_h_dense_rgb_bundle(tmp_path)
    extended = load_dense_rgb_ui_representation(bundle, representation="extended")
    core = load_dense_rgb_ui_representation(bundle, representation="core")

    assert extended.geometry.point_count == len(extended_points)
    assert core.geometry.point_count == len(core_points)
    assert extended.scene.scene_id == core.scene.scene_id
    assert extended.presentation.source_view_count == len(orders)
    assert extended.presentation.canonical_geometry_authoritative is True
    assert extended.presentation.canonical_persistence_promoted is True
    assert extended.presentation.reconstruction_ui_enabled is True
    assert extended.presentation.metric_scale_claimed is False
    assert extended.presentation.measurement_authoritative is False
    assert extended.presentation.certified_metrology is False
    assert extended.presentation.surface_mesh_generated is False
    assert extended.presentation.backend_started is False


def test_h_presentation_surfaces_required_dense_rgb_status_metadata(tmp_path: Path):
    bundle, extended_points, core_points, orders = make_phase_h_dense_rgb_bundle(tmp_path)
    loaded = load_dense_rgb_ui_representation(bundle, representation="extended")
    presentation = loaded.presentation

    assert presentation.point_count == len(extended_points)
    assert presentation.primary_point_count == len(extended_points)
    assert presentation.core_point_count == len(core_points)
    assert presentation.source_view_count == len(orders)
    assert presentation.representation_label == "Extended (primary)"
    assert "qualified source views" in presentation.views_text
    assert "views/voxel" in presentation.support_text
    assert presentation.minimum_view_support >= 2
    assert presentation.maximum_view_support >= presentation.minimum_view_support
    assert "reconstruction units" in presentation.voxel_text
    assert "median confidence" in presentation.quality_text
    assert "metric scale NOT claimed" in presentation.scale_text
    assert "measurement not qualified" in presentation.authority_text
    assert "Persistent Dense RGB Points ready" in presentation.state_text

    # Presentation-only Fit bounds ignore a tiny extreme-point tail.
    central = np.column_stack((
        np.linspace(-1.0, 1.0, 200),
        np.linspace(-0.5, 0.5, 200),
        np.linspace(-0.25, 0.25, 200),
    ))
    fit_points = np.vstack((
        central,
        [[-100.0, -100.0, -100.0], [100.0, 100.0, 100.0]],
    ))
    fit_lower, fit_upper = robust_dense_rgb_fit_bounds(fit_points)
    assert np.all(fit_lower > -10.0)
    assert np.all(fit_upper < 10.0)
    assert np.all(fit_upper > fit_lower)


def test_h_core_presentation_has_higher_fixture_confidence_and_same_scene_identity(tmp_path: Path):
    bundle, *_ = make_phase_h_dense_rgb_bundle(tmp_path)
    extended = load_dense_rgb_ui_representation(bundle, representation="extended")
    core = load_dense_rgb_ui_representation(bundle, representation="core")
    assert core.presentation.scene_id == extended.presentation.scene_id
    assert core.presentation.scene_fingerprint_sha256 == extended.presentation.scene_fingerprint_sha256
    assert core.presentation.median_filter_confidence > extended.presentation.median_filter_confidence
    assert core.presentation.representation_label == "Core (high confidence)"


def test_h_rejects_invalid_representation_before_ui(tmp_path: Path):
    bundle, *_ = make_phase_h_dense_rgb_bundle(tmp_path)
    assert DENSE_RGB_REPRESENTATIONS == ("extended", "core")
    with pytest.raises(ReconstructionValidationError):
        load_dense_rgb_ui_representation(bundle, representation="mesh")


def test_h_rejects_tampered_g_bundle_before_restoring_ui_geometry(tmp_path: Path):
    bundle, *_ = make_phase_h_dense_rgb_bundle(tmp_path)
    geometry_path = bundle / "geometry" / "dense_rgb_extended.npz"
    with geometry_path.open("ab") as handle:
        handle.write(b"tamper")
    with pytest.raises(ReconstructionValidationError):
        load_dense_rgb_ui_representation(bundle, representation="extended")


def test_h_build_presentation_rejects_scene_geometry_identity_drift(tmp_path: Path):
    bundle, *_ = make_phase_h_dense_rgb_bundle(tmp_path)
    loaded = load_dense_rgb_ui_representation(bundle, representation="extended")
    # The normal loader has already proven the positive path. The immutable
    # restored object is deliberately replaced with a wrong scene id here to
    # prove the presentation bridge refuses cross-scene geometry binding.
    geometry = loaded.geometry
    wrong = type(geometry)(
        scene_id="wrong-scene",
        representation=geometry.representation,
        points_reconstruction=geometry.points_reconstruction,
        colors_rgb=geometry.colors_rgb,
        view_observation_counts=geometry.view_observation_counts,
        filter_confidence=geometry.filter_confidence,
        voxel_size_reconstruction_units=geometry.voxel_size_reconstruction_units,
        selected_workspace_orders=geometry.selected_workspace_orders,
        selected_keyframe_ids=geometry.selected_keyframe_ids,
        source_file_sha256=geometry.source_file_sha256,
    )
    with pytest.raises(ReconstructionValidationError, match="identity mismatch"):
        build_dense_rgb_ui_presentation(loaded.scene, wrong, loaded.manifest_path)


def test_h_source_has_no_heavy_backend_execution_or_premature_authority_promotion():
    root = Path(__file__).resolve().parents[2]
    source = (root / "app" / "reconstruction_dense_rgb_ui.py").read_text(encoding="utf-8")
    assert "restore_persistent_dense_geometry" in source
    assert "verify_files=True" in source
    assert "subprocess" not in source
    assert "import pycolmap" not in source.lower()
    assert "from pycolmap" not in source.lower()
    assert "import depth_anything" not in source.lower()
    assert "metric_scale_claimed: bool = False" in source
    assert "measurement_authoritative: bool = False" in source
    assert "surface_mesh_generated: bool = False" in source
    assert "backend_started: bool = False" in source

def test_h_page_and_canvas_source_use_existing_reconstruction_viewport_without_ui_redesign():
    root = Path(__file__).resolve().parents[2]
    page = (root / "app" / "widgets" / "reconstruction_page.py").read_text(encoding="utf-8")
    canvas = (root / "app" / "widgets" / "viewport" / "vispy_canvas.py").read_text(encoding="utf-8")
    assert 'self._dense_rgb_group = self._settings_group("DENSE RGB POINTS")' in page
    assert 'self._dense_rgb_scene_layer_toggle = ToggleSwitch' in page
    assert 'self._scene_layer_toggles.get("dense_rgb_points")' not in page
    primary_start = page.index("primary_groups = (")
    primary_end = page.index(")", primary_start)
    primary_block = page[primary_start:primary_end]
    assert primary_block.index("self._estimated_ruler_scale_group") < primary_block.index("self._dense_rgb_group")
    dense_start = page.index('self._dense_rgb_group = self._settings_group("DENSE RGB POINTS")')
    dense_end = page.index("layout.addWidget(self._dense_rgb_group)", dense_start)
    dense_block = page[dense_start:dense_end]
    assert "QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed" in dense_block
    assert "self._reconstruction_canvas.set_dense_rgb_points(" in page
    assert '"fit_dense_rgb_points"' in page
    assert "def fit_dense_rgb_points(" in canvas
    assert "def set_dense_rgb_points(" in canvas
    assert "MAX_DENSE_RGB_DISPLAY_POINTS = 250_000" in canvas
    assert "sparse_landmarks_to_viewer(points)" in canvas
    assert "scene.visuals.SceneCanvas" not in page
