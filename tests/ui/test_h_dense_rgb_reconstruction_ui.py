from __future__ import annotations

from pathlib import Path

import pytest

from app.main_window import MainWindow
from core.reconstruction.errors import ReconstructionValidationError
from tests.phase_h_dense_rgb_helpers import make_phase_h_dense_rgb_bundle


def _window(qtbot, monkeypatch):
    monkeypatch.setattr("app.application_controller.CameraEnumerator.start", lambda self: None)
    window = MainWindow()
    qtbot.addWidget(window)
    return window


def test_h_loads_g_bundle_as_first_class_dense_rgb_layer(qtbot, monkeypatch, tmp_path: Path) -> None:
    window = _window(qtbot, monkeypatch)
    page = window._reconstruction_page
    bundle, extended_points, core_points, orders = make_phase_h_dense_rgb_bundle(tmp_path)
    canvas = page._reconstruction_canvas
    fits_before = canvas.fit_calls

    assert page.set_persistent_dense_rgb_scene(bundle) is True
    assert page.dense_rgb_presentation is not None
    assert page.dense_rgb_representation == "extended"
    assert page.dense_rgb_manifest_path == (bundle / "persistent_dense_rgb_scene.json").resolve()
    assert canvas.dense_rgb_points_loaded is True
    assert canvas.dense_rgb_points_visible is True
    assert canvas.dense_rgb_total_count == len(extended_points)
    assert canvas.dense_rgb_display_count == len(extended_points)
    assert canvas.fit_calls == fits_before + 1
    assert page._viewport_placeholder.isVisible() is False
    assert "Canonical Dense RGB Points" in page._viewport_state.text()
    assert "Extended" in page._viewport_state.text()
    assert f"{len(orders)} qualified source views" in page._dense_rgb_views_value.text()
    assert "views/voxel" in page._dense_rgb_support_value.text()
    assert f"Extended {len(extended_points):,}" in page._dense_rgb_points_value.text()
    assert f"Core {len(core_points):,}" in page._dense_rgb_points_value.text()
    assert "metric scale NOT claimed" in page._dense_rgb_scale_value.text()
    assert "measurement not qualified" in page._dense_rgb_authority_value.text()
    assert page._dense_rgb_representation_combo.isEnabled() is True
    assert page._dense_rgb_fit_button.isEnabled() is True
    assert page._dense_rgb_unload_button.isEnabled() is True


def test_h_switches_extended_core_without_refitting_or_changing_scene_identity(qtbot, monkeypatch, tmp_path: Path) -> None:
    window = _window(qtbot, monkeypatch)
    page = window._reconstruction_page
    bundle, extended_points, core_points, _orders = make_phase_h_dense_rgb_bundle(tmp_path)
    canvas = page._reconstruction_canvas
    page.set_persistent_dense_rgb_scene(bundle)
    scene_id = canvas.dense_rgb_scene_id
    fits_before = canvas.fit_calls

    index = page._dense_rgb_representation_combo.findData("core")
    page._dense_rgb_representation_combo.setCurrentIndex(index)
    qtbot.wait(1)
    assert page.dense_rgb_representation == "core"
    assert canvas.dense_rgb_representation == "core"
    assert canvas.dense_rgb_total_count == len(core_points)
    assert canvas.dense_rgb_scene_id == scene_id
    assert canvas.fit_calls == fits_before
    assert "Core" in page._viewport_state.text()

    index = page._dense_rgb_representation_combo.findData("extended")
    page._dense_rgb_representation_combo.setCurrentIndex(index)
    qtbot.wait(1)
    assert page.dense_rgb_representation == "extended"
    assert canvas.dense_rgb_total_count == len(extended_points)
    assert canvas.dense_rgb_scene_id == scene_id
    assert canvas.fit_calls == fits_before


def test_h_scene_layer_toggle_is_visibility_only(qtbot, monkeypatch, tmp_path: Path) -> None:
    window = _window(qtbot, monkeypatch)
    page = window._reconstruction_page
    bundle, *_ = make_phase_h_dense_rgb_bundle(tmp_path)
    page.set_persistent_dense_rgb_scene(bundle)
    canvas = page._reconstruction_canvas
    presentation = page.dense_rgb_presentation
    assert presentation is not None

    assert "dense_rgb_points" not in page._scene_layer_toggles
    toggle = page._dense_rgb_scene_layer_toggle
    assert toggle.isEnabled() is True
    assert toggle.isChecked() is True
    toggle.setChecked(False)
    assert canvas.dense_rgb_points_loaded is True
    assert canvas.dense_rgb_points_visible is False
    assert page.dense_rgb_presentation == presentation
    toggle.setChecked(True)
    assert canvas.dense_rgb_points_visible is True
    assert presentation.metric_scale_claimed is False
    assert presentation.measurement_authoritative is False


def test_h_tampered_new_load_is_fail_closed_and_preserves_current_good_scene(qtbot, monkeypatch, tmp_path: Path) -> None:
    window = _window(qtbot, monkeypatch)
    page = window._reconstruction_page
    good_bundle, *_ = make_phase_h_dense_rgb_bundle(tmp_path / "good")
    # helper expects its root parent to exist
    bad_bundle, *_ = make_phase_h_dense_rgb_bundle(tmp_path / "bad")
    page.set_persistent_dense_rgb_scene(good_bundle)
    canvas = page._reconstruction_canvas
    scene_id_before = canvas.dense_rgb_scene_id
    count_before = canvas.dense_rgb_total_count
    presentation_before = page.dense_rgb_presentation

    with (bad_bundle / "geometry" / "dense_rgb_extended.npz").open("ab") as handle:
        handle.write(b"tamper")
    with pytest.raises(ReconstructionValidationError):
        page.set_persistent_dense_rgb_scene(bad_bundle)

    assert canvas.dense_rgb_points_loaded is True
    assert canvas.dense_rgb_scene_id == scene_id_before
    assert canvas.dense_rgb_total_count == count_before
    assert page.dense_rgb_presentation == presentation_before


def test_h_unload_clears_only_dense_rgb_ui_layer(qtbot, monkeypatch, tmp_path: Path) -> None:
    window = _window(qtbot, monkeypatch)
    page = window._reconstruction_page
    bundle, *_ = make_phase_h_dense_rgb_bundle(tmp_path)
    page.set_persistent_dense_rgb_scene(bundle)
    canvas = page._reconstruction_canvas

    page.clear_persistent_dense_rgb_scene()
    assert page.dense_rgb_presentation is None
    assert page.dense_rgb_manifest_path is None
    assert page.dense_rgb_representation == "extended"
    assert canvas.dense_rgb_points_loaded is False
    assert page._dense_rgb_representation_combo.isEnabled() is False
    assert page._dense_rgb_fit_button.isEnabled() is False
    assert page._dense_rgb_unload_button.isEnabled() is False
    assert page._dense_rgb_state_value.text() == "Waiting for persistent G bundle"


def test_h_ui_does_not_mutate_g_manifest_or_enable_metric_measurement_mesh_authority(qtbot, monkeypatch, tmp_path: Path) -> None:
    window = _window(qtbot, monkeypatch)
    page = window._reconstruction_page
    bundle, *_ = make_phase_h_dense_rgb_bundle(tmp_path)
    manifest = bundle / "persistent_dense_rgb_scene.json"
    before = manifest.read_bytes()

    page.set_persistent_dense_rgb_scene(bundle)
    presentation = page.dense_rgb_presentation
    assert presentation is not None
    assert presentation.reconstruction_ui_enabled is True
    assert presentation.canonical_geometry_authoritative is True
    assert presentation.metric_scale_claimed is False
    assert presentation.measurement_authoritative is False
    assert presentation.certified_metrology is False
    assert presentation.surface_mesh_generated is False
    assert presentation.backend_started is False
    assert manifest.read_bytes() == before
