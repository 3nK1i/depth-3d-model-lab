from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np

from app.reconstruction_canonical_scene_presentation import (
    CanonicalScenePresentation,
    CanonicalSceneResourceState,
    CanonicalSceneScaleState,
)
from app.widgets.inspection_page import InspectionPage
from app.widgets.reconstruction_page import ReconstructionPage
from core.inspection import load_inspection_bundle
from core.measurement.interactive_distance import InteractiveDistanceState
from tests.inspection_helpers import write_inspection_bundle


def _inspection_bundle_with_depth(tmp_path: Path):
    summary, arrays = load_inspection_bundle(write_inspection_bundle(tmp_path / "bundle"))
    arrays = dict(arrays)
    arrays["aligned_depth"] = np.full((summary.image_height, summary.image_width), 2.0, dtype=np.float32)
    arrays["aligned_depth_valid_mask"] = np.ones((summary.image_height, summary.image_width), dtype=bool)
    arrays["aligned_intrinsics_matrix"] = np.asarray(
        [[80.0, 0.0, summary.image_width / 2.0], [0.0, 80.0, summary.image_height / 2.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    return summary, arrays


def test_measure_and_inspect_ruler_is_one_button_two_clicks_with_estimated_uncertainty(qtbot, tmp_path: Path) -> None:
    page = InspectionPage()
    qtbot.addWidget(page)
    summary, arrays = _inspection_bundle_with_depth(tmp_path)
    page.load_bundle(summary, arrays)

    button = page._interactive_measure_button
    assert button._icon_name == "ruler"
    assert button.isEnabled()
    button.click()
    assert button.isChecked()
    assert page._image_canvas.measurement_mode is True
    assert page._interactive_measure_bar.isVisible() is False or not page.isVisible()

    page._image_canvas.measurement_point_clicked.emit(20, 24)
    assert "Point A" in page._interactive_measure_primary.text()
    page._image_canvas.measurement_point_clicked.emit(44, 24)

    estimate = page.interactive_distance_estimate
    assert estimate is not None
    assert estimate.state is InteractiveDistanceState.ESTIMATED
    assert estimate.distance_m is not None and estimate.distance_m > 0.0
    assert estimate.expanded_uncertainty_m is not None and estimate.expanded_uncertainty_m > 0.0
    assert "±" in page._interactive_measure_primary.text()
    assert "ESTIMATED" in page._interactive_measure_secondary.text()
    assert estimate.certified_metrology is False


def test_measure_and_inspect_old_bundle_without_numeric_depth_keeps_ruler_disabled(qtbot, tmp_path: Path) -> None:
    page = InspectionPage()
    qtbot.addWidget(page)
    summary, arrays = load_inspection_bundle(write_inspection_bundle(tmp_path / "legacy"))
    page.load_bundle(summary, arrays)
    assert page._interactive_measure_button.isEnabled() is False
    assert "Analyze a frame first" in page._interactive_measure_button.toolTip()


def _canonical_presentation(*, provisional: bool = True) -> CanonicalScenePresentation:
    return CanonicalScenePresentation(
        available=True,
        scene_id="prs-interactive-measurement",
        scene_state="ready",
        artifact_id="canonical:surface",
        artifact_kind="surface_mesh",
        provider_id="fixture",
        canonical_geometry_authoritative=True,
        resource_state=CanonicalSceneResourceState.MATERIALIZED,
        resource_role="surface_obj_path",
        resource_path="/tmp/canonical.obj",
        calibration_available=False,
        scale_state=(CanonicalSceneScaleState.PROVISIONAL if provisional else CanonicalSceneScaleState.NOT_QUALIFIED),
        scale_relative_uncertainty=(0.05 if provisional else None),
        scale_unit_to_metre=(0.5 if provisional else None),
        metric_evidence_id=("fixture-scale" if provisional else ""),
        metric_assessment_state=("estimated_metric" if provisional else "not_qualified"),
        certified_metrology=False,
    )


def _reconstruction_page_with_surface(qtbot) -> ReconstructionPage:
    page = ReconstructionPage()
    qtbot.addWidget(page)
    vertices = np.asarray([[-1.0, 0.0, -1.0], [1.0, 0.0, -1.0], [1.0, 0.0, 1.0], [-1.0, 0.0, 1.0]], dtype=np.float32)
    faces = np.asarray([[0, 1, 2], [0, 2, 3]], dtype=np.uint32)
    colors = np.ones((4, 4), dtype=np.float32)
    page._reconstruction_canvas.set_canonical_mesh(
        vertices_viewer=vertices,
        faces=faces,
        vertex_colors_rgba=colors,
        scene_id="prs-interactive-measurement",
        artifact_id="canonical:surface",
        source_path="/tmp/canonical.obj",
    )
    page._canonical_scene_presentation = _canonical_presentation(provisional=True)
    page._refresh_reconstruction_interactive_measurement_availability()
    return page


def test_reconstruction_ruler_uses_canonical_points_and_provisional_scale_for_estimate(qtbot) -> None:
    page = _reconstruction_page_with_surface(qtbot)
    button = page._viewport_tool_buttons["measure"]
    assert button._icon_name == "ruler"
    assert button.isEnabled()
    button.click()
    assert page._reconstruction_canvas.interaction_mode == "measure"

    page._reconstruction_canvas.measurement_point_picked.emit(np.asarray([0.0, 0.0, 0.0]))
    assert "Point A" in page._interactive_measure_primary.text()
    page._reconstruction_canvas.measurement_point_picked.emit(np.asarray([2.0, 0.0, 0.0]))
    estimate = page.interactive_distance_estimate
    assert estimate is not None
    assert estimate.state is InteractiveDistanceState.ESTIMATED
    assert estimate.distance_m == 1.0
    assert "±" in page._interactive_measure_primary.text()
    assert "ESTIMATED" in page._interactive_measure_secondary.text()
    assert estimate.certified_metrology is False


def test_reconstruction_arbitrary_scale_never_invents_metric_distance(qtbot) -> None:
    page = _reconstruction_page_with_surface(qtbot)
    page._canonical_scene_presentation = _canonical_presentation(provisional=False)
    page._viewport_tool_buttons["measure"].click()
    page._reconstruction_canvas.measurement_point_picked.emit(np.asarray([0.0, 0.0, 0.0]))
    page._reconstruction_canvas.measurement_point_picked.emit(np.asarray([3.0, 4.0, 0.0]))
    estimate = page.interactive_distance_estimate
    assert estimate is not None
    assert estimate.state is InteractiveDistanceState.UNAVAILABLE
    assert estimate.distance_m is None
    assert page._interactive_measure_primary.text() == "Metric estimate unavailable"
    assert "5.0000 reconstruction units" in page._interactive_measure_secondary.text()


def test_reconstruction_escape_cancels_ruler_and_returns_to_orbit(qtbot) -> None:
    page = _reconstruction_page_with_surface(qtbot)
    page._viewport_tool_buttons["measure"].click()
    page._reconstruction_canvas.measurement_point_picked.emit(np.asarray([0.0, 0.0, 0.0]))
    page._on_viewport_escape()
    assert page._reconstruction_canvas.interaction_mode == "orbit"
    assert page._viewport_tool_buttons["orbit"].isChecked()
    assert page.interactive_distance_estimate is None
