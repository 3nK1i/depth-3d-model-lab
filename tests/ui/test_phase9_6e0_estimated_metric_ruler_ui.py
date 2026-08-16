from __future__ import annotations

from dataclasses import replace

import numpy as np

from app.reconstruction_canonical_scene_presentation import (
    CanonicalScenePresentation,
    CanonicalSceneResourceState,
    CanonicalSceneScaleState,
)
from app.reconstruction_estimated_metric_measurement import estimated_scale_from_ruler_reference
from app.widgets.reconstruction_page import ReconstructionPage
from core.measurement.interactive_distance import InteractiveDistanceState


def _canonical(*, unit_to_metre: float | None = None, qualified: bool = False) -> CanonicalScenePresentation:
    return CanonicalScenePresentation(
        available=True,
        scene_id="scene-e0-ui",
        scene_state="ready",
        artifact_id="canonical:surface",
        artifact_kind="surface_mesh",
        provider_id="fixture",
        canonical_geometry_authoritative=True,
        resource_state=CanonicalSceneResourceState.MATERIALIZED,
        resource_role="surface_obj_path",
        resource_path="/tmp/canonical.obj",
        calibration_available=qualified,
        scale_state=(
            CanonicalSceneScaleState.QUALIFIED
            if qualified
            else (
                CanonicalSceneScaleState.PROVISIONAL
                if unit_to_metre is not None
                else CanonicalSceneScaleState.NOT_QUALIFIED
            )
        ),
        scale_relative_uncertainty=(0.02 if unit_to_metre is not None else None),
        scale_unit_to_metre=unit_to_metre,
        metric_evidence_id=("scene-scale" if unit_to_metre is not None else ""),
        metric_assessment_state=("qualified" if qualified else "not_qualified"),
        certified_metrology=False,
    )


def _page(qtbot) -> ReconstructionPage:
    page = ReconstructionPage()
    qtbot.addWidget(page)
    vertices = np.asarray(
        [[-1.0, 0.0, -1.0], [1.0, 0.0, -1.0], [1.0, 0.0, 1.0], [-1.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    faces = np.asarray([[0, 1, 2], [0, 2, 3]], dtype=np.uint32)
    colors = np.ones((4, 4), dtype=np.float32)
    page._reconstruction_canvas.set_canonical_mesh(
        vertices_viewer=vertices,
        faces=faces,
        vertex_colors_rgba=colors,
        scene_id="scene-e0-ui",
        artifact_id="canonical:surface",
        source_path="/tmp/canonical.obj",
    )
    page._canonical_scene_presentation = _canonical()
    page._refresh_reconstruction_interactive_measurement_availability()
    page._refresh_estimated_metric_scale_presentation()
    return page


def test_e0_arbitrary_scene_can_bind_estimated_scale_without_promoting_canonical_authority(qtbot) -> None:
    page = _page(qtbot)
    page._viewport_tool_buttons["measure"].click()
    page._reconstruction_canvas.measurement_point_picked.emit(np.asarray([0.0, 0.0, 0.0]))
    page._reconstruction_canvas.measurement_point_picked.emit(np.asarray([2.0, 0.0, 0.0]))

    assert page.interactive_distance_estimate is not None
    assert page.interactive_distance_estimate.state is InteractiveDistanceState.UNAVAILABLE
    assert page._estimated_ruler_scale_set_button.isEnabled()

    hint = estimated_scale_from_ruler_reference(
        scene_id="scene-e0-ui",
        distance_reconstruction_units=2.0,
        approximate_distance_m=1.2,
        reference_relative_uncertainty=0.20,
    )
    page.set_estimated_metric_scale_hint(hint)

    estimate = page.interactive_distance_estimate
    assert estimate is not None and estimate.state is InteractiveDistanceState.ESTIMATED
    assert estimate.distance_m == 1.2
    assert estimate.expanded_uncertainty_m is not None and estimate.expanded_uncertainty_m > 0.20
    assert page._interactive_measure_primary.text().startswith("Estimated ")
    assert "±" in page._interactive_measure_primary.text()
    assert "raw 2.0000 reconstruction units" in page._interactive_measure_secondary.text()
    assert "Estimated • uncalibrated" in page._estimated_ruler_scale_state_value.text()
    assert page._canonical_scene_presentation.scale_unit_to_metre is None
    assert page.estimated_metric_scale_hint.metric_scale_authoritative is False


def test_e0_later_scene_scale_refines_current_readout_and_preserves_original(qtbot) -> None:
    page = _page(qtbot)
    page._viewport_tool_buttons["measure"].click()
    page._reconstruction_canvas.measurement_point_picked.emit(np.asarray([0.0, 0.0, 0.0]))
    page._reconstruction_canvas.measurement_point_picked.emit(np.asarray([2.0, 0.0, 0.0]))
    hint = estimated_scale_from_ruler_reference(
        scene_id="scene-e0-ui",
        distance_reconstruction_units=2.0,
        approximate_distance_m=1.2,
        reference_relative_uncertainty=0.20,
    )
    page.set_estimated_metric_scale_hint(hint)
    original = page.ruler_metric_record
    assert original is not None and original.current.distance_m == 1.2

    page._canonical_scene_presentation = _canonical(unit_to_metre=0.5, qualified=True)
    page._refresh_estimated_metric_scale_presentation()
    page._refresh_active_ruler_metric_record()

    refined = page.ruler_metric_record
    assert refined is not None
    assert refined.current.state is InteractiveDistanceState.QUALIFIED
    assert refined.current.distance_m == 1.0
    assert refined.original_estimate is not None
    assert refined.original_estimate.distance_m == 1.2
    assert "original estimate preserved" in page._interactive_measure_secondary.text()
    assert "Original estimate" in page._interactive_measure_primary.toolTip()
    assert "scene scale takes precedence" in page._estimated_ruler_scale_state_value.text()


def test_e0_clearing_hint_does_not_fabricate_metric_value(qtbot) -> None:
    page = _page(qtbot)
    page._viewport_tool_buttons["measure"].click()
    page._reconstruction_canvas.measurement_point_picked.emit(np.asarray([0.0, 0.0, 0.0]))
    page._reconstruction_canvas.measurement_point_picked.emit(np.asarray([2.0, 0.0, 0.0]))
    page.set_estimated_metric_scale_hint(
        estimated_scale_from_ruler_reference(
            scene_id="scene-e0-ui",
            distance_reconstruction_units=2.0,
            approximate_distance_m=1.2,
            reference_relative_uncertainty=0.20,
        )
    )
    page.clear_estimated_metric_scale_hint()
    assert page.estimated_metric_scale_hint is None
    assert page.interactive_distance_estimate is not None
    assert page.interactive_distance_estimate.state is InteractiveDistanceState.UNAVAILABLE
    assert page.interactive_distance_estimate.distance_m is None
    assert page._interactive_measure_primary.text() == "Metric estimate unavailable"
