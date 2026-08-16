from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("PyQt6")

from app.reconstruction_digital_twin_persistence import (
    read_optional_digital_twin_workspace_state,
    write_digital_twin_workspace_state,
)
from app.reconstruction_estimated_metric_measurement import (
    build_reconstruction_ruler_metric_record,
    estimated_scale_from_ruler_reference,
)
from app.widgets.reconstruction_page import ReconstructionPage
from core.reconstruction import persistent_scene_from_pipeline_result, write_persistent_reconstruction_scene
from tests.reconstruction_f5_helpers import run_f5_fixture


def _scene(tmp_path: Path):
    _fixture, _providers, result, snapshots = run_f5_fixture(tmp_path / "run")
    return persistent_scene_from_pipeline_result(result, payload_snapshots_by_kind=snapshots)


def test_f_session_ui_exposes_twin_state_separately_from_product_export(qtbot) -> None:
    page = ReconstructionPage()
    qtbot.addWidget(page)
    assert page._digital_twin_workspace_state_value.text() == "No Digital Twin workspace sidecar"
    assert page._digital_twin_state_export_button.text() == "Export State"
    assert page._digital_twin_state_export_open_button.text() == "Open State Export"
    assert "not embedded" in page._digital_twin_state_export_button.toolTip()
    assert page._reconstruction_export_button.text() == "Export Bundle"
    assert page._digital_twin_processing_history_value.text() == "No persisted build history"


def test_f_page_restores_estimated_ruler_provenance_without_launching_appearance(qtbot, tmp_path: Path) -> None:
    page = ReconstructionPage()
    qtbot.addWidget(page)
    scene = _scene(tmp_path)
    root = (tmp_path / "scene").resolve()
    manifest = write_persistent_reconstruction_scene(scene, root)
    page.set_persistent_scene(scene, str(manifest))

    hint = estimated_scale_from_ruler_reference(
        scene_id=scene.scene_id,
        distance_reconstruction_units=2.0,
        approximate_distance_m=1.0,
        reference_relative_uncertainty=0.2,
    )
    ruler = build_reconstruction_ruler_metric_record(
        scene_id=scene.scene_id,
        point_a_reconstruction=(0.0, 0.0, 0.0),
        point_b_reconstruction=(2.0, 0.0, 0.0),
        canonical_scene=page.canonical_scene_presentation,
        estimated_scale_hint=hint,
    )
    page.set_estimated_metric_scale_hint(hint)
    page._ruler_metric_record = ruler
    page._interactive_measure_points_reconstruction = []
    state = page._capture_digital_twin_workspace_state()
    assert state is not None
    write_digital_twin_workspace_state(state, root)

    page.clear_estimated_metric_scale_hint()
    page._ruler_metric_record = None
    page._restore_digital_twin_workspace_state(manifest)
    loaded, _ = read_optional_digital_twin_workspace_state(manifest, expected_scene=scene)
    assert loaded is not None
    assert page.estimated_metric_scale_hint == loaded.estimated_metric_scale_hint
    assert page.ruler_metric_record is not None
    assert page.ruler_metric_record.original_estimate is not None
    assert page._interactive_gaussian_visible is False
    assert page._interactive_nerf_visible is False
