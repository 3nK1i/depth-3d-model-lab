from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PAGE = ROOT / "app/widgets/reconstruction_page.py"
PREVIEW = ROOT / "app/widgets/reconstruction_appearance_preview.py"
CANVAS = ROOT / "app/widgets/viewport/vispy_canvas.py"


def test_96d_workspace_exposes_compact_interaction_status_without_new_picker() -> None:
    page = PAGE.read_text(encoding="utf-8")
    assert 'self._digital_twin_interaction_value = self._settings_value("Waiting for canonical interaction")' in page
    assert 'self._add_status_row(\n            workspace_details_layout, "Interaction", self._digital_twin_interaction_value\n        )' in page
    assert "build_digital_twin_interaction_presentation" in page
    assert "_pick_canonical_surface_point" not in page


def test_96d_static_previews_fail_closed_before_ruler_or_semantic_inspection() -> None:
    page = PAGE.read_text(encoding="utf-8")
    tool = page[page.index("    def _set_viewport_tool"):page.index("    def _fit_reconstruction_view")]
    assert "self._hide_gaussian_preview()" in tool
    assert "self._hide_nerf_preview()" in tool
    semantic = page[page.index("    def _on_semantic_object_selected"):page.index("    def _refresh_semantic_navigation")]
    assert "Static backend previews are not camera-synchronized interaction surfaces" in semantic
    assert "self._hide_gaussian_preview()" in semantic
    assert "self._hide_nerf_preview()" in semantic


def test_96d_interactive_nerf_remains_mouse_transparent_and_is_dimmed_not_promoted() -> None:
    page = PAGE.read_text(encoding="utf-8")
    preview = PREVIEW.read_text(encoding="utf-8")
    assert "set_mouse_passthrough" in preview
    assert "WA_TransparentForMouseEvents" in preview
    assert "_apply_digital_twin_interaction_appearance_policy" in page
    assert "NeRF frozen/dimmed for canonical ruler" in page
    assert "NeRF dimmed for semantic inspection" in page
    assert "min(int(effective), int(cap))" in page


def test_96d_semantic_focus_refreshes_interactive_nerf_from_canonical_camera() -> None:
    page = PAGE.read_text(encoding="utf-8")
    start = page.index("    def _focus_selected_semantic_object")
    end = page.index("    def _clear_semantic_object_selection", start)
    method = page[start:end]
    assert "self._reconstruction_canvas.focus_selected_semantic_object()" in method
    assert "self._request_interactive_nerf_render(force=True)" in method
    assert "self._refresh_digital_twin_navigation_sync()" in method


def test_96d_canvas_still_picks_only_canonical_mesh_for_ruler() -> None:
    canvas = CANVAS.read_text(encoding="utf-8")
    start = canvas.index("    def _pick_canonical_surface_point")
    end = canvas.index("    def set_interactive_measurement_overlay", start)
    method = canvas[start:end]
    assert "self._canonical_mesh_vertices_viewer" in method
    assert "self._canonical_mesh_faces" in method
    assert "interactive_gaussian" not in method
    assert "nerf" not in method.lower()
