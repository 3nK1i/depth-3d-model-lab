from __future__ import annotations

from pathlib import Path

PAGE = Path("app/widgets/reconstruction_page.py").read_text(encoding="utf-8")
UX = Path("app/reconstruction_progressive_sfm_ux.py").read_text(encoding="utf-8")
CANVAS = Path("app/widgets/viewport/vispy_canvas.py").read_text(encoding="utf-8")


def test_i6_6e_page_contains_professional_progressive_sfm_group_and_rows() -> None:
    for required in (
        'self._progressive_sfm_group = self._settings_group("PROGRESSIVE SFM")',
        'self._add_status_row(progressive_sfm_layout, "Status"',
        'self._add_status_row(progressive_sfm_layout, "Cameras"',
        'self._add_status_row(progressive_sfm_layout, "Landmarks"',
        'self._add_status_row(progressive_sfm_layout, "Transport"',
        'self._add_status_row(progressive_sfm_layout, "Viewer"',
        'self._add_status_row(progressive_sfm_layout, "Payload"',
        'self._add_status_row(progressive_sfm_layout, "Authority"',
        'self._add_status_row(progressive_sfm_layout, "Boundary"',
    ):
        assert required in PAGE


def test_i6_6e_primary_stack_extends_i6_5d_i6_6d_order_without_reordering_predecessors() -> None:
    start = PAGE.index("primary_groups = (")
    end = PAGE.index(")", start)
    block = PAGE[start:end]
    order = [
        "self._digital_twin_workspace_group",
        "self._digital_twin_processing_group",
        "self._canonical_scene_group",
        "self._estimated_ruler_scale_group",
        "self._reconstruction_advisor_group",
        "self._dense_rgb_group",
        "self._reconstruction_provider_group",
        "self._frame_view_plan_group",
        "self._progressive_sfm_group",
    ]
    positions = [block.index(item) for item in order]
    assert positions == sorted(positions)


def test_i6_6e_is_presentation_only_and_preserves_replace_only_viewer() -> None:
    for forbidden in (
        "PyQt6",
        "vispy",
        "import pycolmap",
        "subprocess",
        "start_offline",
        "incremental_mapping",
        "select_dense_provider",
        "i5_auto_policy_engine",
    ):
        assert forbidden not in UX
    assert "show_progressive_sparse_reconstruction(" in CANVAS
    assert "clear_progressive_sparse_reconstruction" in CANVAS
    assert "update_canvas=False" in CANVAS
    assert "progressive_viewer_payload_bytes(frame)" in PAGE
    assert "ProgressiveSfmViewerPerformance()" in PAGE
