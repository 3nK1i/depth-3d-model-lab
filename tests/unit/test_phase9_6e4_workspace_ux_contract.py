from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PAGE = ROOT / "app" / "widgets" / "reconstruction_page.py"


def _source() -> str:
    return PAGE.read_text(encoding="utf-8")


def test_e4_is_presentation_only_and_keeps_existing_group_objects() -> None:
    source = _source()
    assert "def _apply_digital_twin_workspace_ux_order" in source
    assert "self._digital_twin_workspace_group" in source
    assert "self._digital_twin_processing_group" in source
    assert "self._canonical_scene_group" in source
    assert "self._estimated_ruler_scale_group" in source
    assert "layout.removeWidget(group)" in source
    assert "layout.insertWidget(index, group)" in source
    assert 'self._digital_twin_workspace_details_button = QPushButton("Show details")' in source
    assert "self._digital_twin_workspace_details.hide()" in source


def test_e4_does_not_weaken_processing_or_authority_contracts() -> None:
    source = _source()
    # Job visibility is presentation-only; the existing controller and lifecycle
    # remain the source of truth and no new backend launch surface is introduced.
    assert "DigitalTwinProcessingJobController(self)" in source
    assert "snapshot.lifecycle is DigitalTwinProcessingLifecycle.IDLE" in source
    assert "_focus_digital_twin_processing_for_new_job(snapshot)" in source
    assert "Canonical geometry remains authoritative" in source
