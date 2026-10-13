from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import pytest

pytest.importorskip("PyQt6")

from app.reconstruction_processing_jobs import (
    DigitalTwinProcessingJobSpec,
    DigitalTwinProcessingKind,
    begin_digital_twin_processing_snapshot,
)
from app.widgets.reconstruction_page import ReconstructionPage
from app.widgets.reconstruction_settings_disclosure import ReconstructionSettingsDisclosureGroup


def _settings_group_titles(page: ReconstructionPage) -> tuple[str, ...]:
    layout = page._settings_scroll.widget().layout()
    titles: list[str] = []
    for index in range(layout.count()):
        widget = layout.itemAt(index).widget()
        if isinstance(widget, ReconstructionSettingsDisclosureGroup):
            titles.append(widget.title)
    return tuple(titles)


def _job(tmp_path: Path, *, title: str = "Build Full-Scene DA3"):
    return begin_digital_twin_processing_snapshot(
        DigitalTwinProcessingJobSpec(
            job_id=str(uuid4()),
            kind=DigitalTwinProcessingKind.DENSE_GEOMETRY,
            title=title,
            backend_label="Depth Anything 3",
            runtime_label="external eia-da3",
            output_dir=tmp_path / str(uuid4()),
            expected_protocol="depth3d.digital-twin.da3-full-scene.v1",
        )
    )


def test_e4_promotes_unified_workspace_without_removing_legacy_groups(qtbot) -> None:
    page = ReconstructionPage()
    qtbot.addWidget(page)

    titles = _settings_group_titles(page)
    assert titles[:4] == (
        "DIGITAL TWIN WORKSPACE",
        "BUILD & PROCESSING",
        "CANONICAL SCENE",
        "RULER METRIC ESTIMATE",
    )
    for legacy in (
        "WORKFLOW",
        "SESSION & PERSISTENCE",
        "ADVANCED RECONSTRUCTION",
        "OFFLINE INSPECTION",
        "APPEARANCE REPRESENTATIONS",
        "AUTHORITY & SAFETY",
    ):
        assert legacy in titles



def test_e4_workspace_summary_uses_nested_progressive_disclosure(qtbot) -> None:
    page = ReconstructionPage()
    qtbot.addWidget(page)

    assert page._digital_twin_workspace_details.isHidden() is True
    assert page._digital_twin_workspace_details_button.text() == "Show details"
    navigation = page._digital_twin_navigation_value.text()
    resources = page._digital_twin_resource_value.text()

    page._digital_twin_workspace_details_button.click()
    qtbot.wait(10)
    assert page._digital_twin_workspace_details.isHidden() is False
    assert page._digital_twin_workspace_details_button.text() == "Hide details"
    assert page._digital_twin_navigation_value.text() == navigation
    assert page._digital_twin_resource_value.text() == resources

    page._digital_twin_workspace_details_button.click()
    qtbot.wait(10)
    assert page._digital_twin_workspace_details.isHidden() is True
    assert page._digital_twin_workspace_details_button.text() == "Show details"

def test_e4_preserves_existing_disclosure_defaults(qtbot) -> None:
    page = ReconstructionPage()
    qtbot.addWidget(page)

    # E4 changes visual hierarchy, not the established Phase 8 disclosure contract.
    assert page._workflow_group.expanded is True
    assert page._session_group.expanded is True
    assert page._appearance_group.expanded is True
    assert page._authority_group.expanded is True
    assert page._digital_twin_processing_group.expanded is False
    assert page._estimated_ruler_scale_group.expanded is False


def test_e4_new_heavy_job_focuses_processing_once_and_respects_user_afterward(
    qtbot, tmp_path: Path
) -> None:
    page = ReconstructionPage()
    qtbot.addWidget(page)
    first = _job(tmp_path)

    assert page._digital_twin_processing_group.expanded is False
    page._on_digital_twin_processing_snapshot_changed(first)
    qtbot.wait(10)
    assert page._digital_twin_processing_group.expanded is True
    assert page._digital_twin_processing_focus_job_id == first.spec.job_id

    # After the one-time focus, the user's disclosure choice wins for this job.
    page._digital_twin_processing_group.set_expanded(False)
    page._on_digital_twin_processing_snapshot_changed(first)
    qtbot.wait(10)
    assert page._digital_twin_processing_group.expanded is False

    # A genuinely new job is important enough to surface once again.
    second = _job(tmp_path, title="Rebuild NeRF")
    page._on_digital_twin_processing_snapshot_changed(second)
    qtbot.wait(10)
    assert page._digital_twin_processing_group.expanded is True
    assert page._digital_twin_processing_focus_job_id == second.spec.job_id


def test_e4_primary_stack_remains_horizontal_scroll_free_at_1366(qtbot) -> None:
    page = ReconstructionPage()
    qtbot.addWidget(page)
    page.resize(1366, 768)
    page.show()
    qtbot.wait(20)

    assert page._settings_scroll.horizontalScrollBarPolicy().name == "ScrollBarAlwaysOff"
    assert page._settings_scroll.horizontalScrollBar().maximum() == 0
    assert page._digital_twin_workspace_group.y() < page._workflow_group.y()
