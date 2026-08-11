#!/usr/bin/env python3
"""Decisive mock-only full Reconstruction UX review for Phase 8.8G6."""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import QApplication

from app.application_controller import CameraEnumerator
from app.offline_reconstruction_controller import OfflineReconstructionRunState
from app.reconstruction_checkpoint_resume import ReconstructionCheckpointResumeState
from app.reconstruction_failure_recovery import ReconstructionFailureRecoveryState
from app.reconstruction_session_model import ReconstructionSessionIdentityStatus
from app.widgets.mode_tabs import AppMode
from app.widgets.reconstruction_settings_disclosure import ReconstructionSettingsDisclosureGroup
from app.main_window import MainWindow
from tests.reconstruction_g6_helpers import write_full_review_evidence

OUTPUT = Path("/tmp/phase8_8g6_full_ux_review").resolve()
EXPECTED_GROUP_TITLES = (
    "WORKFLOW",
    "SESSION & PERSISTENCE",
    "OFFLINE INSPECTION",
    "SEMANTIC / OBJECTS",
    "INSPECTION TARGET",
    "RECONSTRUCTION MEASUREMENTS",
    "MEASUREMENT VALIDITY",
    "INSPECTION / QC",
    "REFERENCE COMPARISON",
    "APPEARANCE REPRESENTATIONS",
    "GAUSSIAN SPLATTING VISUALIZATION",
    "NERF VISUALIZATION",
    "RESOURCE & FALLBACK",
    "AUTHORITY & SAFETY",
)


def _groups(page):
    body = page._settings_scroll.widget()
    layout = body.layout()
    return tuple(
        widget
        for index in range(layout.count())
        if isinstance((widget := layout.itemAt(index).widget()), ReconstructionSettingsDisclosureGroup)
    )


def main() -> int:
    shutil.rmtree(OUTPUT, ignore_errors=True)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    fixture = write_full_review_evidence(OUTPUT / "fixture")

    CameraEnumerator.start = lambda self: None  # type: ignore[method-assign]
    app = QApplication.instance() or QApplication(sys.argv)
    window = MainWindow()
    window._mode_tabs.set_active_mode(AppMode.RECONSTRUCTION)
    page = window._reconstruction_page
    controller = window._reconstruction_controller
    window.resize(1366, 768)
    window.show()
    # Match the passing G6 UI review: allow the visible stacked page, splitter,
    # and scroll area to finish propagating size hints before asserting geometry.
    # A single processEvents() can still expose Qt's transient pre-layout range.
    QTest.qWait(20)
    app.processEvents()

    errors: list[str] = []
    try:
        groups = _groups(page)
        group_order_ok = tuple(group.title for group in groups) == EXPECTED_GROUP_TITLES
        unique_groups = len({group.title for group in groups}) == len(EXPECTED_GROUP_TITLES)
        vertical_only = bool(
            page._settings_scroll.horizontalScrollBarPolicy().name == "ScrollBarAlwaysOff"
            and page._settings_scroll.horizontalScrollBar().maximum() == 0
            and page._settings_scroll.verticalScrollBarPolicy().name == "ScrollBarAsNeeded"
        )
        initial_safe = bool(
            not page._start_button.isEnabled()
            and not page._pause_button.isEnabled()
            and not page._stop_button.isEnabled()
            and page._session_load_button.isEnabled()
            and page._session_checkpoint_load_button.isEnabled()
            and not page._session_resume_button.isEnabled()
            and not page._session_retry_button.isEnabled()
        )

        controller.select_image_sequence(fixture.source_dir)
        app.processEvents()
        source_ready = bool(
            page._start_button.isEnabled()
            and page._workflow_input_value.text() == "3 images"
            and page._source_value.text() == fixture.source_dir.name
            and not page._session_retry_button.isEnabled()
        )

        page.set_error("G6 review fixture • contained offline failure")
        page.set_offline_state(OfflineReconstructionRunState.FAILED)
        app.processEvents()
        recovery_route = bool(
            page.failure_recovery_presentation is not None
            and page.failure_recovery_presentation.state is ReconstructionFailureRecoveryState.RETRY_AVAILABLE
            and not page._start_button.isEnabled()
            and page._session_retry_button.isEnabled()
            and not page._session_resume_button.isEnabled()
        )

        page.set_offline_state(OfflineReconstructionRunState.IDLE)
        controller.load_persistent_scene(fixture.scene_dir)
        controller.load_engine_checkpoint(fixture.checkpoint_manifest)
        compatibility = controller.set_engine_resume_context(
            plan=fixture.plan,
            providers=fixture.providers,
            available_payload_refs=fixture.checkpoint.payload_refs,
        )
        app.processEvents()
        session_model = page.reconstruction_session_model
        same_session_evidence = bool(
            compatibility is not None
            and compatibility.compatible
            and session_model is not None
            and session_model.identity_status is ReconstructionSessionIdentityStatus.CONSISTENT
            and session_model.session_id == fixture.scene_session_id
            and session_model.canonical_geometry_authoritative
            and not session_model.certified_metrology
            and page._session_save_button.isEnabled()
            and page._session_reload_button.isEnabled()
            and page._session_resume_button.isEnabled()
        )

        page.set_error("G6 review fixture • checkpoint and persistent scene retained")
        page.set_offline_state(OfflineReconstructionRunState.FAILED)
        app.processEvents()
        explicit_resume_route = bool(
            page.checkpoint_resume_presentation is not None
            and page.checkpoint_resume_presentation.state is ReconstructionCheckpointResumeState.COMPATIBLE
            and page._session_resume_button.isEnabled()
            and page.failure_recovery_presentation is not None
            and page.failure_recovery_presentation.state is ReconstructionFailureRecoveryState.PRESERVED_EVIDENCE
            and not page._session_retry_button.isEnabled()
        )

        before_actions = (
            page._session_load_button.isEnabled(),
            page._session_save_button.isEnabled(),
            page._session_reload_button.isEnabled(),
            page._session_checkpoint_load_button.isEnabled(),
            page._session_resume_button.isEnabled(),
            page._session_retry_button.isEnabled(),
        )
        page._session_group.set_expanded(False)
        app.processEvents()
        collapsed_actions = before_actions == (
            page._session_load_button.isEnabled(),
            page._session_save_button.isEnabled(),
            page._session_reload_button.isEnabled(),
            page._session_checkpoint_load_button.isEnabled(),
            page._session_resume_button.isEnabled(),
            page._session_retry_button.isEnabled(),
        )
        page._session_group.set_expanded(True)
        app.processEvents()

        disclosure_defaults = bool(
            page._workflow_group.expanded
            and page._session_group.expanded
            and page._inspection_target_group.expanded
            and page._appearance_group.expanded
            and page._authority_group.expanded
            and not page._reconstruction_measurement_group.expanded
            and not page._measurement_validity_group.expanded
            and not page._inspection_qc_group.expanded
            and not page._reference_comparison_group.expanded
            and not page._gaussian_visualization_group.expanded
            and not page._nerf_visualization_group.expanded
            and not page._appearance_resource_group.expanded
        )
        authority_safe = bool(
            page._metric_scale_value.text() == "Evidence-driven"
            and page._metrology_value.text() == "Not certified"
            and session_model is not None
            and session_model.canonical_geometry_authoritative
            and not session_model.certified_metrology
        )
    except Exception as exc:  # pragma: no cover - smoke diagnostics
        errors.append(f"{type(exc).__name__}: {exc}")
        group_order_ok = unique_groups = vertical_only = initial_safe = source_ready = False
        recovery_route = same_session_evidence = explicit_resume_route = False
        collapsed_actions = disclosure_defaults = authority_safe = False

    screenshot = OUTPUT / "reconstruction_g6_full_ux_review.png"
    screenshot_ok = window.grab().save(str(screenshot))
    summary = {
        "phase": "8.8G6",
        "pass": bool(
            group_order_ok
            and unique_groups
            and vertical_only
            and initial_safe
            and source_ready
            and recovery_route
            and same_session_evidence
            and explicit_resume_route
            and collapsed_actions
            and disclosure_defaults
            and authority_safe
            and screenshot_ok
            and not errors
        ),
        "complete_group_order": group_order_ok,
        "unique_settings_groups": unique_groups,
        "vertical_scroll_only": vertical_only,
        "idle_entry_safe": initial_safe,
        "ready_source_start_path": source_ready,
        "terminal_failure_routes_to_retry": recovery_route,
        "same_session_scene_checkpoint_consistent": same_session_evidence,
        "checkpoint_failure_routes_to_prepare_resume": explicit_resume_route,
        "disclosure_preserves_action_state": collapsed_actions,
        "progressive_disclosure_defaults_correct": disclosure_defaults,
        "canonical_geometry_authority_preserved": authority_safe,
        "certified_metrology": False,
        "new_runtime_behavior_added_by_g6": False,
        "checkpoint_resume_executed_by_g6": False,
        "failure_recovery_executed_by_g6": False,
        "heavy_backend_executed_by_g6": False,
        "new_persistence_format_written_by_g6": False,
        "screenshot": str(screenshot),
        "errors": errors,
    }
    (OUTPUT / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    window.close()
    app.processEvents()
    print("Phase 8.8G6 full Reconstruction UX review smoke: " + ("PASS" if summary["pass"] else "FAIL"))
    print(
        f"groups={group_order_ok} idle={initial_safe} recovery={recovery_route} "
        f"resume={explicit_resume_route} authority={authority_safe}"
    )
    print(f"screenshot={screenshot}")
    print(f"summary={OUTPUT / 'summary.json'}")
    return 0 if summary["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
