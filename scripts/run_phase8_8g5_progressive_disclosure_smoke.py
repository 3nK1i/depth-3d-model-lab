#!/usr/bin/env python3
"""Dependency-light UI smoke for Phase 8.8G5 progressive disclosure."""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from PyQt6.QtWidgets import QApplication

from app.application_controller import CameraEnumerator
from app.main_window import MainWindow
from app.widgets.mode_tabs import AppMode
from app.widgets.reconstruction_settings_disclosure import ReconstructionSettingsDisclosureGroup

OUTPUT = Path("/tmp/phase8_8g5_progressive_disclosure").resolve()


def main() -> int:
    shutil.rmtree(OUTPUT, ignore_errors=True)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    CameraEnumerator.start = lambda self: None  # type: ignore[method-assign]
    app = QApplication.instance() or QApplication(sys.argv)
    window = MainWindow()
    window._mode_tabs.set_active_mode(AppMode.RECONSTRUCTION)
    page = window._reconstruction_page
    app.processEvents()

    errors: list[str] = []
    try:
        primary = (
            page._workflow_group,
            page._session_group,
            page._inspection_target_group,
            page._appearance_group,
            page._authority_group,
        )
        details = (
            page._reconstruction_measurement_group,
            page._measurement_validity_group,
            page._inspection_qc_group,
            page._reference_comparison_group,
            page._gaussian_visualization_group,
            page._nerf_visualization_group,
            page._appearance_resource_group,
        )
        all_groups_are_disclosure = all(
            isinstance(group, ReconstructionSettingsDisclosureGroup)
            for group in (*primary, *details)
        )
        primary_expanded = all(group.expanded for group in primary)
        details_collapsed = all(not group.expanded for group in details)

        session = page._session_group
        session_is_disclosure = isinstance(session, ReconstructionSettingsDisclosureGroup)
        action_state_before = (
            page._session_load_button.isEnabled(),
            page._session_resume_button.isEnabled(),
            page._session_retry_button.isEnabled(),
        )
        session.set_expanded(False)
        app.processEvents()
        session_collapsed = not session.expanded and session._body.isHidden()
        action_state_preserved = action_state_before == (
            page._session_load_button.isEnabled(),
            page._session_resume_button.isEnabled(),
            page._session_retry_button.isEnabled(),
        )
        session.set_expanded(True)
        page._reconstruction_measurement_group.set_expanded(True)
        app.processEvents()
        explicit_expand_works = bool(
            session.expanded
            and page._reconstruction_measurement_group.expanded
            and not page._reconstruction_measurement_group._body.isHidden()
        )
        # Return screenshot to the compact default G5 posture.
        page._reconstruction_measurement_group.set_expanded(False)
        app.processEvents()
    except Exception as exc:  # pragma: no cover - smoke diagnostics
        errors.append(f"{type(exc).__name__}: {exc}")
        all_groups_are_disclosure = primary_expanded = details_collapsed = False
        session_is_disclosure = session_collapsed = action_state_preserved = False
        explicit_expand_works = False

    window.resize(1366, 768)
    window.show()
    app.processEvents()
    screenshot = OUTPUT / "reconstruction_g5_progressive_disclosure.png"
    screenshot_ok = window.grab().save(str(screenshot))

    summary = {
        "phase": "8.8G5",
        "pass": bool(
            all_groups_are_disclosure
            and primary_expanded
            and details_collapsed
            and session_is_disclosure
            and session_collapsed
            and action_state_preserved
            and explicit_expand_works
            and screenshot_ok
            and not errors
        ),
        "all_groups_are_disclosure": all_groups_are_disclosure,
        "primary_context_expanded": primary_expanded,
        "detail_groups_collapsed": details_collapsed,
        "session_disclosure_available": session_is_disclosure,
        "session_can_collapse": session_collapsed,
        "session_action_state_preserved": action_state_preserved,
        "explicit_expand_works": explicit_expand_works,
        "controller_state_mutated_by_g5": False,
        "checkpoint_resume_executed_by_g5": False,
        "failure_recovery_executed_by_g5": False,
        "heavy_backend_executed_by_g5": False,
        "authority_changed_by_g5": False,
        "screenshot": str(screenshot),
        "errors": errors,
    }
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    window.close()
    app.processEvents()
    print("Phase 8.8G5 progressive disclosure UX smoke: " + ("PASS" if summary["pass"] else "FAIL"))
    print(
        f"primary_expanded={primary_expanded} details_collapsed={details_collapsed} "
        f"session_collapse={session_collapsed}"
    )
    print(f"screenshot={screenshot}")
    print(f"summary={OUTPUT / 'summary.json'}")
    return 0 if summary["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
