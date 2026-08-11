#!/usr/bin/env python3
"""Decisive F1 smoke: persisted Phase 8.7F5 appearance artifacts -> UI controls."""

from __future__ import annotations

import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from PyQt6.QtWidgets import QApplication

from app.application_controller import CameraEnumerator
from app.main_window import MainWindow
from app.widgets.mode_tabs import AppMode
from core.reconstruction import (
    persistent_scene_from_pipeline_result,
    read_persistent_reconstruction_scene,
    write_persistent_reconstruction_scene,
)
from tests.reconstruction_f5_helpers import run_f5_fixture

OUTPUT = Path("/tmp/phase8_8f1_appearance_controls")


def _persistent_scene(path: Path, *, degrade_gaussian: bool, request_id: str):
    _fixture, _providers, result, snapshots = run_f5_fixture(
        path / "source",
        degrade_gaussian=degrade_gaussian,
        request_id=request_id,
    )
    scene = persistent_scene_from_pipeline_result(result, payload_snapshots_by_kind=snapshots)
    manifest = write_persistent_reconstruction_scene(scene, path / "persistent")
    return read_persistent_reconstruction_scene(manifest), manifest


def main() -> int:
    shutil.rmtree(OUTPUT, ignore_errors=True)
    OUTPUT.mkdir(parents=True, exist_ok=True)

    ready_scene, ready_manifest = _persistent_scene(
        OUTPUT / "ready",
        degrade_gaussian=False,
        request_id="phase8-8f1-ready",
    )
    degraded_scene, _degraded_manifest = _persistent_scene(
        OUTPUT / "degraded",
        degrade_gaussian=True,
        request_id="phase8-8f1-degraded",
    )

    CameraEnumerator.start = lambda self: None  # type: ignore[method-assign]
    app = QApplication.instance() or QApplication(sys.argv)
    window = MainWindow()
    window._mode_tabs.set_active_mode(AppMode.RECONSTRUCTION)
    page = window._reconstruction_page
    page.set_persistent_scene(ready_scene, str(ready_manifest))
    ready = page.appearance_representation_presentation
    ready_labels = [entry.label for entry in ready.entries] if ready else []
    default_selected = page.selected_appearance_representation
    gaussian_selected = bool(default_selected and default_selected.label == "Gaussian")
    gaussian_provider = page._appearance_provider_value.text()
    viewport_after_ready = page._viewport_state.text()

    nerf_index = page._appearance_representation_combo.findText("NeRF")
    if nerf_index >= 0:
        page._appearance_representation_combo.setCurrentIndex(nerf_index)
    nerf_selected = page.selected_appearance_representation
    nerf_switch_ok = bool(
        nerf_selected is not None
        and nerf_selected.label == "NeRF"
        and "nerfstudio" in page._appearance_provider_value.text().lower()
        and page._viewport_state.text() == viewport_after_ready
    )

    page.set_persistent_scene(degraded_scene, "degraded-persistent-scene.json")
    degraded = page.appearance_representation_presentation
    degraded_ok = bool(
        degraded is not None
        and [entry.label for entry in degraded.entries] == ["NeRF"]
        and degraded.missing_kinds == ("gaussian_scene",)
        and "Gaussian missing" in page._appearance_state_value.text()
    )

    # Restore the full scene for the screenshot and final output.
    page.set_persistent_scene(ready_scene, str(ready_manifest))
    nerf_index = page._appearance_representation_combo.findText("NeRF")
    if nerf_index >= 0:
        page._appearance_representation_combo.setCurrentIndex(nerf_index)
    page._readiness_badge.setText("F1 APPEARANCE CONTROLS • PERSISTED 8.7F5")
    page._viewport_state.setText("F1 metadata selection only • no neural render")
    window.resize(1366, 768)
    window.show()
    app.processEvents()
    page._settings_scroll.verticalScrollBar().setValue(page._settings_scroll.verticalScrollBar().maximum())
    app.processEvents()
    screenshot = OUTPUT / "reconstruction_f1_appearance_controls.png"
    screenshot_ok = bool(window.grab().save(str(screenshot)))

    final = page.appearance_representation_presentation
    selected = page.selected_appearance_representation
    authority_ok = bool(
        final is not None
        and final.canonical_geometry_authoritative
        and not final.certified_metrology
        and all(entry.authority_text == "Appearance only" for entry in final.entries)
        and all(not entry.canonical_geometry_authoritative for entry in final.entries)
        and all(not entry.certified_metrology for entry in final.entries)
        and "canonical geometry remains authoritative" in page._appearance_authority_value.text()
    )
    metadata_only_ok = bool(nerf_switch_ok)

    summary = {
        "phase": "8.8F1",
        "pass": bool(
            ready is not None
            and ready_labels == ["Gaussian", "NeRF"]
            and gaussian_selected
            and "gsplat" in gaussian_provider.lower()
            and nerf_switch_ok
            and degraded_ok
            and authority_ok
            and metadata_only_ok
            and screenshot_ok
        ),
        "source_phase8_7f5_persistent_scene_roundtrip": True,
        "source_scene_state": getattr(getattr(ready_scene, "state", None), "value", ""),
        "available_representations": ready_labels,
        "default_selected_representation": default_selected.label if default_selected else "",
        "selected_representation": selected.label if selected else "",
        "gaussian_provider_ui": gaussian_provider,
        "selected_provider_ui": page._appearance_provider_value.text(),
        "render_ui": page._appearance_render_value.text(),
        "scale_ui": page._appearance_scale_value.text(),
        "state_ui": page._appearance_state_value.text(),
        "authority_ui": page._appearance_authority_value.text(),
        "degraded_scene_gaussian_missing_explicit": degraded_ok,
        "external_backend_executed_by_f1": False,
        "render_executed_by_f1": False,
        "representation_view_switch_executed_by_f1": False,
        "resource_probe_executed_by_f1": False,
        "canonical_geometry_authoritative": bool(final and final.canonical_geometry_authoritative),
        "appearance_canonical_geometry_authoritative": False,
        "certified_metrology": False,
        "screenshot": str(screenshot),
        "errors": [],
    }
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    window.close()
    app.processEvents()

    print("Phase 8.8F1 appearance representation controls smoke: " + ("PASS" if summary["pass"] else "FAIL"))
    print(f"available={summary['available_representations']} selected={summary['selected_representation']!r}")
    print(f"authority={summary['authority_ui']!r}")
    print(f"screenshot={screenshot}")
    print(f"summary={OUTPUT / 'summary.json'}")
    return 0 if summary["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
