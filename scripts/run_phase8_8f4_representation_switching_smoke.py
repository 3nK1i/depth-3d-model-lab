#!/usr/bin/env python3
"""Decisive F4 smoke: switch existing Gaussian/NeRF previews in Reconstruction."""

from __future__ import annotations

import json
from pathlib import Path
import shutil
import sys

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from PyQt6.QtWidgets import QApplication

from app.application_controller import CameraEnumerator
from app.main_window import MainWindow
from app.widgets.mode_tabs import AppMode
from core.reconstruction import (
    GaussianSceneArtifact,
    NerfSceneArtifact,
    persistent_scene_from_pipeline_result,
    read_persistent_reconstruction_scene,
    write_persistent_reconstruction_scene,
)
from tests.reconstruction_f5_helpers import run_f5_fixture

OUTPUT = Path("/tmp/phase8_8f4_representation_switching")


def _write_preview(path: Path, *, label: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", (800, 450), (8, 11, 17))
    draw = ImageDraw.Draw(image)
    draw.rectangle((155, 85, 645, 365), outline=(85, 150, 220), width=5)
    draw.ellipse((280, 120, 520, 360), fill=(58, 105, 175))
    draw.text((18, 18), label, fill=(195, 205, 220))
    image.save(path)


def _activate(page, index: int, app: QApplication) -> None:
    page._appearance_representation_combo.setCurrentIndex(index)
    page._appearance_representation_combo.activated.emit(index)
    app.processEvents()


def main() -> int:
    shutil.rmtree(OUTPUT, ignore_errors=True)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    fixture, _providers, result, snapshots = run_f5_fixture(
        OUTPUT / "source", request_id="phase8-8f4-smoke"
    )
    gaussian = fixture.store.find_unique(GaussianSceneArtifact)
    nerf = fixture.store.find_unique(NerfSceneArtifact)
    gaussian_preview = (Path(gaussian.artifact_dir) / "preview.png").resolve()
    nerf_preview = (Path(nerf.artifact_dir) / "preview.png").resolve()
    _write_preview(gaussian_preview, label="F4 qualified Gaussian preview")
    _write_preview(nerf_preview, label="F4 qualified NeRF preview")
    gaussian_metrics = Path(gaussian.artifact_dir) / "qualification_metrics.json"
    gaussian_metrics.write_text(
        json.dumps({"render_written": True, "preview_path": str(gaussian_preview)}), encoding="utf-8"
    )
    nerf_evidence = Path(nerf.artifact_dir) / "e2_runtime_evidence.json"
    nerf_evidence.write_text(
        json.dumps({"preview_path": str(nerf_preview), "training_completed_without_oom": True}),
        encoding="utf-8",
    )

    scene = persistent_scene_from_pipeline_result(result, payload_snapshots_by_kind=snapshots)
    manifest = write_persistent_reconstruction_scene(scene, OUTPUT / "persistent")
    scene = read_persistent_reconstruction_scene(manifest)

    CameraEnumerator.start = lambda self: None  # type: ignore[method-assign]
    app = QApplication.instance() or QApplication(sys.argv)
    window = MainWindow()
    window._mode_tabs.set_active_mode(AppMode.RECONSTRUCTION)
    page = window._reconstruction_page
    page.set_persistent_scene(scene, str(manifest))
    app.processEvents()

    gaussian_index = page._appearance_representation_combo.findText("Gaussian")
    nerf_index = page._appearance_representation_combo.findText("NeRF")
    initial_canonical = bool(
        gaussian_index >= 0
        and nerf_index >= 0
        and page.active_view_representation == "canonical_geometry"
        and not page.gaussian_preview_visible
        and not page.nerf_preview_visible
    )

    # User activation of the already-selected default is an explicit F4 switch.
    page._appearance_representation_combo.activated.emit(page._appearance_representation_combo.currentIndex())
    app.processEvents()
    gaussian_active = bool(
        page.gaussian_preview_visible
        and not page.nerf_preview_visible
        and page._appearance_preview.source_path == str(gaussian_preview)
        and page.active_view_representation == "gaussian_scene"
    )

    _activate(page, nerf_index, app)
    nerf_active = bool(
        not page.gaussian_preview_visible
        and page.nerf_preview_visible
        and page._appearance_preview.source_path == str(nerf_preview)
        and page.active_view_representation == "nerf_scene"
    )

    _activate(page, gaussian_index, app)
    gaussian_restored = bool(
        page.gaussian_preview_visible
        and not page.nerf_preview_visible
        and page._appearance_preview.source_path == str(gaussian_preview)
    )

    selected_before_hide = page.selected_appearance_representation
    page._gaussian_hide_button.click()
    app.processEvents()
    hide_returns_canonical = bool(
        page.active_view_representation == "canonical_geometry"
        and not page.gaussian_preview_visible
        and selected_before_hide is not None
        and page.selected_appearance_representation is not None
        and page.selected_appearance_representation.artifact_id == selected_before_hide.artifact_id
    )

    # F4 refuses an unavailable selected target and does not choose the other one.
    nerf_preview.unlink()
    page._refresh_nerf_visualization_controls()
    _activate(page, nerf_index, app)
    unavailable_retains_canonical = bool(
        page.active_view_representation == "canonical_geometry"
        and not page.gaussian_preview_visible
        and not page.nerf_preview_visible
        and "NeRF unavailable" in page._appearance_viewport_value.text()
    )

    # Restore and capture the real F4 switched state.
    _write_preview(nerf_preview, label="F4 qualified NeRF preview")
    page._refresh_nerf_visualization_controls()
    _activate(page, nerf_index, app)
    page._readiness_badge.setText("F4 REPRESENTATION SWITCHING • PERSISTED 8.7 ARTIFACTS")
    window.resize(1366, 768)
    window.show()
    app.processEvents()
    page._settings_scroll.verticalScrollBar().setValue(page._settings_scroll.verticalScrollBar().maximum())
    app.processEvents()
    screenshot = OUTPUT / "reconstruction_f4_representation_switching.png"
    screenshot_ok = bool(window.grab().save(str(screenshot)))

    authority_ok = bool(
        getattr(scene, "canonical_geometry_authoritative", False)
        and page._appearance_authority_value.text() == "Appearance only • canonical geometry remains authoritative"
        and not bool(getattr(scene, "certified_metrology", False))
    )

    summary = {
        "phase": "8.8F4",
        "pass": bool(
            initial_canonical
            and gaussian_active
            and nerf_active
            and gaussian_restored
            and hide_returns_canonical
            and unavailable_retains_canonical
            and page.nerf_preview_visible
            and authority_ok
            and screenshot_ok
        ),
        "source_phase8_7f5_persistent_scene_roundtrip": True,
        "initial_view_canonical": initial_canonical,
        "current_selection_reactivation_switches": gaussian_active,
        "gaussian_to_nerf_switch": nerf_active,
        "nerf_to_gaussian_switch": gaussian_restored,
        "hide_returns_canonical": hide_returns_canonical,
        "unavailable_target_retains_canonical": unavailable_retains_canonical,
        "cross_representation_fallback_executed_by_f4": False,
        "external_gaussian_backend_executed_by_f4": False,
        "external_nerf_backend_executed_by_f4": False,
        "render_computed_by_f4": False,
        "resource_probe_executed_by_f4": False,
        "fallback_policy_executed_by_f4": False,
        "canonical_geometry_authoritative": bool(getattr(scene, "canonical_geometry_authoritative", False)),
        "appearance_canonical_geometry_authoritative": False,
        "certified_metrology": False,
        "active_view": page.active_view_representation,
        "viewport_ui": page._appearance_viewport_value.text(),
        "authority_ui": page._appearance_authority_value.text(),
        "screenshot": str(screenshot),
        "errors": [],
    }
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    window.close()
    app.processEvents()

    print("Phase 8.8F4 representation switching smoke: " + ("PASS" if summary["pass"] else "FAIL"))
    print(f"active_view={summary['active_view']!r} viewport={summary['viewport_ui']!r}")
    print(f"authority={summary['authority_ui']!r}")
    print(f"screenshot={screenshot}")
    print(f"summary={OUTPUT / 'summary.json'}")
    return 0 if summary["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
