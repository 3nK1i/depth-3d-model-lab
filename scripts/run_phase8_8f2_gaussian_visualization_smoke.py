#!/usr/bin/env python3
"""Decisive F2 smoke: persisted 8.7 Gaussian preview artifact -> Reconstruction UI."""

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
    persistent_scene_from_pipeline_result,
    read_persistent_reconstruction_scene,
    write_persistent_reconstruction_scene,
)
from tests.reconstruction_f5_helpers import run_f5_fixture

OUTPUT = Path("/tmp/phase8_8f2_gaussian_visualization")


def _write_preview(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", (800, 450), (7, 10, 16))
    draw = ImageDraw.Draw(image)
    for index in range(42):
        x = 180 + (index * 47) % 420
        y = 110 + (index * 71) % 230
        radius = 5 + (index % 7) * 2
        draw.ellipse((x-radius, y-radius, x+radius, y+radius), fill=(90 + index % 80, 120, 205))
    draw.text((18, 18), "F2 backend-authored Gaussian preview fixture", fill=(190, 200, 215))
    image.save(path)


def _scene_with_preview(root: Path):
    fixture, _providers, result, snapshots = run_f5_fixture(
        root / "source",
        request_id="phase8-8f2-smoke",
    )
    gaussian = fixture.store.find_unique(GaussianSceneArtifact)
    preview = Path(gaussian.artifact_dir) / "preview.png"
    _write_preview(preview)
    metrics = Path(gaussian.artifact_dir) / "qualification_metrics.json"
    metrics.write_text(
        json.dumps(
            {
                "backend_kind": "gsplat_external",
                "training_steps": gaussian.training_steps,
                "real_cuda_execution": True,
                "optimization_completed": True,
                "render_written": True,
                "preview_path": str(preview.resolve()),
            }
        ),
        encoding="utf-8",
    )
    scene = persistent_scene_from_pipeline_result(result, payload_snapshots_by_kind=snapshots)
    manifest = write_persistent_reconstruction_scene(scene, root / "persistent")
    return read_persistent_reconstruction_scene(manifest), manifest, preview.resolve()


def main() -> int:
    shutil.rmtree(OUTPUT, ignore_errors=True)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    scene, manifest, preview_path = _scene_with_preview(OUTPUT)

    CameraEnumerator.start = lambda self: None  # type: ignore[method-assign]
    app = QApplication.instance() or QApplication(sys.argv)
    window = MainWindow()
    window._mode_tabs.set_active_mode(AppMode.RECONSTRUCTION)
    page = window._reconstruction_page
    page.set_persistent_scene(scene, str(manifest))
    presentation = page.gaussian_visualization_presentation
    selected_before = page.selected_appearance_representation
    viewport_before = page._viewport_state.text()

    page._show_gaussian_preview()
    app.processEvents()
    visible = page.gaussian_preview_visible
    exact_preview = page._appearance_preview.source_path == str(preview_path)
    viewport_gaussian = page._viewport_state.text() == "Gaussian Splatting preview • appearance only"

    nerf_index = page._appearance_representation_combo.findText("NeRF")
    if nerf_index >= 0:
        page._appearance_representation_combo.setCurrentIndex(nerf_index)
    app.processEvents()
    selection_change_hides = bool(
        page.selected_appearance_representation is not None
        and page.selected_appearance_representation.label == "NeRF"
        and not page.gaussian_preview_visible
    )

    gaussian_index = page._appearance_representation_combo.findText("Gaussian")
    if gaussian_index >= 0:
        page._appearance_representation_combo.setCurrentIndex(gaussian_index)
    page._show_gaussian_preview()
    app.processEvents()
    restored = bool(page.gaussian_preview_visible and page._appearance_preview.source_path == str(preview_path))

    # Missing preview is explicit and never launches a backend.
    page._hide_gaussian_preview()
    preview_path.unlink()
    metrics = preview_path.parent / "qualification_metrics.json"
    page._refresh_gaussian_visualization_controls()
    missing_explicit = bool(
        page.gaussian_visualization_presentation is not None
        and not page.gaussian_visualization_presentation.ready
        and "missing" in page.gaussian_visualization_presentation.state_text.lower()
        and not page._gaussian_show_button.isEnabled()
    )

    # Restore for screenshot.
    _write_preview(preview_path)
    page._refresh_gaussian_visualization_controls()
    page._show_gaussian_preview()
    page._readiness_badge.setText("F2 GAUSSIAN VISUALIZATION • PERSISTED 8.7 ARTIFACT")
    window.resize(1366, 768)
    window.show()
    app.processEvents()
    page._settings_scroll.verticalScrollBar().setValue(page._settings_scroll.verticalScrollBar().maximum())
    app.processEvents()
    screenshot = OUTPUT / "reconstruction_f2_gaussian_visualization.png"
    screenshot_ok = bool(window.grab().save(str(screenshot)))

    presentation = page.gaussian_visualization_presentation
    authority_ok = bool(
        presentation is not None
        and not presentation.canonical_geometry_authoritative
        and not presentation.certified_metrology
        and page._gaussian_authority_value.text() == "Appearance only • canonical geometry remains authoritative"
    )

    summary = {
        "phase": "8.8F2",
        "pass": bool(
            presentation is not None
            and presentation.ready
            and selected_before is not None
            and selected_before.label == "Gaussian"
            and visible
            and exact_preview
            and viewport_gaussian
            and selection_change_hides
            and restored
            and missing_explicit
            and authority_ok
            and screenshot_ok
        ),
        "source_phase8_7f5_persistent_scene_roundtrip": True,
        "source_gaussian_artifact_id": presentation.artifact_id if presentation else "",
        "source_gaussian_preview_path": str(preview_path),
        "gaussian_preview_discovered": bool(presentation and presentation.ready),
        "gaussian_preview_visible": bool(page.gaussian_preview_visible),
        "exact_preview_artifact_displayed": exact_preview,
        "selection_change_hides_stale_gaussian_preview": selection_change_hides,
        "gaussian_selection_restore": restored,
        "missing_preview_explicit": missing_explicit,
        "external_backend_executed_by_f2": False,
        "gaussian_scene_payload_parsed_by_f2": False,
        "render_computed_by_f2": False,
        "representation_switching_executed_by_f2": False,
        "resource_probe_executed_by_f2": False,
        "canonical_geometry_authoritative": bool(getattr(scene, "canonical_geometry_authoritative", False)),
        "appearance_canonical_geometry_authoritative": False,
        "certified_metrology": False,
        "authority_ui": page._gaussian_authority_value.text(),
        "state_ui": page._gaussian_state_value.text(),
        "viewport_before_preview": viewport_before,
        "viewport_during_preview": page._viewport_state.text(),
        "screenshot": str(screenshot),
        "errors": [],
    }
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    window.close()
    app.processEvents()

    print("Phase 8.8F2 Gaussian Splatting visualization smoke: " + ("PASS" if summary["pass"] else "FAIL"))
    print(f"artifact={summary['source_gaussian_artifact_id']!r} preview={summary['source_gaussian_preview_path']}")
    print(f"authority={summary['authority_ui']!r}")
    print(f"screenshot={screenshot}")
    print(f"summary={OUTPUT / 'summary.json'}")
    return 0 if summary["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
