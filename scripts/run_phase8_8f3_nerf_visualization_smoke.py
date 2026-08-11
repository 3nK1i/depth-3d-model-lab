#!/usr/bin/env python3
"""Decisive F3 smoke: persisted 8.7 NeRF preview artifact -> Reconstruction UI."""

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
    NerfSceneArtifact,
    persistent_scene_from_pipeline_result,
    read_persistent_reconstruction_scene,
    write_persistent_reconstruction_scene,
)
from tests.reconstruction_f5_helpers import run_f5_fixture

OUTPUT = Path("/tmp/phase8_8f3_nerf_visualization")


def _write_preview(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", (800, 450), (8, 11, 17))
    draw = ImageDraw.Draw(image)
    draw.rectangle((165, 90, 635, 360), outline=(85, 150, 220), width=5)
    draw.ellipse((275, 115, 525, 365), fill=(58, 105, 175))
    draw.line((190, 325, 610, 135), fill=(160, 195, 225), width=4)
    draw.text((18, 18), "F3 backend-authored NeRF preview fixture", fill=(195, 205, 220))
    image.save(path)


def _scene_with_preview(root: Path):
    fixture, _providers, result, snapshots = run_f5_fixture(
        root / "source",
        request_id="phase8-8f3-smoke",
    )
    nerf = fixture.store.find_unique(NerfSceneArtifact)
    preview = Path(nerf.artifact_dir) / "preview.png"
    _write_preview(preview)
    evidence = Path(nerf.artifact_dir) / "e2_runtime_evidence.json"
    evidence.write_text(
        json.dumps(
            {
                "preview_path": str(preview.resolve()),
                "training_completed_without_oom": True,
                "qualification_training_steps": nerf.training_steps,
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

    nerf_index = page._appearance_representation_combo.findText("NeRF")
    gaussian_index = page._appearance_representation_combo.findText("Gaussian")
    if nerf_index >= 0:
        page._appearance_representation_combo.setCurrentIndex(nerf_index)
    app.processEvents()
    presentation = page.nerf_visualization_presentation
    selected_before = page.selected_appearance_representation
    viewport_before = page._viewport_state.text()

    page._show_nerf_preview()
    app.processEvents()
    visible = page.nerf_preview_visible
    exact_preview = page._appearance_preview.source_path == str(preview_path)
    viewport_nerf = page._viewport_state.text() == "NeRF preview • appearance only"

    if gaussian_index >= 0:
        page._appearance_representation_combo.setCurrentIndex(gaussian_index)
    app.processEvents()
    selection_change_hides = bool(
        page.selected_appearance_representation is not None
        and page.selected_appearance_representation.label == "Gaussian"
        and not page.nerf_preview_visible
    )

    if nerf_index >= 0:
        page._appearance_representation_combo.setCurrentIndex(nerf_index)
    page._show_nerf_preview()
    app.processEvents()
    restored = bool(page.nerf_preview_visible and page._appearance_preview.source_path == str(preview_path))

    page._hide_nerf_preview()
    preview_path.unlink()
    page._refresh_nerf_visualization_controls()
    missing_explicit = bool(
        page.nerf_visualization_presentation is not None
        and not page.nerf_visualization_presentation.ready
        and "missing" in page.nerf_visualization_presentation.state_text.lower()
        and not page._nerf_show_button.isEnabled()
    )

    _write_preview(preview_path)
    page._refresh_nerf_visualization_controls()
    page._show_nerf_preview()
    page._readiness_badge.setText("F3 NERF VISUALIZATION • PERSISTED 8.7 ARTIFACT")
    window.resize(1366, 768)
    window.show()
    app.processEvents()
    page._settings_scroll.verticalScrollBar().setValue(page._settings_scroll.verticalScrollBar().maximum())
    app.processEvents()
    screenshot = OUTPUT / "reconstruction_f3_nerf_visualization.png"
    screenshot_ok = bool(window.grab().save(str(screenshot)))

    presentation = page.nerf_visualization_presentation
    authority_ok = bool(
        presentation is not None
        and not presentation.canonical_geometry_authoritative
        and not presentation.certified_metrology
        and page._nerf_authority_value.text() == "Appearance only • canonical geometry remains authoritative"
    )

    summary = {
        "phase": "8.8F3",
        "pass": bool(
            presentation is not None
            and presentation.ready
            and selected_before is not None
            and selected_before.label == "NeRF"
            and visible
            and exact_preview
            and viewport_nerf
            and selection_change_hides
            and restored
            and missing_explicit
            and authority_ok
            and screenshot_ok
        ),
        "source_phase8_7f5_persistent_scene_roundtrip": True,
        "source_nerf_artifact_id": presentation.artifact_id if presentation else "",
        "source_nerf_preview_path": str(preview_path),
        "nerf_preview_discovered": bool(presentation and presentation.ready),
        "nerf_preview_visible": bool(page.nerf_preview_visible),
        "exact_preview_artifact_displayed": exact_preview,
        "selection_change_hides_stale_nerf_preview": selection_change_hides,
        "nerf_selection_restore": restored,
        "missing_preview_explicit": missing_explicit,
        "external_backend_executed_by_f3": False,
        "nerf_checkpoint_parsed_by_f3": False,
        "render_computed_by_f3": False,
        "representation_switching_executed_by_f3": False,
        "resource_probe_executed_by_f3": False,
        "canonical_geometry_authoritative": bool(getattr(scene, "canonical_geometry_authoritative", False)),
        "appearance_canonical_geometry_authoritative": False,
        "certified_metrology": False,
        "authority_ui": page._nerf_authority_value.text(),
        "state_ui": page._nerf_state_value.text(),
        "viewport_before_preview": viewport_before,
        "viewport_during_preview": page._viewport_state.text(),
        "screenshot": str(screenshot),
        "errors": [],
    }
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    window.close()
    app.processEvents()

    print("Phase 8.8F3 NeRF visualization smoke: " + ("PASS" if summary["pass"] else "FAIL"))
    print(f"artifact={summary['source_nerf_artifact_id']!r} preview={summary['source_nerf_preview_path']}")
    print(f"authority={summary['authority_ui']!r}")
    print(f"screenshot={screenshot}")
    print(f"summary={OUTPUT / 'summary.json'}")
    return 0 if summary["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
