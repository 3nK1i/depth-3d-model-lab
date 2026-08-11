#!/usr/bin/env python3
"""Interactive Phase 8.8F5 resource/fallback UX preview."""

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

OUTPUT = Path("/tmp/phase8_8f5_resource_fallback_preview")


def _write_preview(path: Path, *, label: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", (800, 450), (8, 11, 17))
    draw = ImageDraw.Draw(image)
    draw.rectangle((150, 82, 650, 368), outline=(82, 148, 218), width=5)
    draw.ellipse((280, 118, 520, 358), fill=(56, 104, 174))
    draw.text((18, 18), label, fill=(198, 208, 222))
    draw.text((18, 40), "F5 static preview only • no backend launched", fill=(128, 145, 165))
    image.save(path)


def main() -> int:
    shutil.rmtree(OUTPUT, ignore_errors=True)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    fixture, _providers, result, snapshots = run_f5_fixture(
        OUTPUT / "source", request_id="phase8-8f5-preview"
    )
    gaussian = fixture.store.find_unique(GaussianSceneArtifact)
    nerf = fixture.store.find_unique(NerfSceneArtifact)
    gaussian_preview = (Path(gaussian.artifact_dir) / "preview.png").resolve()
    nerf_preview = (Path(nerf.artifact_dir) / "preview.png").resolve()
    _write_preview(gaussian_preview, label="F5 qualified Gaussian appearance preview")
    _write_preview(nerf_preview, label="F5 qualified NeRF appearance preview")
    (Path(gaussian.artifact_dir) / "qualification_metrics.json").write_text(
        json.dumps({"render_written": True, "preview_path": str(gaussian_preview)}),
        encoding="utf-8",
    )
    (Path(nerf.artifact_dir) / "e2_runtime_evidence.json").write_text(
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
    page._readiness_badge.setText("F5 RESOURCE / FALLBACK UX • NO EXTERNAL BACKEND")
    window.resize(1366, 768)
    window.show()
    app.processEvents()
    page._settings_scroll.ensureWidgetVisible(page._appearance_resource_group)
    app.processEvents()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
