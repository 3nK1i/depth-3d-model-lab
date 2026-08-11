#!/usr/bin/env python3
"""Interactive Phase 8.8F2 Gaussian visualization preview; no Gaussian backend."""

from __future__ import annotations

import json
from pathlib import Path
import shutil
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

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

OUTPUT = Path("/tmp/phase8_8f2_gaussian_preview")


def _make_gaussian_preview(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    width, height = 960, 540
    image = Image.new("RGB", (width, height), (5, 8, 13))
    glow = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(glow, "RGBA")
    rng = np.random.default_rng(88)
    centers = np.array([[360, 260], [465, 205], [540, 305], [610, 245], [450, 340]], dtype=float)
    for _ in range(650):
        center = centers[int(rng.integers(0, len(centers)))]
        x, y = center + rng.normal(0.0, [68, 52], size=2)
        radius = float(rng.uniform(2.0, 9.0))
        alpha = int(rng.integers(45, 135))
        color = (
            int(rng.integers(70, 135)),
            int(rng.integers(125, 220)),
            int(rng.integers(185, 255)),
            alpha,
        )
        draw.ellipse((x-radius, y-radius, x+radius, y+radius), fill=color)
    glow = glow.filter(ImageFilter.GaussianBlur(radius=1.8))
    image = Image.alpha_composite(image.convert("RGBA"), glow).convert("RGB")
    overlay = ImageDraw.Draw(image)
    overlay.text((22, 20), "Synthetic qualified Gaussian preview artifact", fill=(175, 190, 210))
    overlay.text((22, 42), "F2 display only - no backend launched", fill=(115, 130, 150))
    image.save(path)


def main() -> int:
    shutil.rmtree(OUTPUT, ignore_errors=True)
    OUTPUT.mkdir(parents=True, exist_ok=True)

    fixture, _providers, result, snapshots = run_f5_fixture(
        OUTPUT / "source",
        request_id="phase8-8f2-preview",
    )
    gaussian = fixture.store.find_unique(GaussianSceneArtifact)
    preview = Path(gaussian.artifact_dir) / "preview.png"
    _make_gaussian_preview(preview)
    (Path(gaussian.artifact_dir) / "qualification_metrics.json").write_text(
        json.dumps(
            {
                "backend_kind": "gsplat_external",
                "training_steps": gaussian.training_steps,
                "real_cuda_execution": True,
                "optimization_completed": True,
                "render_written": True,
                "preview_path": str(preview.resolve()),
            },
            indent=2,
        ) + "\n",
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
    page._readiness_badge.setText("F2 GAUSSIAN VISUALIZATION • NO EXTERNAL BACKEND")
    page._viewport_state.setText("F2 Gaussian preview available • press Show preview")
    window.show()
    app.processEvents()
    page._settings_scroll.verticalScrollBar().setValue(page._settings_scroll.verticalScrollBar().maximum())
    app.processEvents()
    return int(app.exec())


if __name__ == "__main__":
    raise SystemExit(main())
