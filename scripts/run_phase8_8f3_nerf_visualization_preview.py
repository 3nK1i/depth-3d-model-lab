#!/usr/bin/env python3
"""Interactive Phase 8.8F3 NeRF visualization preview; no NeRF backend."""

from __future__ import annotations

import json
from pathlib import Path
import shutil
import sys

from PIL import Image, ImageDraw, ImageFilter

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

OUTPUT = Path("/tmp/phase8_8f3_nerf_preview")


def _make_nerf_preview(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    width, height = 960, 540
    image = Image.new("RGB", (width, height), (7, 10, 15))
    glow = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(glow, "RGBA")
    for index in range(10):
        inset = 58 + index * 18
        alpha = max(20, 105 - index * 8)
        draw.rounded_rectangle(
            (inset, 78 + index * 6, width - inset, height - 70 - index * 5),
            radius=20,
            outline=(70, 145, 220, alpha),
            width=4,
        )
    draw.ellipse((350, 145, 610, 405), fill=(55, 115, 190, 80))
    glow = glow.filter(ImageFilter.GaussianBlur(radius=5.0))
    image = Image.alpha_composite(image.convert("RGBA"), glow).convert("RGB")
    overlay = ImageDraw.Draw(image)
    overlay.text((22, 20), "Synthetic qualified NeRF preview artifact", fill=(180, 195, 215))
    overlay.text((22, 42), "F3 display only - no backend launched", fill=(120, 140, 165))
    image.save(path)


def main() -> int:
    shutil.rmtree(OUTPUT, ignore_errors=True)
    OUTPUT.mkdir(parents=True, exist_ok=True)

    fixture, _providers, result, snapshots = run_f5_fixture(
        OUTPUT / "source",
        request_id="phase8-8f3-preview",
    )
    nerf = fixture.store.find_unique(NerfSceneArtifact)
    preview = Path(nerf.artifact_dir) / "preview.png"
    _make_nerf_preview(preview)
    (Path(nerf.artifact_dir) / "e2_runtime_evidence.json").write_text(
        json.dumps(
            {
                "preview_path": str(preview.resolve()),
                "training_completed_without_oom": True,
                "qualification_training_steps": nerf.training_steps,
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
    nerf_index = page._appearance_representation_combo.findText("NeRF")
    if nerf_index >= 0:
        page._appearance_representation_combo.setCurrentIndex(nerf_index)
    page._readiness_badge.setText("F3 NERF VISUALIZATION • NO EXTERNAL BACKEND")
    page._viewport_state.setText("F3 NeRF preview available • press Show preview")
    window.show()
    app.processEvents()
    page._settings_scroll.verticalScrollBar().setValue(page._settings_scroll.verticalScrollBar().maximum())
    app.processEvents()
    return int(app.exec())


if __name__ == "__main__":
    raise SystemExit(main())
