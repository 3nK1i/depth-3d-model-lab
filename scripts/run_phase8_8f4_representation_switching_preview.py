#!/usr/bin/env python3
"""Interactive Phase 8.8F4 representation-switching preview; no neural backend."""

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
    GaussianSceneArtifact,
    NerfSceneArtifact,
    persistent_scene_from_pipeline_result,
    read_persistent_reconstruction_scene,
    write_persistent_reconstruction_scene,
)
from tests.reconstruction_f5_helpers import run_f5_fixture

OUTPUT = Path("/tmp/phase8_8f4_representation_switching_preview")


def _make_preview(path: Path, *, label: str, ellipse: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    width, height = 960, 540
    image = Image.new("RGB", (width, height), (7, 10, 15))
    glow = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(glow, "RGBA")
    if ellipse:
        for x, y, radius in ((350, 270, 80), (480, 210, 58), (600, 300, 94), (470, 360, 46)):
            draw.ellipse((x-radius, y-radius, x+radius, y+radius), fill=(80, 145, 220, 95))
    else:
        for index in range(9):
            inset = 72 + index * 18
            draw.rounded_rectangle(
                (inset, 88 + index * 6, width - inset, height - 76 - index * 5),
                radius=20,
                outline=(70, 145, 220, max(24, 110 - index * 9)),
                width=4,
            )
        draw.ellipse((350, 145, 610, 405), fill=(55, 115, 190, 80))
    glow = glow.filter(ImageFilter.GaussianBlur(radius=5.0))
    image = Image.alpha_composite(image.convert("RGBA"), glow).convert("RGB")
    overlay = ImageDraw.Draw(image)
    overlay.text((22, 20), label, fill=(185, 200, 220))
    overlay.text((22, 42), "F4 switches existing previews only - no backend launched", fill=(120, 140, 165))
    image.save(path)


def main() -> int:
    shutil.rmtree(OUTPUT, ignore_errors=True)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    fixture, _providers, result, snapshots = run_f5_fixture(
        OUTPUT / "source", request_id="phase8-8f4-preview"
    )
    gaussian = fixture.store.find_unique(GaussianSceneArtifact)
    nerf = fixture.store.find_unique(NerfSceneArtifact)
    gaussian_preview = Path(gaussian.artifact_dir) / "preview.png"
    nerf_preview = Path(nerf.artifact_dir) / "preview.png"
    _make_preview(gaussian_preview, label="Qualified Gaussian appearance preview", ellipse=True)
    _make_preview(nerf_preview, label="Qualified NeRF appearance preview", ellipse=False)
    (Path(gaussian.artifact_dir) / "qualification_metrics.json").write_text(
        json.dumps({"render_written": True, "preview_path": str(gaussian_preview.resolve())}, indent=2) + "\n",
        encoding="utf-8",
    )
    (Path(nerf.artifact_dir) / "e2_runtime_evidence.json").write_text(
        json.dumps({"preview_path": str(nerf_preview.resolve()), "training_completed_without_oom": True}, indent=2) + "\n",
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
    page._readiness_badge.setText("F4 REPRESENTATION SWITCHING • NO EXTERNAL BACKEND")
    page._viewport_state.setText("F4 ready • activate Gaussian or NeRF in Representation")
    window.show()
    app.processEvents()
    page._settings_scroll.verticalScrollBar().setValue(page._settings_scroll.verticalScrollBar().maximum())
    app.processEvents()
    return int(app.exec())


if __name__ == "__main__":
    raise SystemExit(main())
