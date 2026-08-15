#!/usr/bin/env python3
"""Real-Qt/VisPy preview for Phase 9.3G advanced reconstruction provider UI."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
import tempfile
import time

os.environ.pop("DEPTH3D_TEST_FAKE_VISPY", None)

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from PyQt6.QtWidgets import QApplication

from app.widgets.reconstruction_page import ReconstructionPage
from core.reconstruction.da3 import DA3_PROVIDER_ID
from tests.phase9_3f_helpers import make_phase9_3f_scene


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/tmp/phase9_3g_dense_provider_preview.png"),
        help="PNG screenshot path",
    )
    parser.add_argument("--show", action="store_true", help="Keep the real preview open.")
    return parser


def _pump(app: QApplication, count: int = 10) -> None:
    for _ in range(count):
        app.processEvents()
        time.sleep(0.02)


def main() -> int:
    args = _parser().parse_args()
    app = QApplication.instance() or QApplication(sys.argv[:1])
    with tempfile.TemporaryDirectory(prefix="phase9_3g_preview_") as tmp:
        root = Path(tmp).resolve()
        scene, _semantic, _surface = make_phase9_3f_scene(
            root / "source", request_id="phase9-3g-preview"
        )
        page = ReconstructionPage()
        page.resize(1366, 768)
        page.set_persistent_scene(scene, "phase9_3g_preview/persistent_reconstruction_scene.json")
        page._advanced_reconstruction_group.set_expanded(True)
        da3_index = page._dense_provider_combo.findData(DA3_PROVIDER_ID)
        if da3_index < 0:
            raise RuntimeError("Phase 9.3G preview did not expose the DA3 provider contract.")
        page._dense_provider_combo.setCurrentIndex(da3_index)
        page.show()
        page._settings_scroll.ensureWidgetVisible(page._advanced_reconstruction_group)
        _pump(app, 15)

        presentation = page.dense_provider_presentation
        entry = page.selected_dense_provider_entry
        canvas = page._reconstruction_canvas
        if presentation is None or not presentation.available or entry is None:
            raise RuntimeError("Phase 9.3G provider presentation is unavailable.")
        if presentation.backend_started:
            raise RuntimeError("Phase 9.3G provider inspection unexpectedly started a backend.")
        if entry.provider_id != DA3_PROVIDER_ID or entry.scene_bound:
            raise RuntimeError("Phase 9.3G DA3 preview must remain a non-bound provider inspection entry.")
        if not canvas.canonical_mesh_loaded:
            raise RuntimeError("Phase 9.3G preview lost the existing canonical mesh.")
        if scene.certified_metrology or not scene.canonical_geometry_authoritative:
            raise RuntimeError("Phase 9.3G provider inspection changed scene authority.")

        output = args.output.expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        if not page.grab().save(str(output), "PNG"):
            raise RuntimeError(f"Unable to save preview screenshot: {output}")

        print(f"Phase 9.3G advanced reconstruction provider preview: {output}")
        print(
            f"providers={len(presentation.entries)} selected={entry.provider_id} "
            f"scene_bound={entry.scene_bound} canonical_surface={canvas.canonical_mesh_loaded}"
        )
        print("Provider selection is inspection-only; it does not replan or execute reconstruction.")
        print("DA3 runtime was not probed or launched; DA3 -> canonical integration remains Phase 9.3H.")
        print("Canonical geometry authority and metrology state remain unchanged.")
        print("No YOLO/SAM2/reconstruction/DA3/Gaussian/NeRF backend was launched by scene loading or provider inspection.")

        if args.show:
            exit_code = app.exec()
            page.shutdown_viewport()
            if canvas.canonical_mesh_loaded:
                raise RuntimeError("Canonical mesh remained loaded after viewport shutdown.")
            page.close()
            print("Canonical/provider presentation resources released on viewport shutdown.")
            return int(exit_code)

        page.shutdown_viewport()
        if canvas.canonical_mesh_loaded:
            raise RuntimeError("Canonical mesh remained loaded after viewport shutdown.")
        page.close()
        print("Canonical/provider presentation resources released on viewport shutdown.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
