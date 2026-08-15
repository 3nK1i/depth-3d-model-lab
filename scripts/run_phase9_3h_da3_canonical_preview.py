#!/usr/bin/env python3
"""Real-Qt/VisPy post-integration preview for Phase 9.3H.

The preview reuses deterministic DA3-shaped evidence and never launches the real
DA3 runtime. Real heavy DA3 execution remains a Phase 9.3J qualification item.
"""

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

from app.main_window import MainWindow
from app.widgets.mode_tabs import AppMode
from core.reconstruction.da3 import DA3_PROVIDER_ID
from tests.phase9_3h_helpers import integrate_phase9_3h_fixture


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/tmp/phase9_3h_da3_canonical_preview.png"),
        help="PNG screenshot path",
    )
    parser.add_argument("--show", action="store_true", help="Keep the real preview open.")
    return parser


def _pump(app: QApplication, count: int = 12) -> None:
    for _ in range(count):
        app.processEvents()
        time.sleep(0.02)


def main() -> int:
    args = _parser().parse_args()
    app = QApplication.instance() or QApplication(sys.argv[:1])
    with tempfile.TemporaryDirectory(prefix="phase9_3h_preview_") as tmp:
        root = Path(tmp).resolve()
        integration, scene_manifest, _scene_root = integrate_phase9_3h_fixture(root / "fixture")
        source_dir = root / "fixture" / "evidence"

        window = MainWindow()
        window.resize(1366, 768)
        window._mode_tabs.set_active_mode(AppMode.RECONSTRUCTION)
        controller = window._reconstruction_controller
        page = window._reconstruction_page
        controller.select_image_sequence(source_dir)
        scene = controller.load_persistent_scene(scene_manifest)
        page._advanced_reconstruction_group.set_expanded(True)
        da3_index = page._dense_provider_combo.findData(DA3_PROVIDER_ID)
        if da3_index < 0:
            raise RuntimeError("9.3H preview did not expose bound DA3 provenance.")
        page._dense_provider_combo.setCurrentIndex(da3_index)
        window.show()
        page._settings_scroll.ensureWidgetVisible(page._advanced_reconstruction_group)
        _pump(app, 18)

        entry = page.selected_dense_provider_entry
        canvas = page._reconstruction_canvas
        if entry is None or not entry.scene_bound or entry.provider_id != DA3_PROVIDER_ID:
            raise RuntimeError("9.3H preview DA3 evidence is not scene-bound.")
        if not page._da3_integrate_button.isEnabled():
            raise RuntimeError("9.3H explicit DA3 action is not enabled for the selected image sequence.")
        if page.selected_camera_calibration_profile is not None:
            raise RuntimeError("9.3H preview unexpectedly requires an operational calibration profile.")
        if not canvas.canonical_mesh_loaded:
            raise RuntimeError("9.3H preview did not load the canonical DA3-derived surface.")
        if page._canonical_camera_presentation is None or page._canonical_camera_presentation.persisted_pose_count != 3:
            raise RuntimeError("9.3H preview did not load persisted DA3 camera poses.")
        if page._semantic_object_combo.isEnabled():
            raise RuntimeError("9.3H preview fabricated semantic objects for the new DA3 scene.")
        if "unavailable" not in page._semantic_structure_value.text().lower():
            raise RuntimeError("9.3H preview did not expose unavailable semantic context truthfully.")
        if not scene.canonical_geometry_authoritative or scene.certified_metrology:
            raise RuntimeError("9.3H preview changed canonical/metrology authority.")
        if integration.surface.unit_to_metre is not None:
            raise RuntimeError("9.3H preview invented metric scale.")

        output = args.output.expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        if not window.grab().save(str(output), "PNG"):
            raise RuntimeError(f"Unable to save 9.3H preview screenshot: {output}")

        print(f"Phase 9.3H DA3 canonical post-integration preview: {output}")
        print(
            f"canonical_vertices={integration.surface.mesh.vertex_count} "
            f"canonical_faces={integration.surface.mesh.face_count} cameras=3 "
            f"da3_scene_bound={entry.scene_bound} integrate_action_enabled={page._da3_integrate_button.isEnabled()}"
        )
        print("Calibration is unavailable in this preview and the canonical scene remains usable at arbitrary scale.")
        print("DA3 remains supporting evidence; canonical dense/surface providers own canonical geometry authority.")
        print("Semantic and inspection/QC context are unavailable for this new scene and were not fabricated.")
        print("The preview did not probe or launch the real DA3 runtime; real DA3 execution remains Phase 9.3J qualification.")

        if args.show:
            exit_code = app.exec()
            window.close()
            print("Phase 9.3H reconstruction/controller/viewport resources released on shutdown.")
            return int(exit_code)

        window.close()
        print("Phase 9.3H reconstruction/controller/viewport resources released on shutdown.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
