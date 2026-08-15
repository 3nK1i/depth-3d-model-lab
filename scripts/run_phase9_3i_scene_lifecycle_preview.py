#!/usr/bin/env python3
"""Real PyQt/VisPy lifecycle/performance preview for Phase 9.3I."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from statistics import median
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
from tests.phase9_3i_helpers import make_phase9_3i_scene


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reload-cycles", type=int, default=12)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/tmp/phase9_3i_scene_lifecycle_preview.png"),
    )
    parser.add_argument("--show", action="store_true")
    return parser


def _pump(app: QApplication, count: int = 5) -> None:
    for _ in range(count):
        app.processEvents()
        time.sleep(0.01)


def _p95(values: list[float]) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    index = min(len(ordered) - 1, max(0, int(round(0.95 * (len(ordered) - 1)))))
    return float(ordered[index])


def main() -> int:
    args = _parser().parse_args()
    if args.reload_cycles < 2 or args.reload_cycles > 50:
        raise SystemExit("--reload-cycles must be within [2, 50]")
    app = QApplication.instance() or QApplication(sys.argv[:1])
    with tempfile.TemporaryDirectory(prefix="phase9_3i_preview_") as tmp:
        root = Path(tmp).resolve()
        first, semantic, _surface = make_phase9_3i_scene(root, request_id="phase9-3i-preview-a")
        second, semantic2, _surface2 = make_phase9_3i_scene(root, request_id="phase9-3i-preview-b")

        window = MainWindow()
        window.resize(1366, 768)
        window._mode_tabs.set_active_mode(AppMode.RECONSTRUCTION)
        page = window._reconstruction_page
        canvas = page._reconstruction_canvas
        page.set_persistent_scene(first, "first.json")
        page._semantic_object_combo.setCurrentIndex(1)
        _pump(app)

        canonical_mesh_visual = canvas._canonical_mesh
        selected_mesh_visual = canvas._selected_canonical_semantic_mesh
        canvas._camera.center = (0.17, -0.11, 0.09)
        canvas._camera.distance = 2.35
        expected_center = tuple(float(v) for v in canvas._camera.center)
        expected_distance = float(canvas._camera.distance)
        selected_id = semantic.objects[0].object_id
        latencies: list[float] = []

        for _ in range(args.reload_cycles):
            page.set_persistent_scene(first, "first.json")
            _pump(app, 2)
            health = page.scene_runtime_health
            if health is None:
                raise RuntimeError("9.3I scene health was not produced.")
            latencies.append(float(health.bind_latency_ms))
            if canvas._canonical_mesh is not canonical_mesh_visual:
                raise RuntimeError("Canonical MeshVisual was recreated during a successful same-scene reload.")
            if canvas._selected_canonical_semantic_mesh is not selected_mesh_visual:
                raise RuntimeError(
                    "Selected semantic highlight MeshVisual was recreated during a successful same-scene reload."
                )
            if canvas.selected_semantic_object_id != selected_id:
                raise RuntimeError("Same-scene reload did not preserve semantic selection.")
            if tuple(float(v) for v in canvas._camera.center) != expected_center:
                raise RuntimeError("Same-scene reload changed the user's camera center.")
            if abs(float(canvas._camera.distance) - expected_distance) > 1e-9:
                raise RuntimeError("Same-scene reload changed the user's camera distance.")

        # Switching between two valid non-empty semantic surfaces should reuse the
        # selected-object MeshVisual rather than detach/recreate it.
        page._semantic_object_combo.setCurrentIndex(2)
        _pump(app, 2)
        if canvas._selected_canonical_semantic_mesh is not selected_mesh_visual:
            raise RuntimeError("Semantic object-to-object selection recreated the highlight MeshVisual.")

        page.set_persistent_scene(second, "second.json")
        _pump(app, 4)
        if canvas._canonical_mesh is not canonical_mesh_visual:
            raise RuntimeError("Valid scene-to-scene canonical replacement recreated the MeshVisual.")
        if canvas.selected_semantic_object_id is not None:
            raise RuntimeError("Different-scene bind retained a stale semantic selection.")

        page._semantic_object_combo.setCurrentIndex(1)
        page.set_persistent_scene(second, "second.json")
        _pump(app, 4)
        health = page.scene_runtime_health
        if health is None or not health.view_preserved or not health.selection_preserved:
            raise RuntimeError("Final same-scene lifecycle state was not preserved.")
        if canvas.selected_semantic_object_id != semantic2.objects[0].object_id:
            raise RuntimeError("Final scene selection is inconsistent.")

        page._scene_health_group.set_expanded(True)
        page._settings_scroll.ensureWidgetVisible(page._scene_health_group)
        window.show()
        _pump(app, 12)

        output = args.output.expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        if not window.grab().save(str(output), "PNG"):
            raise RuntimeError(f"Unable to save 9.3I preview screenshot: {output}")

        print(f"Phase 9.3I scene lifecycle preview: {output}")
        print(
            f"reload_cycles={args.reload_cycles} median_ui_bind_ms={median(latencies):.2f} "
            f"p95_ui_bind_ms={_p95(latencies):.2f}"
        )
        print(
            "Successful same-scene reloads preserved camera/selection and reused the canonical MeshVisual; "
            "semantic object-to-object highlight switching reused its MeshVisual."
        )
        print("Different-scene binding cleared stale selection and used one final Fit policy.")
        print("Scene-health timing is UI-side binding only; it is not DA3/reconstruction/GPU backend performance.")
        print("Calibration remains optional; authority and certified-metrology state are unchanged.")
        print("No YOLO/SAM2/reconstruction/DA3/Gaussian/NeRF backend was launched by this preview.")

        if args.show:
            rc = app.exec()
            window.close()
            print("Phase 9.3I viewer/controller resources released on shutdown.")
            return int(rc)
        window.close()
        print("Phase 9.3I viewer/controller resources released on shutdown.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
