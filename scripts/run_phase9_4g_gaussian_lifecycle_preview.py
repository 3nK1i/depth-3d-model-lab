#!/usr/bin/env python3
"""Real PyQt/VisPy 9.4G Gaussian viewer lifecycle/performance validation."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
from statistics import median
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from PyQt6.QtWidgets import QApplication

from app.main_window import MainWindow
from core.reconstruction import current_process_gaussian_heavy_modules, read_persistent_reconstruction_scene
from core.runtime.stress import read_process_rss_mib


def _pump(app: QApplication, count: int = 4) -> None:
    for _ in range(max(1, count)):
        app.processEvents()


def _p95(values: list[float]) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    index = min(len(ordered) - 1, max(0, int(round(0.95 * (len(ordered) - 1)))))
    return float(ordered[index])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene-manifest", required=True)
    parser.add_argument("--cycles", type=int, default=12)
    parser.add_argument("--output", default="/tmp/phase9_4g_gaussian_lifecycle.png")
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()
    cycles = max(1, int(args.cycles))
    manifest = Path(args.scene_manifest).expanduser().resolve()
    scene = read_persistent_reconstruction_scene(manifest)
    before_modules = current_process_gaussian_heavy_modules()
    app = QApplication.instance() or QApplication([])
    baseline_rss = read_process_rss_mib(os.getpid())

    window = MainWindow()
    page = window._reconstruction_page
    page.set_persistent_scene(scene, str(manifest))
    page._gaussian_visualization_group.set_expanded(True)
    _pump(app, 6)

    load_ms: list[float] = []
    apply_ms: list[float] = []
    initial_generation = page._reconstruction_canvas.interactive_gaussian_resource_generation
    initial_releases = page._reconstruction_canvas.interactive_gaussian_release_count

    # Keep non-default controls across every release/reload cycle.
    page._gaussian_density_combo.setCurrentIndex(page._gaussian_density_combo.findData(50))
    page._gaussian_size_slider.setValue(125)
    page._gaussian_opacity_slider.setValue(70)

    for cycle in range(cycles):
        page._show_interactive_gaussian()
        _pump(app, 3)
        health = page.gaussian_runtime_health
        if health is None or not health.resident:
            raise RuntimeError(f"Cycle {cycle}: Gaussian runtime health did not become resident.")
        if page._reconstruction_canvas.interactive_gaussian_display_count < 1:
            raise RuntimeError(f"Cycle {cycle}: no interactive Gaussian splats were displayed.")
        load_ms.append(health.load_latency_ms)
        apply_ms.append(health.apply_latency_ms)
        page._hide_interactive_gaussian()
        _pump(app, 3)
        if page._interactive_gaussian_scene is not None or page._gaussian_display_buffers is not None:
            raise RuntimeError(f"Cycle {cycle}: page retained Gaussian NumPy staging after Hide.")
        if page._reconstruction_canvas.interactive_gaussian_visible:
            raise RuntimeError(f"Cycle {cycle}: canvas retained visible Gaussian resources after Hide.")
        if not page._reconstruction_canvas.canonical_mesh_loaded:
            raise RuntimeError(f"Cycle {cycle}: Gaussian release disturbed canonical picking geometry.")

    expected_releases = initial_releases + cycles
    if page._reconstruction_canvas.interactive_gaussian_release_count != expected_releases:
        raise RuntimeError("VisPy Gaussian marker release count did not match Hide cycles.")
    if page._reconstruction_canvas.interactive_gaussian_resource_generation != initial_generation + cycles:
        raise RuntimeError("VisPy Gaussian marker resource generation did not advance per release.")

    # Finish resident so the screenshot exposes live diagnostics and bounded display.
    page._show_interactive_gaussian()
    _pump(app, 4)
    health = page.gaussian_runtime_health
    if health is None or not health.resident:
        raise RuntimeError("Final Gaussian viewer state is not resident.")
    if page._gaussian_density_combo.currentData() != 50 or page._gaussian_size_slider.value() != 125 or page._gaussian_opacity_slider.value() != 70:
        raise RuntimeError("Presentation controls did not survive resource release/reload cycles.")

    window.show()
    _pump(app, 10)
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if not window.grab().save(str(output), "PNG"):
        raise RuntimeError(f"Unable to save 9.4G lifecycle screenshot: {output}")

    final_rss = read_process_rss_mib(os.getpid())
    after_modules = current_process_gaussian_heavy_modules()
    if after_modules:
        raise RuntimeError(f"Production process imported forbidden Gaussian heavy modules: {after_modules}")

    print(f"Phase 9.4G Gaussian lifecycle preview: {output}")
    print(
        f"cycles={cycles} device={health.device_label} cap={health.max_display_gaussians:,} "
        f"displayed={health.controlled_display_count:,}"
    )
    print(
        f"median_safe_load_ms={median(load_ms):.2f} p95_safe_load_ms={_p95(load_ms):.2f} "
        f"median_display_apply_ms={median(apply_ms):.2f} p95_display_apply_ms={_p95(apply_ms):.2f}"
    )
    print(
        f"canvas_releases={page._reconstruction_canvas.interactive_gaussian_release_count} "
        f"resource_generation={page._reconstruction_canvas.interactive_gaussian_resource_generation}"
    )
    print(f"rss_baseline_mib={baseline_rss} rss_final_mib={final_rss} (advisory only)")
    print("Hide/Preview/scene switch release interactive Gaussian CPU staging and VisPy marker buffers; reload is on demand.")
    print("Timings are UI-side safe-load/display-apply measurements only; GPU VRAM is not inferred.")
    print("Canonical geometry remains picking/measurement authority; metrology remains uncertified.")
    print("No gsplat/Graphdeco training or render backend was started.")

    if args.show:
        rc = app.exec()
        window.close()
        return int(rc)
    window.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
