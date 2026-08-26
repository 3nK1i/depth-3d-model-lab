#!/usr/bin/env python3
"""I3.2 promoted persistent Dense RGB Reconstruction-workspace sanity check."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from PyQt6.QtWidgets import QApplication

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.widgets.reconstruction_page import ReconstructionPage  # noqa: E402


def _args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--i3-2-dir", type=Path, default=Path("/tmp/eia_i3_2_second_scene_promotion"))
    p.add_argument("--show", action="store_true")
    return p.parse_args()


def main() -> int:
    a = _args(); d = a.i3_2_dir.expanduser().resolve()
    s = json.loads((d / "i3_2_summary.json").read_text(encoding="utf-8"))
    if not bool(s.get("automated_pass")):
        raise RuntimeError("I3.2 UI sanity requires automated_pass=True.")
    manifest = Path(str(s["artifacts"]["persistent_scene_manifest"])).expanduser().resolve()
    expected_ext = int(s["geometry"]["promoted_extended_points"])
    expected_core = int(s["geometry"]["promoted_core_points"])
    app = QApplication.instance() or QApplication(sys.argv[:1]); page = ReconstructionPage(); page.resize(1366, 860)

    page.set_persistent_dense_rgb_scene(manifest, representation="extended", fit_scene=True)
    ext = page.dense_rgb_presentation
    if ext is None or ext.point_count != expected_ext or not page._reconstruction_canvas.dense_rgb_points_loaded:
        raise RuntimeError("I3.2 Extended UI load mismatch.")
    toggle = page._dense_rgb_scene_layer_toggle
    if not toggle.isEnabled() or not toggle.isChecked():
        raise RuntimeError("I3.2 Dense RGB scene-layer toggle did not enable/check after load.")
    toggle.setChecked(False)
    if page._reconstruction_canvas.dense_rgb_points_visible:
        raise RuntimeError("I3.2 Dense RGB layer OFF did not hide points.")
    toggle.setChecked(True)
    if not page._reconstruction_canvas.dense_rgb_points_visible:
        raise RuntimeError("I3.2 Dense RGB layer ON did not restore points.")

    page.set_persistent_dense_rgb_scene(manifest, representation="core", fit_scene=False)
    core = page.dense_rgb_presentation
    if core is None or core.point_count != expected_core or page.dense_rgb_representation != "core":
        raise RuntimeError("I3.2 Core representation switch mismatch.")
    page._fit_dense_rgb_scene()
    page.clear_persistent_dense_rgb_scene()
    if page._reconstruction_canvas.dense_rgb_points_loaded or page.dense_rgb_presentation is not None:
        raise RuntimeError("I3.2 unload did not clear Dense RGB scene.")
    page.set_persistent_dense_rgb_scene(manifest, representation="extended", fit_scene=True)
    final = page.dense_rgb_presentation
    if final is None or final.point_count != expected_ext:
        raise RuntimeError("I3.2 reload after unload failed.")

    result = {
        "phase": "I3.2-UI", "pass": True, "status": "pass", "scene_id": final.scene_id,
        "scene_fingerprint_sha256": final.scene_fingerprint_sha256,
        "extended_points": expected_ext, "core_points": expected_core,
        "checks": {"load_extended": True, "scene_layer_off_on": True, "switch_core": True, "fit": True, "unload": True, "reload_extended": True},
        "backend_started": False,
        "authority": {"metric_scale_claimed": False, "measurement_authoritative": False, "surface_mesh_generated": False},
    }
    out = d / "i3_2_ui_sanity.json"; out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("I3.2 Reconstruction UI sanity: PASS")
    print(f"scene_id={final.scene_id} extended={expected_ext} core={expected_core}")
    print("load=True layer_off_on=True core_switch=True fit=True unload_reload=True backend_started=False")
    print("metric_scale_claimed=False measurement_authoritative=False surface_mesh_generated=False")
    print(f"summary={out}")
    if not a.show:
        page.shutdown_viewport(); page.deleteLater(); app.processEvents(); return 0
    page.show(); page.raise_(); page.activateWindow(); return int(app.exec())


if __name__ == "__main__":
    raise SystemExit(main())
