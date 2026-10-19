#!/usr/bin/env python3
"""Reload and integrity-check a Phase-G persistent Dense RGB scene bundle.

This command intentionally needs only the persisted G directory.  It does not
read the original F/E3 validation directories and never starts DA3/PyCOLMAP.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.reconstruction.dense_scene_persistence import (  # noqa: E402
    DenseSceneFileRole,
    read_persistent_dense_rgb_scene,
    resolve_dense_scene_reference,
    restore_persistent_dense_geometry,
    verify_persistent_dense_scene_files,
)

_DEFAULT_SCENE_DIR = Path("/tmp/depth3d_g_persistent_dense_rgb_geometry")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-dir", type=Path, default=_DEFAULT_SCENE_DIR)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    scene_dir = args.scene_dir.expanduser().resolve()
    scene = read_persistent_dense_rgb_scene(scene_dir, verify_files=True)
    references = verify_persistent_dense_scene_files(scene, scene_dir)
    extended = restore_persistent_dense_geometry(scene_dir, representation="extended")
    core = restore_persistent_dense_geometry(scene_dir, representation="core")
    extended_ply = resolve_dense_scene_reference(scene_dir, scene, DenseSceneFileRole.PRIMARY_EXTENDED_PLY)
    core_ply = resolve_dense_scene_reference(scene_dir, scene, DenseSceneFileRole.HIGH_CONFIDENCE_CORE_PLY)
    print("G persistent Dense RGB restore: PASS")
    print(f"scene_id={scene.scene_id}")
    print(f"fingerprint={scene.scene_fingerprint_sha256}")
    print(f"views={scene.selected_view_count} orders={'->'.join(map(str, scene.selected_workspace_orders))}")
    print(f"extended_points={extended.point_count} core_points={core.point_count}")
    print(f"file_references_verified={len(references)}")
    print(f"extended_export={extended_ply}")
    print(f"core_export={core_ply}")
    print("restore_source=G_bundle_only DA3=False PyCOLMAP=False metric_scale_claimed=False")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
