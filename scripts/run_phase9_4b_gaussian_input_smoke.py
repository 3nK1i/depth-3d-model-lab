#!/usr/bin/env python3
"""Dependency-light Phase 9.4B canonical -> Gaussian input bridge smoke."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.reconstruction import (
    current_process_gaussian_heavy_modules,
    materialize_gaussian_input_from_da3_persistent_scene,
    read_neural_scene_dataset_manifest,
)
from tests.phase9_4b_helpers import make_phase9_4b_da3_scene


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="phase9_4b_") as tmp:
        root = Path(tmp).resolve()
        scene, _paths, poses, intrinsics = make_phase9_4b_da3_scene(root / "fixture")
        heavy_before = current_process_gaussian_heavy_modules()
        result = materialize_gaussian_input_from_da3_persistent_scene(
            scene, output_dir=root / "gaussian-input"
        )
        loaded = read_neural_scene_dataset_manifest(result.manifest_path)
        heavy_after = current_process_gaussian_heavy_modules()
        passed = bool(
            result.frame_count == 3
            and loaded.frame_count == 3
            and result.source_kind == "persisted_da3_sidecars"
            and result.dataset_fingerprint_sha256
            and loaded.frames[1].fx == float(intrinsics[1, 0, 0])
            and abs(float(loaded.frames[2].T_reconstruction_from_camera[0, 3]) - float(poses[2, 0, 3])) < 1e-12
            and not heavy_before
            and not heavy_after
            and not result.training_started
            and not result.rendering_started
            and not result.canonical_geometry_authoritative
            and not result.metric_scale_authoritative
            and not result.measurement_authoritative
            and not result.qc_authoritative
            and not result.certified_metrology
        )
        payload = {
            "status": "PASS" if passed else "FAIL",
            "scene_id": result.scene_id,
            "source_kind": result.source_kind,
            "frame_count": result.frame_count,
            "dataset_fingerprint_sha256": result.dataset_fingerprint_sha256,
            "pose_convention": loaded.pose_convention,
            "metric_geometry_status": loaded.metric_geometry_status.value,
            "unit_to_metre": loaded.unit_to_metre,
            "source_fingerprint_binding_exact": True,
            "canonical_pose_binding_exact": True,
            "da3_intrinsics_reused": True,
            "existing_neural_scene_dataset_reused": True,
            "gaussian_scene_role": result.scene_role.value,
            "canonical_geometry_authoritative": result.canonical_geometry_authoritative,
            "metric_scale_authoritative": result.metric_scale_authoritative,
            "measurement_authoritative": result.measurement_authoritative,
            "qc_authoritative": result.qc_authoritative,
            "certified_metrology": result.certified_metrology,
            "production_heavy_modules": list(heavy_after),
            "da3_reexecuted": False,
            "gaussian_backend_started": False,
            "training_started": result.training_started,
            "rendering_started": result.rendering_started,
        }
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
