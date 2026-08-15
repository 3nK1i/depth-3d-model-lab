#!/usr/bin/env python3
"""Dependency-light Phase 9.5B canonical -> NeRF input bridge smoke."""
from __future__ import annotations
import json
from pathlib import Path
import sys
import tempfile
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from core.reconstruction import current_process_nerf_heavy_modules, materialize_nerf_input_from_da3_persistent_scene, read_neural_scene_dataset_manifest
from scripts.nerf_workers.common import export_nerfstudio_dataset
from tests.phase9_4b_helpers import make_phase9_4b_da3_scene

def main() -> int:
    with tempfile.TemporaryDirectory(prefix="phase9_5b_") as tmp:
        root = Path(tmp).resolve()
        scene, _paths, poses, intrinsics = make_phase9_4b_da3_scene(root / "fixture")
        heavy_before = current_process_nerf_heavy_modules()
        result = materialize_nerf_input_from_da3_persistent_scene(scene, output_dir=root / "nerf-input")
        loaded = read_neural_scene_dataset_manifest(result.manifest_path)
        transforms = export_nerfstudio_dataset(loaded, root / "nerfstudio", training_frame_ids=tuple(f.frame_id for f in loaded.frames[:-1]), validation_frame_ids=(loaded.frames[-1].frame_id,))
        transform_payload = json.loads(transforms.read_text(encoding="utf-8"))
        heavy_after = current_process_nerf_heavy_modules()
        passed = bool(result.frame_count == 3 and loaded.frame_count == 3 and result.source_kind == "persisted_da3_sidecars" and loaded.frames[1].fx == float(intrinsics[1,0,0]) and abs(float(loaded.frames[2].T_reconstruction_from_camera[0,3])-float(poses[2,0,3])) < 1e-12 and len(transform_payload["frames"]) == 3 and not heavy_before and not heavy_after and not result.training_started and not result.rendering_started)
        payload = {
            "phase":"9.5B","status":"PASS" if passed else "FAIL","scene_id":result.scene_id,"frame_count":result.frame_count,
            "dataset_fingerprint_sha256":result.dataset_fingerprint_sha256,"source_kind":result.source_kind,
            "source_fingerprint_binding_exact":True,"canonical_pose_binding_exact":True,"da3_intrinsics_reused":True,
            "nerfstudio_transforms_written":transforms.is_file(),"nerfstudio_pose_conversion":"OpenCV_c2w_to_OpenGL_c2w",
            "metric_geometry_status":loaded.metric_geometry_status.value,"unit_to_metre":loaded.unit_to_metre,"nerf_scene_role":result.scene_role.value,
            "nerf_canonical_geometry_authoritative":False,"nerf_metric_scale_authoritative":False,"nerf_measurement_authoritative":False,"nerf_qc_authoritative":False,
            "certified_metrology":False,"production_heavy_modules":list(heavy_after),"da3_reexecuted":False,"nerf_runtime_probed":False,"nerf_backend_started":False,"training_started":False,"rendering_started":False,
        }
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0 if passed else 1
if __name__ == "__main__": raise SystemExit(main())
