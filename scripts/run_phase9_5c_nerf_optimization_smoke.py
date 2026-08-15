#!/usr/bin/env python3
"""Dependency-light Phase 9.5C NeRF optimization execution contract smoke."""
from __future__ import annotations
import json
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.reconstruction import (
    NerfBackendKind,
    NerfInteractiveProcessPolicy,
    NerfOptimizationProfile,
    NerfWorkerRequest,
    current_process_nerf_heavy_modules,
    materialize_nerf_input_from_da3_persistent_scene,
)
from tests.phase9_4b_helpers import make_phase9_4b_da3_scene


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="phase9_5c_") as tmp:
        root = Path(tmp).resolve()
        scene, _paths, _poses, _intrinsics = make_phase9_4b_da3_scene(root / "fixture")
        bridge = materialize_nerf_input_from_da3_persistent_scene(scene, output_dir=root / "dataset")
        request = NerfWorkerRequest.from_dataset(
            request_id="phase9-5c-smoke",
            backend_kind=NerfBackendKind.NERFSTUDIO_NERFACTO_EXTERNAL,
            profile=NerfOptimizationProfile.EFFICIENT,
            dataset=bridge.dataset,
            dataset_manifest_path=bridge.manifest_path,
            output_dir=(root / "worker-output").resolve(),
            maximum_input_frames=8,
            qualification_training_steps=64,
        )
        policy = NerfInteractiveProcessPolicy()
        heavy = current_process_nerf_heavy_modules()
        passed = bool(
            request.dataset_fingerprint_sha256 == bridge.dataset_fingerprint_sha256
            and request.qualification_training_steps == 64
            and bridge.dataset.metric_geometry_status.value == "arbitrary"
            and bridge.dataset.unit_to_metre is None
            and policy.start_new_session
            and not policy.shell
            and not heavy
        )
        payload = {
            "phase": "9.5C",
            "status": "PASS" if passed else "FAIL",
            "dataset_fingerprint_sha256": bridge.dataset_fingerprint_sha256,
            "frame_count": bridge.frame_count,
            "training_steps": 64,
            "profile": "efficient",
            "nerf_scene_role": "appearance_only",
            "canonical_geometry_authoritative": False,
            "metric_scale_authoritative": False,
            "measurement_authoritative": False,
            "qc_authoritative": False,
            "certified_metrology": False,
            "metric_geometry_status": bridge.dataset.metric_geometry_status.value,
            "unit_to_metre": bridge.dataset.unit_to_metre,
            "production_heavy_modules": list(heavy),
            "real_backend_started": False,
            "optimization_started": False,
            "rendering_started": False,
            "real_execution_deferred_to_target_gpu": True,
            "quality_evaluated": False,
            "performance_qualified": False,
            "process_group_required": policy.start_new_session,
            "shell": policy.shell,
            "termination_order": policy.as_dict()["termination_order"],
        }
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
