#!/usr/bin/env python3
"""Dependency-light Phase 8.7E2 setup/contract smoke (no Nerfstudio install required)."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.reconstruction import (
    DenseSceneProfile,
    MetricGeometryStatus,
    NERFSTUDIO_E2_CUDA,
    NERFSTUDIO_E2_PYTHON,
    NERFSTUDIO_E2_RTX2060_COMPUTE_CAPABILITY,
    NERFSTUDIO_E2_TORCH,
    NERFSTUDIO_E2_TORCHVISION,
    NERFSTUDIO_RELEASE_REVISION_PREFIX,
    NERFSTUDIO_RELEASE_TAG,
    TINY_CUDA_NN_E2_REVISION,
    NeuralSceneDataset,
    NeuralSceneFrame,
    SfmScaleStatus,
    SfmWorldAlignmentStatus,
)
from scripts.nerf_workers.common import export_nerfstudio_dataset


def _frame(tmp: Path, index: int) -> NeuralSceneFrame:
    image = tmp / f"frame_{index:03d}.ppm"
    image.write_bytes(b"P6\n2 2\n255\n" + bytes([32 + index] * 12))
    pose = np.eye(4, dtype=np.float64)
    pose[0, 3] = 0.1 * index
    return NeuralSceneFrame(
        frame_id=f"e2-contract-{index:03d}", image_path=str(image.resolve()),
        width=2, height=2, fx=2.0, fy=2.0, cx=1.0, cy=1.0,
        T_reconstruction_from_camera=pose,
        pose_layer="optimized_keyframe", camera_profile_id="e2-contract-camera",
        timestamp_ns=index,
    )


def main() -> int:
    setup_script = ROOT / "scripts/setup_nerfstudio_backend.sh"
    worker = ROOT / "scripts/nerf_workers/nerfstudio_worker.py"
    for path in (setup_script, worker):
        if not path.is_file():
            raise SystemExit(f"missing E2 file: {path}")

    setup_text = setup_script.read_text(encoding="utf-8")
    required_tokens = (
        'ENV_NAME="eia-nerf-nerfstudio"',
        'TORCH_VERSION="2.1.2+cu118"',
        'TORCHVISION_VERSION="0.16.2+cu118"',
        'CUDA_TOOLKIT_VERSION="11.8.0"',
        'NERFSTUDIO_TAG="v1.1.5"',
        TINY_CUDA_NN_E2_REVISION,
        'CUDA_ARCH="75"',
        'production `depth` environment',
    )
    missing = [token for token in required_tokens if token not in setup_text]
    if missing:
        raise SystemExit(f"E2 setup pin/isolation tokens missing: {missing}")

    with tempfile.TemporaryDirectory(prefix="phase8_7e2_contracts_") as tmp_raw:
        tmp = Path(tmp_raw)
        dataset = NeuralSceneDataset(
            session_id="phase8-7e2-contract-smoke",
            scene_profile=DenseSceneProfile.OBJECT_INSPECTION,
            reconstruction_frame_id="reconstruction",
            camera_profile_id="e2-contract-camera",
            scale_profile_id="arbitrary",
            metric_geometry_status=MetricGeometryStatus.ARBITRARY,
            scale_status=SfmScaleStatus.ARBITRARY,
            world_alignment_status=SfmWorldAlignmentStatus.ARBITRARY,
            unit_to_metre=None,
            relative_scale_uncertainty=None,
            frames=(_frame(tmp, 0), _frame(tmp, 1)),
        )
        transforms = export_nerfstudio_dataset(dataset, tmp / "nerfstudio")
        payload = json.loads(transforms.read_text(encoding="utf-8"))
        converted = np.asarray(payload["frames"][0]["transform_matrix"], dtype=np.float64)
        if converted[1, 1] != -1.0 or converted[2, 2] != -1.0:
            raise SystemExit("OpenCV->Nerfstudio camera-axis conversion failed")
        if len(payload["train_filenames"]) != 1 or len(payload["test_filenames"]) != 1:
            raise SystemExit("explicit E2 train/test split missing")

    summary = {
        "phase": "8.7E2",
        "pass": True,
        "backend": "nerfstudio_nerfacto_external",
        "release_tag": NERFSTUDIO_RELEASE_TAG,
        "revision_prefix": NERFSTUDIO_RELEASE_REVISION_PREFIX,
        "tiny_cuda_nn_revision": TINY_CUDA_NN_E2_REVISION,
        "python": NERFSTUDIO_E2_PYTHON,
        "torch": NERFSTUDIO_E2_TORCH,
        "torchvision": NERFSTUDIO_E2_TORCHVISION,
        "cuda": NERFSTUDIO_E2_CUDA,
        "rtx2060_compute_capability": list(NERFSTUDIO_E2_RTX2060_COMPUTE_CAPABILITY),
        "isolated_environment_required": True,
        "real_backend_qualification_required": True,
        "quality_benchmark_deferred_to": "8.7E3",
        "performance_hardening_deferred_to": "8.7E4",
        "canonical_geometry_authoritative": False,
        "certified_metrology": False,
        "scene_role": "appearance_only",
        "reconstruction_ui_enabled": False,
    }
    out = Path("/tmp/phase8_7e2_nerf_contracts/summary.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("Phase 8.7E2 NeRF qualification contracts: PASS")
    print(f"nerfstudio={NERFSTUDIO_RELEASE_TAG}@{NERFSTUDIO_RELEASE_REVISION_PREFIX}")
    print(f"tiny_cuda_nn={TINY_CUDA_NN_E2_REVISION}")
    print(f"runtime=python{NERFSTUDIO_E2_PYTHON} torch{NERFSTUDIO_E2_TORCH} cuda{NERFSTUDIO_E2_CUDA}")
    print("isolated_environment_required=True production_depth_environment_modified=False")
    print("quality_benchmark_deferred_to=8.7E3 performance_hardening_deferred_to=8.7E4")
    print("scene_role=appearance_only canonical_geometry_authoritative=False certified_metrology=False")
    print(f"summary={out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
