#!/usr/bin/env python3
"""Phase 8.7E1 NeRF contract/isolation smoke without a real NeRF install."""
from __future__ import annotations
import json
from pathlib import Path
import shutil
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.reconstruction import (
    DenseSceneProfile,
    MetricGeometryStatus,
    NerfBackendKind,
    NerfOptimizationProfile,
    NerfSceneRole,
    NerfTrainingState,
    NerfUsageScope,
    NerfWorkerProcessSpec,
    NerfWorkerRequest,
    NeuralSceneDataset,
    NeuralSceneFrame,
    SfmScaleStatus,
    SfmWorldAlignmentStatus,
    nerf_backend_provenance,
    nerf_backend_runtime_pin,
    run_nerf_external_worker,
    write_neural_scene_dataset_manifest,
)

OUTPUT = Path('/tmp/phase8_7e1_nerf_contracts')


def _dataset() -> NeuralSceneDataset:
    images = OUTPUT / 'images'
    images.mkdir(parents=True, exist_ok=True)
    frames = []
    for index in range(3):
        image = (images / f'frame-{index}.ppm').resolve()
        image.write_bytes(b'P6\n2 2\n255\n' + bytes([100 + index, 90, 80]) * 4)
        pose = np.eye(4, dtype=np.float64)
        pose[0, 3] = index * 0.05
        frames.append(NeuralSceneFrame(
            frame_id=f'frame-{index}', image_path=str(image), width=2, height=2,
            fx=2.0, fy=2.0, cx=1.0, cy=1.0,
            T_reconstruction_from_camera=pose, pose_layer='optimized_keyframe',
            camera_profile_id='phase8-7e1-camera', timestamp_ns=index,
        ))
    return NeuralSceneDataset(
        session_id='phase8-7e1-smoke',
        scene_profile=DenseSceneProfile.OBJECT_INSPECTION,
        reconstruction_frame_id='reconstruction',
        camera_profile_id='phase8-7e1-camera',
        scale_profile_id='arbitrary',
        metric_geometry_status=MetricGeometryStatus.ARBITRARY,
        scale_status=SfmScaleStatus.ARBITRARY,
        world_alignment_status=SfmWorldAlignmentStatus.ARBITRARY,
        unit_to_metre=None,
        relative_scale_uncertainty=None,
        frames=tuple(frames),
    )


def main() -> int:
    if OUTPUT.exists():
        shutil.rmtree(OUTPUT)
    OUTPUT.mkdir(parents=True)
    dataset = _dataset()
    manifest = write_neural_scene_dataset_manifest(dataset, (OUTPUT / 'dataset.json').resolve())
    request = NerfWorkerRequest.from_dataset(
        request_id='phase8-7e1-smoke',
        backend_kind=NerfBackendKind.DEVELOPMENT_MOCK,
        profile=NerfOptimizationProfile.EFFICIENT,
        dataset=dataset,
        dataset_manifest_path=manifest,
        output_dir=(OUTPUT / 'worker').resolve(),
        maximum_input_frames=16,
    )
    spec = NerfWorkerProcessSpec(
        python_executable=str(Path(sys.executable).resolve()),
        worker_script=str((ROOT / 'scripts/run_nerf_worker_mock.py').resolve()),
        timeout_s=30.0,
        require_isolated_interpreter=False,
    )
    artifact = run_nerf_external_worker(spec, request)
    real = nerf_backend_provenance(NerfBackendKind.NERFSTUDIO_NERFACTO_EXTERNAL)
    pin = nerf_backend_runtime_pin()
    assert artifact.training_state is NerfTrainingState.CONTRACT_ONLY
    assert artifact.scene_role is NerfSceneRole.APPEARANCE_ONLY
    assert real.usage_scope is NerfUsageScope.PERMISSIVE_GENERAL_USE
    assert real.commercial_use_permitted_by_backend_license is True
    summary = {
        'phase': '8.7E1',
        'pass': True,
        'real_backend_kind': NerfBackendKind.NERFSTUDIO_NERFACTO_EXTERNAL.value,
        'method_name': pin.method_name,
        'release_tag': pin.release_tag,
        'revision_prefix': pin.revision_prefix,
        'documented_memory_mib': pin.documented_memory_mib,
        'external_process_required': True,
        'production_environment_modified': False,
        'real_backend_installed': False,
        'real_backend_qualification_deferred_to': '8.7E2',
        'scene_role': artifact.scene_role.value,
        'canonical_geometry_authoritative': artifact.canonical_geometry_authoritative,
        'certified_metrology': artifact.certified_metrology,
        'reconstruction_ui_enabled': False,
    }
    (OUTPUT / 'summary.json').write_text(json.dumps(summary, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    print('Phase 8.7E1 NeRF contracts + external isolation: PASS')
    print(f"backend={summary['real_backend_kind']} method={summary['method_name']}")
    print(f"runtime_pin={pin.release_tag}@{pin.revision_prefix} documented_memory_mib={pin.documented_memory_mib}")
    print('external_process_required=True production_environment_modified=False')
    print('real_backend_installed=False real_backend_qualification_deferred_to=8.7E2')
    print('scene_role=appearance_only canonical_geometry_authoritative=False certified_metrology=False')
    print('reconstruction_ui_enabled=False')
    print(f"summary={OUTPUT / 'summary.json'}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
