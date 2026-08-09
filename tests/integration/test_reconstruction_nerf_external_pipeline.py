from __future__ import annotations
from pathlib import Path
import sys
import numpy as np

from core.reconstruction import (
    DenseSceneProfile,
    MetricGeometryStatus,
    NerfBackendKind,
    NerfOptimizationProfile,
    NerfSceneRole,
    NerfTrainingState,
    NerfWorkerProcessSpec,
    NerfWorkerRequest,
    NeuralSceneDataset,
    NeuralSceneFrame,
    SfmScaleStatus,
    SfmWorldAlignmentStatus,
    run_nerf_external_worker,
    write_neural_scene_dataset_manifest,
)

ROOT = Path(__file__).resolve().parents[2]


def test_neural_scene_to_external_nerf_contract_pipeline(tmp_path: Path) -> None:
    image = (tmp_path / 'frame.ppm').resolve()
    image.write_bytes(b'P6\n2 2\n255\n' + bytes([100, 90, 80]) * 4)
    frames = []
    for index in range(4):
        pose = np.eye(4, dtype=np.float64)
        pose[0, 3] = 0.04 * index
        frames.append(NeuralSceneFrame(
            frame_id=f'frame-{index}', image_path=str(image), width=2, height=2,
            fx=2.0, fy=2.0, cx=1.0, cy=1.0,
            T_reconstruction_from_camera=pose, pose_layer='optimized_keyframe',
            camera_profile_id='camera', timestamp_ns=index,
        ))
    dataset = NeuralSceneDataset(
        session_id='integration-nerf', scene_profile=DenseSceneProfile.OBJECT_INSPECTION,
        reconstruction_frame_id='reconstruction', camera_profile_id='camera', scale_profile_id='arbitrary',
        metric_geometry_status=MetricGeometryStatus.ARBITRARY,
        scale_status=SfmScaleStatus.ARBITRARY, world_alignment_status=SfmWorldAlignmentStatus.ARBITRARY,
        unit_to_metre=None, relative_scale_uncertainty=None, frames=tuple(frames),
    )
    manifest = write_neural_scene_dataset_manifest(dataset, (tmp_path / 'dataset.json').resolve())
    request = NerfWorkerRequest.from_dataset(
        request_id='integration-request', backend_kind=NerfBackendKind.DEVELOPMENT_MOCK,
        profile=NerfOptimizationProfile.EFFICIENT, dataset=dataset,
        dataset_manifest_path=manifest, output_dir=(tmp_path / 'worker').resolve(), maximum_input_frames=16,
    )
    spec = NerfWorkerProcessSpec(
        python_executable=str(Path(sys.executable).resolve()),
        worker_script=str((ROOT / 'scripts/run_nerf_worker_mock.py').resolve()),
        timeout_s=30.0, require_isolated_interpreter=False,
    )
    artifact = run_nerf_external_worker(spec, request)
    assert artifact.input_frame_count == 4
    assert artifact.training_state is NerfTrainingState.CONTRACT_ONLY
    assert artifact.scene_role is NerfSceneRole.APPEARANCE_ONLY
    assert artifact.render_capable is False
    assert artifact.canonical_geometry_authoritative is False
    assert artifact.certified_metrology is False
