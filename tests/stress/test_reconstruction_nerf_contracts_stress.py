from __future__ import annotations
from pathlib import Path
import numpy as np

from core.reconstruction import (
    DenseSceneProfile,
    MetricGeometryStatus,
    NerfBackendKind,
    NerfOptimizationProfile,
    NerfWorkerRequest,
    NeuralSceneDataset,
    NeuralSceneFrame,
    SfmScaleStatus,
    SfmWorldAlignmentStatus,
    neural_scene_dataset_fingerprint,
)


def test_large_neural_scene_remains_deterministic_for_nerf_requests(tmp_path: Path) -> None:
    image = (tmp_path / 'frame.ppm').resolve()
    image.write_bytes(b'P6\n2 2\n255\n' + bytes([100, 90, 80]) * 4)
    frames = []
    for index in range(500):
        pose = np.eye(4, dtype=np.float64)
        pose[0, 3] = index * 0.001
        frames.append(NeuralSceneFrame(
            frame_id=f'frame-{index}', image_path=str(image), width=2, height=2,
            fx=2.0, fy=2.0, cx=1.0, cy=1.0,
            T_reconstruction_from_camera=pose, pose_layer='optimized_propagated',
            camera_profile_id='camera', timestamp_ns=index,
        ))
    dataset = NeuralSceneDataset(
        session_id='stress-nerf', scene_profile=DenseSceneProfile.SPATIAL_DIGITAL_TWIN,
        reconstruction_frame_id='reconstruction', camera_profile_id='camera', scale_profile_id='arbitrary',
        metric_geometry_status=MetricGeometryStatus.ARBITRARY,
        scale_status=SfmScaleStatus.ARBITRARY, world_alignment_status=SfmWorldAlignmentStatus.ARBITRARY,
        unit_to_metre=None, relative_scale_uncertainty=None, frames=tuple(frames),
    )
    fingerprint = neural_scene_dataset_fingerprint(dataset)
    for index in range(100):
        request = NerfWorkerRequest.from_dataset(
            request_id=f'request-{index}', backend_kind=NerfBackendKind.DEVELOPMENT_MOCK,
            profile=NerfOptimizationProfile.EFFICIENT, dataset=dataset,
            dataset_manifest_path=(tmp_path / 'dataset.json').resolve(),
            output_dir=(tmp_path / f'out-{index}').resolve(), maximum_input_frames=500,
        )
        assert request.dataset_fingerprint_sha256 == fingerprint
        assert request.maximum_input_frames == 500
