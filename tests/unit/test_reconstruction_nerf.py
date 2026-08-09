from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import sys

import numpy as np
import pytest

from core.reconstruction import (
    DenseSceneProfile,
    MetricGeometryStatus,
    NerfBackendKind,
    NerfOptimizationProfile,
    NerfSceneArtifact,
    NerfSceneRole,
    NerfTrainingState,
    NerfUsageScope,
    NerfWorkerExecutionError,
    NerfWorkerProcessSpec,
    NerfWorkerRequest,
    NeuralSceneDataset,
    NeuralSceneFrame,
    SfmScaleStatus,
    SfmWorldAlignmentStatus,
    nerf_backend_provenance,
    nerf_backend_runtime_pin,
    neural_scene_dataset_fingerprint,
)


def _dataset(tmp_path: Path, *, metric: bool = False) -> NeuralSceneDataset:
    image = (tmp_path / 'frame.ppm').resolve()
    image.write_bytes(b'P6\n2 2\n255\n' + bytes([100, 90, 80]) * 4)
    frames = []
    for index in range(3):
        pose = np.eye(4, dtype=np.float64)
        pose[0, 3] = index * 0.05
        frames.append(NeuralSceneFrame(
            frame_id=f'frame-{index}', image_path=str(image), width=2, height=2,
            fx=2.0, fy=2.0, cx=1.0, cy=1.0,
            T_reconstruction_from_camera=pose, pose_layer='optimized_keyframe',
            camera_profile_id='camera', timestamp_ns=index,
        ))
    return NeuralSceneDataset(
        session_id='unit-nerf',
        scene_profile=DenseSceneProfile.OBJECT_INSPECTION,
        reconstruction_frame_id='reconstruction',
        camera_profile_id='camera',
        scale_profile_id='metric' if metric else 'arbitrary',
        metric_geometry_status=MetricGeometryStatus.PROVISIONAL_METRIC if metric else MetricGeometryStatus.ARBITRARY,
        scale_status=SfmScaleStatus.ESTIMATED_METRIC if metric else SfmScaleStatus.ARBITRARY,
        world_alignment_status=SfmWorldAlignmentStatus.ARBITRARY,
        unit_to_metre=0.01 if metric else None,
        relative_scale_uncertainty=0.05 if metric else None,
        frames=tuple(frames),
    )


def _payload(tmp_path: Path, *, trained: bool = False, metric: bool = False) -> dict:
    backend = NerfBackendKind.DEVELOPMENT_MOCK
    return {
        'schema_version': 1,
        'request_id': 'request',
        'backend_id': 'mock',
        'backend_version': '1',
        'backend_revision': 'internal',
        'backend_provenance': nerf_backend_provenance(backend).as_dict(),
        'method_name': 'nerfacto',
        'training_state': 'trained' if trained else 'contract_only',
        'scene_role': 'appearance_only',
        'artifact_dir': str((tmp_path / 'artifact').resolve()),
        'config_path': str((tmp_path / 'artifact/config.yml').resolve()),
        'checkpoint_path': str((tmp_path / 'artifact/model.ckpt').resolve()),
        'dataset_fingerprint_sha256': 'a' * 64,
        'input_frame_count': 3,
        'training_steps': 100 if trained else 0,
        'training_duration_s': 1.0 if trained else 0.0,
        'render_capable': trained,
        'device': 'cuda:0' if trained else 'mock',
        'pose_convention': 'T_reconstruction_from_camera',
        'reconstruction_frame_id': 'reconstruction',
        'metric_geometry_status': 'provisional_metric' if metric else 'arbitrary',
        'unit_to_metre': 0.01 if metric else None,
        'relative_scale_uncertainty': 0.05 if metric else None,
        'canonical_geometry_authoritative': False,
        'certified_metrology': False,
    }


def test_nerfstudio_provenance_is_permissive_and_nerfacto() -> None:
    provenance = nerf_backend_provenance(NerfBackendKind.NERFSTUDIO_NERFACTO_EXTERNAL)
    assert provenance.license_name == 'Apache-2.0'
    assert provenance.usage_scope is NerfUsageScope.PERMISSIVE_GENERAL_USE
    assert provenance.commercial_use_permitted_by_backend_license is True
    assert provenance.method_name == 'nerfacto'
    assert provenance.runtime_dependency_name == 'tiny-cuda-nn'
    assert provenance.runtime_dependency_license == 'BSD-3-Clause'


def test_nerfstudio_runtime_pin_is_reproducible() -> None:
    pin = nerf_backend_runtime_pin()
    assert pin.release_tag == 'v1.1.5'
    assert pin.revision_prefix == '6b60855'
    assert pin.method_name == 'nerfacto'
    assert pin.documented_memory_mib == 6144


def test_worker_request_uses_neural_scene_fingerprint(tmp_path: Path) -> None:
    dataset = _dataset(tmp_path)
    request = NerfWorkerRequest.from_dataset(
        request_id='request', backend_kind=NerfBackendKind.DEVELOPMENT_MOCK,
        profile=NerfOptimizationProfile.EFFICIENT, dataset=dataset,
        dataset_manifest_path=(tmp_path / 'dataset.json').resolve(),
        output_dir=(tmp_path / 'out').resolve(), maximum_input_frames=8,
    )
    assert request.dataset_fingerprint_sha256 == neural_scene_dataset_fingerprint(dataset)
    assert request.method_name == 'nerfacto'


def test_worker_request_rejects_non_nerfacto_method(tmp_path: Path) -> None:
    dataset = _dataset(tmp_path)
    request = NerfWorkerRequest.from_dataset(
        request_id='request', backend_kind=NerfBackendKind.DEVELOPMENT_MOCK,
        profile=NerfOptimizationProfile.EFFICIENT, dataset=dataset,
        dataset_manifest_path=(tmp_path / 'dataset.json').resolve(),
        output_dir=(tmp_path / 'out').resolve(), maximum_input_frames=8,
    )
    with pytest.raises(Exception):
        replace(request, method_name='instant-ngp')


def test_contract_only_artifact_is_not_render_capable(tmp_path: Path) -> None:
    artifact = NerfSceneArtifact.from_payload(_payload(tmp_path))
    assert artifact.training_state is NerfTrainingState.CONTRACT_ONLY
    assert artifact.render_capable is False
    assert artifact.scene_role is NerfSceneRole.APPEARANCE_ONLY


def test_contract_only_artifact_rejects_training_claim(tmp_path: Path) -> None:
    payload = _payload(tmp_path)
    payload['training_steps'] = 1
    with pytest.raises(NerfWorkerExecutionError):
        NerfSceneArtifact.from_payload(payload)


def test_trained_artifact_requires_render_capability(tmp_path: Path) -> None:
    payload = _payload(tmp_path, trained=True)
    payload['render_capable'] = False
    with pytest.raises(NerfWorkerExecutionError):
        NerfSceneArtifact.from_payload(payload)


def test_nerf_artifact_rejects_canonical_geometry_claim(tmp_path: Path) -> None:
    payload = _payload(tmp_path)
    payload['canonical_geometry_authoritative'] = True
    with pytest.raises(NerfWorkerExecutionError):
        NerfSceneArtifact.from_payload(payload)


def test_nerf_artifact_rejects_certified_metrology_claim(tmp_path: Path) -> None:
    payload = _payload(tmp_path)
    payload['certified_metrology'] = True
    with pytest.raises(NerfWorkerExecutionError):
        NerfSceneArtifact.from_payload(payload)


def test_metric_nerf_artifact_preserves_scale_context(tmp_path: Path) -> None:
    artifact = NerfSceneArtifact.from_payload(_payload(tmp_path, metric=True))
    assert artifact.metric_geometry_status is MetricGeometryStatus.PROVISIONAL_METRIC
    assert artifact.unit_to_metre == pytest.approx(0.01)
    assert artifact.relative_scale_uncertainty == pytest.approx(0.05)
    assert artifact.canonical_geometry_authoritative is False


def test_worker_spec_requires_isolation_by_default() -> None:
    spec = NerfWorkerProcessSpec(
        python_executable=str(Path(sys.executable).resolve()),
        worker_script=str((Path(__file__).resolve().parents[2] / 'scripts/run_nerf_worker_mock.py').resolve()),
    )
    assert spec.require_isolated_interpreter is True
    assert spec.interpreter_is_isolated is False


def test_external_worker_refuses_production_interpreter_by_default(tmp_path: Path) -> None:
    from core.reconstruction import run_nerf_external_worker
    dataset = _dataset(tmp_path)
    manifest = (tmp_path / 'dataset.json').resolve()
    from core.reconstruction import write_neural_scene_dataset_manifest
    write_neural_scene_dataset_manifest(dataset, manifest)
    request = NerfWorkerRequest.from_dataset(
        request_id='isolated', backend_kind=NerfBackendKind.DEVELOPMENT_MOCK,
        profile=NerfOptimizationProfile.EFFICIENT, dataset=dataset,
        dataset_manifest_path=manifest, output_dir=(tmp_path / 'out').resolve(), maximum_input_frames=8,
    )
    spec = NerfWorkerProcessSpec(
        python_executable=str(Path(sys.executable).resolve()),
        worker_script=str((Path(__file__).resolve().parents[2] / 'scripts/run_nerf_worker_mock.py').resolve()),
    )
    with pytest.raises(NerfWorkerExecutionError, match='isolated interpreter'):
        run_nerf_external_worker(spec, request)


def test_missing_nerf_python_probe_fails_closed(tmp_path: Path) -> None:
    from core.reconstruction import probe_nerf_environment
    probe = probe_nerf_environment((tmp_path / 'missing-python').resolve())
    assert probe.available is False
    assert probe.backend_ready is False
    assert probe.error == 'python_executable_missing'


def test_available_probe_requires_versions() -> None:
    from core.reconstruction import NerfEnvironmentProbe
    with pytest.raises(Exception):
        NerfEnvironmentProbe(
            python_executable=str(Path(sys.executable).resolve()),
            available=True,
            nerfstudio_version='',
            torch_version='2.1.2',
        )
