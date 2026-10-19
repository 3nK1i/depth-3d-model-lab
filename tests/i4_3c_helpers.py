from __future__ import annotations

import json
from pathlib import Path
from uuid import uuid4

import cv2
import numpy as np

from core.contracts import ImageSize
from core.reconstruction.contracts import InputSourceKind, InputSourceMetadata
from core.reconstruction.da3 import (
    DA3_EFFICIENT_MODEL_ID,
    DA3EnvironmentProbe,
    DA3ResultManifest,
)
from core.reconstruction.da3_dense_provider import DA3ProviderRunContext
from core.reconstruction.da3_provider import DA3ProviderProfile, read_da3_input_manifest
from core.reconstruction.dense_provider_architecture import (
    DenseProviderImplementation,
    DenseProviderSelectionDecision,
    DenseProviderSelectionMode,
    DenseProviderSelectionRequest,
)
from core.reconstruction.sfm_contracts import (
    OfflineSfmResult,
    ReconstructionGauge,
    SfmCameraEstimate,
    SfmCameraModelEstimate,
    SfmLandmarkEstimate,
    SfmObservation,
    SfmStageStatistics,
    SfmWorkspaceImage,
    SfmWorkspaceManifest,
)
from core.reconstruction.sfm_pycolmap import PYCOLMAP_BACKEND_ID


def _sha256(path: Path) -> str:
    import hashlib
    return hashlib.sha256(path.read_bytes()).hexdigest()


def make_da3_sfm_result(tmp_path: Path, *, view_count: int = 3, camera_ids: tuple[str, ...] | None = None) -> OfflineSfmResult:
    if view_count < 2:
        raise ValueError("view_count must be >= 2")
    root = tmp_path / "sfm"
    images_dir = root / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    size = ImageSize(64, 48)
    source = InputSourceMetadata(
        source_id="i43c-source",
        kind=InputSourceKind.IMAGE_SEQUENCE,
        display_name="I4.3C synthetic image set",
        source_uri=str(tmp_path / "source"),
        expected_image_size=size,
    )
    camera_ids = camera_ids or tuple("camera-0" for _ in range(view_count))
    if len(camera_ids) != view_count:
        raise ValueError("camera_ids must match view_count")

    workspace_images = []
    cameras = []
    for index in range(view_count):
        image = np.zeros((48, 64, 3), dtype=np.uint8)
        image[..., 0] = 20 + index
        image[..., 1] = np.arange(64, dtype=np.uint8)[None, :]
        image[..., 2] = np.arange(48, dtype=np.uint8)[:, None]
        path = images_dir / f"frame_{index:04d}.png"
        assert cv2.imwrite(str(path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        keyframe_id = f"kf-{index}"
        frame_id = f"frame-{index}"
        workspace_images.append(
            SfmWorkspaceImage(
                order=index,
                keyframe_id=keyframe_id,
                frame_id=frame_id,
                source_sequence_index=index,
                timestamp_ns=(index + 1) * 1_000_000,
                image_size=size,
                source_locator=str(path),
                staged_image_path=f"images/{path.name}",
                staged_sha256=_sha256(path),
            )
        )
        transform = np.eye(4, dtype=np.float64)
        transform[0, 3] = 0.05 * index
        cameras.append(
            SfmCameraEstimate(
                estimate_id=f"estimate-{index}",
                camera_id=camera_ids[index],
                keyframe_id=keyframe_id,
                frame_id=frame_id,
                backend_image_id=index + 1,
                T_reconstruction_from_camera=transform,
            )
        )

    session_id = str(uuid4())
    workspace = SfmWorkspaceManifest(
        session_id=session_id,
        source=source,
        workspace_root=str(root),
        images_directory="images",
        database_path="database.db",
        sparse_directory="sparse",
        checkpoints_directory="checkpoints",
        reports_directory="reports",
        logs_directory="logs",
        images=tuple(workspace_images),
    )

    unique_camera_ids = sorted(set(camera_ids))
    models = tuple(
        SfmCameraModelEstimate(
            camera_id=camera_id,
            backend_camera_id=index,
            backend_model_name="PINHOLE",
            image_size=size,
            parameters=(100.0, 100.0, 32.0, 24.0),
        )
        for index, camera_id in enumerate(unique_camera_ids)
    )

    landmarks = []
    xs = np.linspace(-0.24, 0.24, 5)
    ys = np.linspace(-0.16, 0.16, 4)
    landmark_index = 0
    for y in ys:
        for x in xs:
            landmark_id = f"lm-{landmark_index}"
            position = np.asarray([x, y, 2.0], dtype=np.float64)
            observations = []
            for camera_index in range(view_count):
                cam_x = 0.05 * camera_index
                px = 100.0 * ((x - cam_x) / 2.0) + 32.0
                py = 100.0 * (y / 2.0) + 24.0
                observations.append(
                    SfmObservation(
                        observation_id=f"obs-{landmark_index}-{camera_index}",
                        landmark_id=landmark_id,
                        keyframe_id=f"kf-{camera_index}",
                        feature_xy=(float(px), float(py)),
                        feature_index=landmark_index,
                    )
                )
            landmarks.append(
                SfmLandmarkEstimate(
                    landmark_id=landmark_id,
                    track_id=f"track-{landmark_index}",
                    position_reconstruction=position,
                    observations=tuple(observations),
                    color_rgb=(200, 100, 50),
                )
            )
            landmark_index += 1

    observations = sum(len(item.observations) for item in landmarks)
    return OfflineSfmResult(
        session_id=session_id,
        backend_id=PYCOLMAP_BACKEND_ID,
        backend_version="4.1.1",
        gauge=ReconstructionGauge(),
        cameras=tuple(cameras),
        landmarks=tuple(landmarks),
        statistics=SfmStageStatistics(
            input_keyframes=view_count,
            registered_keyframes=view_count,
            landmarks=len(landmarks),
            observations=observations,
            elapsed_ns=1,
            mean_reprojection_error_px=0.1,
        ),
        workspace=workspace,
        camera_models=models,
        metadata={"selected_model_id": 0},
    )


def good_da3_environment(tmp_path: Path, *, available: bool = True) -> DA3EnvironmentProbe:
    python = tmp_path / "depth3d-da3" / "bin" / "python3.11"
    repo = tmp_path / "depth-anything-3"
    python.parent.mkdir(parents=True, exist_ok=True)
    repo.mkdir(parents=True, exist_ok=True)
    python.write_text("", encoding="utf-8")
    return DA3EnvironmentProbe(
        python_executable=str(python.resolve()),
        repository_root=str(repo.resolve()),
        source_revision="a" * 40,
        python_version="3.11",
        torch_version="2.5.1",
        torchvision_version="0.20.1",
        xformers_version="0.0.28",
        torch_cuda_version="12.4",
        cuda_available=available,
        device_names=("Synthetic RTX",) if available else (),
        device_capabilities=((8, 9),) if available else (),
        depth_anything_3_importable=available,
        turing_compatibility_patch_applied=available,
        runtime_lock_valid=available,
        error="" if available else "not_ready",
    )


def make_context(tmp_path: Path, *, view_count: int = 3, camera_ids: tuple[str, ...] | None = None) -> tuple[DA3ProviderRunContext, DA3EnvironmentProbe]:
    sfm = make_da3_sfm_result(tmp_path, view_count=view_count, camera_ids=camera_ids)
    env = good_da3_environment(tmp_path)
    worker = tmp_path / "da3_worker.py"
    worker.write_text("", encoding="utf-8")
    context = DA3ProviderRunContext(
        sfm_result=sfm,
        reconstruction_frame_id="offline-sfm-reconstruction",
        output_dir=tmp_path / "provider-output",
        da3_python=Path(env.python_executable),
        da3_repository=Path(env.repository_root),
        worker_script=worker,
        profile=DA3ProviderProfile(model_id=DA3_EFFICIENT_MODEL_ID, process_res=392),
    )
    return context, env


def explicit_da3_selection(view_count: int, *, provider_id: str = "depth-anything-3-external"):
    request = DenseProviderSelectionRequest(
        mode=DenseProviderSelectionMode.EXPLICIT_PROVIDER,
        available_view_count=view_count,
        authoritative_sfm_available=True,
        authoritative_sfm_frame_id="offline-sfm-reconstruction",
        requested_provider_id=provider_id,
        allow_fallback=False,
    )
    decision = DenseProviderSelectionDecision(
        request_mode=DenseProviderSelectionMode.EXPLICIT_PROVIDER,
        requested_provider_id=provider_id,
        selected_provider_id=provider_id,
        selected_implementation=DenseProviderImplementation.DA3,
        considered_provider_ids=(provider_id,),
        automatic=False,
        engineer_override_applied=False,
        fallback_used=False,
        reason_codes=("explicit_engineer_selection",),
    )
    return request, decision


class CollectingProgressSink:
    def __init__(self) -> None:
        self.events = []

    def publish(self, event) -> None:
        self.events.append(event)


class FakeDA3Runner:
    def __init__(self, *, depth_value: float = 1.0, output_shape: tuple[int, int] = (48, 64), mutate_path: Path | None = None) -> None:
        self.depth_value = float(depth_value)
        self.output_shape = tuple(output_shape)
        self.mutate_path = mutate_path
        self.calls = 0

    def run(self, input_manifest_path, output_dir):
        self.calls += 1
        source = read_da3_input_manifest(input_manifest_path)
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        n = len(source.image_paths)
        h, w = self.output_shape
        depth = np.full((n, h, w), self.depth_value, dtype=np.float32)
        confidence = np.ones_like(depth, dtype=np.float32)
        extrinsics = np.tile(np.eye(4, dtype=np.float32)[None, :3, :], (n, 1, 1))
        intrinsics = np.tile(np.asarray([[100.0, 0.0, w / 2.0], [0.0, 100.0, h / 2.0], [0.0, 0.0, 1.0]], dtype=np.float32)[None], (n, 1, 1))
        npz = out / "da3_output.npz"
        np.savez_compressed(npz, depth=depth, confidence=confidence, extrinsics_w2c=extrinsics, intrinsics=intrinsics)
        manifest = DA3ResultManifest(
            request_id=source.request_id,
            model_id=DA3_EFFICIENT_MODEL_ID,
            model_revision="b" * 40,
            source_revision="a" * 40,
            output_npz_path=str(npz.resolve()),
            frame_count=n,
            width=w,
            height=h,
            process_res=392,
            inference_duration_s=0.2,
            peak_vram_mib=300.0,
            gpu_name="Synthetic RTX",
        )
        manifest_path = out / "result_manifest.json"
        manifest_path.write_text(json.dumps(manifest.as_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
        if self.mutate_path is not None:
            self.mutate_path.write_bytes(self.mutate_path.read_bytes() + b"mutation")
        return manifest_path, manifest
