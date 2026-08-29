from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import numpy as np

from core.reconstruction.dense_provider_architecture import (
    DenseProviderImplementation,
    DenseProviderSelectionDecision,
    DenseProviderSelectionMode,
    DenseProviderSelectionRequest,
)
from core.reconstruction.rolling_depth_dense_provider import (
    I43D2_PROVIDER_ID,
    PreparedRollingDepthVideo,
    RollingDepthEnvironmentProbe,
    RollingDepthProviderRunContext,
    RollingDepthWorkerResult,
)
from core.reconstruction.sfm_contracts import SfmLandmarkEstimate, SfmObservation
from tests.i4_3c_helpers import make_da3_sfm_result


def make_rolling_sfm_result(tmp_path: Path, *, view_count: int = 3):
    base = make_da3_sfm_result(tmp_path, view_count=view_count)
    camera_by_keyframe = {item.keyframe_id: item for item in base.cameras}
    model_by_id = {item.camera_id: item for item in base.camera_models}
    landmarks = []
    for index, landmark in enumerate(base.landmarks):
        old = np.asarray(landmark.position_reconstruction, dtype=np.float64)
        # Deliberate depth variation makes a global affine observable in tests.
        z = 1.55 + 0.045 * index
        point = np.asarray([old[0], old[1], z], dtype=np.float64)
        observations = []
        for observation in landmark.observations:
            camera = camera_by_keyframe[observation.keyframe_id]
            model = model_by_id[camera.camera_id]
            fx, fy, cx, cy = model.parameters
            T_c_r = np.linalg.inv(np.asarray(camera.T_reconstruction_from_camera, dtype=np.float64))
            p_c = T_c_r @ np.append(point, 1.0)
            x = float(fx * (p_c[0] / p_c[2]) + cx)
            y = float(fy * (p_c[1] / p_c[2]) + cy)
            observations.append(replace(observation, feature_xy=(x, y)))
        landmarks.append(
            SfmLandmarkEstimate(
                landmark_id=landmark.landmark_id,
                track_id=landmark.track_id,
                position_reconstruction=point,
                observations=tuple(observations),
                color_rgb=landmark.color_rgb,
            )
        )
    return replace(base, landmarks=tuple(landmarks))


def good_rolling_environment(tmp_path: Path, *, available: bool = True) -> RollingDepthEnvironmentProbe:
    repo = (tmp_path / "RollingDepth").resolve()
    python = (tmp_path / "eia-rollingdepth/bin/python").resolve()
    checkpoint = (repo / "checkpoint/rollingdepth-v1-0").resolve()
    lock = (repo / ".eia_rolling_depth_runtime_lock.json").resolve()
    repo.mkdir(parents=True, exist_ok=True)
    python.parent.mkdir(parents=True, exist_ok=True)
    checkpoint.mkdir(parents=True, exist_ok=True)
    python.write_text("", encoding="utf-8")
    (checkpoint / "model_index.json").write_text("{}\n", encoding="utf-8")
    lock.write_text(json.dumps({"source_revision": "d" * 40}) + "\n", encoding="utf-8")
    return RollingDepthEnvironmentProbe(
        python_executable=str(python),
        repository_root=str(repo),
        runtime_lock_path=str(lock),
        checkpoint_path=str(checkpoint),
        source_revision="d" * 40,
        python_version="3.12.7",
        torch_version="2.5.1",
        diffusers_version="0.30.0.dev0",
        torch_cuda_version="12.4",
        source_ready=available,
        checkpoint_ready=available,
        runtime_lock_valid=available,
        upstream_importable=available,
        modified_diffusers_ready=available,
        torch_available=available,
        ffmpeg_available=available,
        cuda_available=available,
        device_names=("Synthetic RTX",) if available else (),
        error="" if available else "not_ready",
    )


def make_rolling_context(tmp_path: Path, *, view_count: int = 3):
    sfm = make_rolling_sfm_result(tmp_path, view_count=view_count)
    env = good_rolling_environment(tmp_path)
    worker = tmp_path / "rolling_worker.py"
    worker.write_text("", encoding="utf-8")
    context = RollingDepthProviderRunContext(
        sfm_result=sfm,
        reconstruction_frame_id="offline-sfm-reconstruction",
        output_dir=tmp_path / "rolling-provider-output",
        rolling_depth_python=Path(env.python_executable),
        rolling_depth_repository=Path(env.repository_root),
        worker_script=worker,
        device="cuda:0",
        precision="fp16",
    )
    return context, env


def explicit_rolling_selection(view_count: int, *, provider_id: str = I43D2_PROVIDER_ID):
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
        selected_implementation=DenseProviderImplementation.ROLLING_DEPTH,
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


class FakeRollingDepthSequenceBuilder:
    def __init__(self, *, mutate_path: Path | None = None, output_hw: tuple[int, int] | None = None) -> None:
        self.calls = 0
        self.last_source_indices = []
        self.mutate_path = mutate_path
        self.output_hw = output_hw

    def build(self, *, images, paths, output_dir: Path) -> PreparedRollingDepthVideo:
        self.calls += 1
        self.last_source_indices = [item.source_sequence_index for item in images]
        output_dir.mkdir(parents=True, exist_ok=False)
        video = output_dir / "sequence.mkv"
        video.write_bytes(b"synthetic-lossless-video")
        if self.mutate_path is not None:
            self.mutate_path.write_bytes(self.mutate_path.read_bytes() + b"mutation")
        h, w = self.output_hw or (images[0].image_size.height, images[0].image_size.width)
        import hashlib
        sha = hashlib.sha256(video.read_bytes()).hexdigest()
        return PreparedRollingDepthVideo(video, len(images), w, h, 30.0, sha)


class FakeRollingDepthWorkerRunner:
    def __init__(
        self,
        sfm_result,
        *,
        source_revision: str = "d" * 40,
        scale: float = 1.6,
        shift: float = 0.35,
        output_shape: tuple[int, int] | None = None,
        mutate_path: Path | None = None,
        depth_semantics: str = "temporally_coaligned_relative_depth",
        per_frame_alignment: bool = False,
    ) -> None:
        self.sfm_result = sfm_result
        self.source_revision = source_revision
        self.scale = float(scale)
        self.shift = float(shift)
        self.output_shape = output_shape
        self.mutate_path = mutate_path
        self.depth_semantics = depth_semantics
        self.per_frame_alignment = per_frame_alignment
        self.calls = 0
        self.last_manifest = None

    def run(self, manifest_path: Path, output_dir: Path) -> RollingDepthWorkerResult:
        self.calls += 1
        payload = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
        self.last_manifest = payload
        sequence = payload["sequence"]
        n = len(sequence)
        h = int(payload["expected_height"])
        w = int(payload["expected_width"])
        if self.output_shape is not None:
            h, w = self.output_shape
        # Fake the provider-native RollingDepth polarity.  The production
        # adapter canonicalizes this sign once before the generic SfM affine
        # fit, so ``scale``/``shift`` below continue to describe the expected
        # canonical sequence-to-SfM mapping.
        depth = np.full((n, h, w), -(2.0 - self.shift) / self.scale, dtype=np.float32)
        if self.output_shape is None:
            image_index = {item["keyframe_id"]: i for i, item in enumerate(sequence)}
            camera_by_keyframe = {item.keyframe_id: item for item in self.sfm_result.cameras}
            for landmark in self.sfm_result.landmarks:
                p_r = np.append(np.asarray(landmark.position_reconstruction, dtype=np.float64), 1.0)
                for obs in landmark.observations:
                    if obs.keyframe_id not in image_index:
                        continue
                    camera = camera_by_keyframe[obs.keyframe_id]
                    z = float((np.linalg.inv(camera.T_reconstruction_from_camera) @ p_r)[2])
                    source = -((z - self.shift) / self.scale)
                    x, y = obs.feature_xy
                    x0 = max(0, min(w - 1, int(np.floor(x))))
                    y0 = max(0, min(h - 1, int(np.floor(y))))
                    x1 = min(w - 1, x0 + 1)
                    y1 = min(h - 1, y0 + 1)
                    depth[image_index[obs.keyframe_id], y0:y1 + 1, x0:x1 + 1] = source
        output_dir.mkdir(parents=True, exist_ok=True)
        npz = output_dir / "rolling_depth_evidence.npz"
        np.savez_compressed(npz, depth=depth)
        if self.mutate_path is not None:
            self.mutate_path.write_bytes(self.mutate_path.read_bytes() + b"mutation")
        result = {
            "schema_version": 1,
            "phase": "I4.3D2",
            "status": "pass",
            "source_revision": self.source_revision,
            "frame_count": n,
            "depth_npz_path": str(npz.resolve()),
            "depth_semantics": self.depth_semantics,
            "native_temporal_coalignment": True,
            "per_frame_alignment": self.per_frame_alignment,
            "native_metric": False,
            "native_camera_pose_authority": False,
        }
        result_path = output_dir / "rolling_depth_result_manifest.json"
        result_path.write_text(json.dumps(result), encoding="utf-8")
        return RollingDepthWorkerResult(result_path, npz, depth, result)


def scrambled_temporal_metadata_sfm_result(sfm_result):
    images = list(sfm_result.workspace.images)
    n = len(images)
    permutation = list(range(n))
    if n >= 3:
        permutation = [2, 0, 1] + list(range(3, n))
    remapped = []
    for workspace_order, item in enumerate(images):
        source_index = permutation[workspace_order]
        remapped.append(
            replace(
                item,
                order=workspace_order,
                source_sequence_index=source_index,
                timestamp_ns=(source_index + 1) * 1_000_000,
            )
        )
    return replace(sfm_result, workspace=replace(sfm_result.workspace, images=tuple(remapped)))
