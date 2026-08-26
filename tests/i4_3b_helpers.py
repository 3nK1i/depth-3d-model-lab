from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path
import struct
from uuid import uuid4

import numpy as np

from core.contracts import ImageSize
from core.reconstruction.contracts import InputSourceKind, InputSourceMetadata
from core.reconstruction.dense_provider_architecture import (
    DenseProviderImplementation,
    DenseProviderSelectionDecision,
    DenseProviderSelectionMode,
    DenseProviderSelectionRequest,
)
from core.reconstruction.mvs_environment import (
    PYCOLMAP_MVS_REQUIRED_APIS,
    MvsGpuResource,
    MvsHostResourceProbe,
    PycolmapMvsEnvironmentProbe,
)
from core.reconstruction.mvs_provider import I43B_PROVIDER_ID
from core.reconstruction.sfm_contracts import (
    OfflineSfmResult,
    ReconstructionGauge,
    SfmCameraEstimate,
    SfmStageStatistics,
    SfmWorkspaceImage,
    SfmWorkspaceManifest,
)
from core.reconstruction.sfm_environment import PYCOLMAP_CUDA_PACKAGE, PYCOLMAP_RECOMMENDED_VERSION
from core.reconstruction.sfm_pycolmap import PYCOLMAP_BACKEND_ID


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_minimal_ply(path: Path, count: int = 3) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [f"{i}.0 0.0 0.0 255 255 255\n" for i in range(count)]
    path.write_text(
        "ply\nformat ascii 1.0\n"
        f"element vertex {count}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        "end_header\n" + "".join(rows),
        encoding="ascii",
    )
    with Path(f"{path}.vis").open("wb") as handle:
        handle.write(struct.pack("<Q", count))
        for _ in range(count):
            handle.write(struct.pack("<II", 1, 0))


def make_sfm_result(tmp_path: Path, *, view_count: int = 3, backend_id: str = PYCOLMAP_BACKEND_ID) -> OfflineSfmResult:
    root = tmp_path / "sfm"
    images_dir = root / "images"
    sparse_dir = root / "sparse" / "0"
    images_dir.mkdir(parents=True)
    sparse_dir.mkdir(parents=True)
    for filename in ("cameras.bin", "images.bin", "points3D.bin"):
        (sparse_dir / filename).write_bytes((filename + "-i43b").encode("ascii"))

    source = InputSourceMetadata(
        source_id="i43b-source",
        kind=InputSourceKind.IMAGE_SEQUENCE,
        display_name="I4.3B synthetic image set",
        source_uri=str(tmp_path / "source"),
        expected_image_size=ImageSize(640, 480),
    )
    workspace_images = []
    cameras = []
    for index in range(view_count):
        name = f"frame_{index:04d}.png"
        path = images_dir / name
        path.write_bytes(f"synthetic-i43b-image-{index}".encode("ascii"))
        keyframe_id = f"kf-{index}"
        frame_id = f"frame-{index}"
        workspace_images.append(
            SfmWorkspaceImage(
                order=index,
                keyframe_id=keyframe_id,
                frame_id=frame_id,
                source_sequence_index=index,
                timestamp_ns=index * 1_000_000,
                image_size=ImageSize(640, 480),
                source_locator=str(path),
                staged_image_path=f"images/{name}",
                staged_sha256=sha256_file(path),
            )
        )
        transform = np.eye(4, dtype=np.float64)
        transform[0, 3] = 0.1 * index
        cameras.append(
            SfmCameraEstimate(
                estimate_id=f"estimate-{index}",
                camera_id="camera-0",
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
    return OfflineSfmResult(
        session_id=session_id,
        backend_id=backend_id,
        backend_version="4.1.1",
        gauge=ReconstructionGauge(),
        cameras=tuple(cameras),
        landmarks=(),
        statistics=SfmStageStatistics(
            input_keyframes=view_count,
            registered_keyframes=view_count,
            landmarks=0,
            observations=0,
            elapsed_ns=1,
        ),
        workspace=workspace,
        metadata={"selected_model_id": 0},
    )


def good_environment(python_path: str = "/synthetic/eia-pycolmap-probe/bin/python3.11") -> PycolmapMvsEnvironmentProbe:
    return PycolmapMvsEnvironmentProbe(
        python_executable=python_path,
        available=True,
        pycolmap_version=PYCOLMAP_RECOMMENDED_VERSION,
        distribution=PYCOLMAP_CUDA_PACKAGE,
        distribution_version=PYCOLMAP_RECOMMENDED_VERSION,
        cuda_devices=1,
        apis=tuple((name, True) for name in PYCOLMAP_MVS_REQUIRED_APIS),
        patch_match_options_constructible=True,
        stereo_fusion_options_constructible=True,
        patch_match_option_fields=("gpu_index", "geom_consistency", "cache_size"),
        stereo_fusion_option_fields=("cache_size", "max_image_size"),
    )


def good_resources(path: Path) -> MvsHostResourceProbe:
    return MvsHostResourceProbe(
        workspace_path=str(path),
        disk_free_gib=100.0,
        ram_available_gib=32.0,
        gpus=(MvsGpuResource(index=0, name="Synthetic GPU", total_vram_gib=24.0, free_vram_gib=20.0),),
    )


def explicit_selection(view_count: int, *, frame_id: str = "sfm-frame-1", provider_id: str = I43B_PROVIDER_ID):
    request = DenseProviderSelectionRequest(
        mode=DenseProviderSelectionMode.EXPLICIT_PROVIDER,
        available_view_count=view_count,
        authoritative_sfm_available=True,
        authoritative_sfm_frame_id=frame_id,
        requested_provider_id=provider_id,
        allow_fallback=False,
    )
    decision = DenseProviderSelectionDecision(
        request_mode=DenseProviderSelectionMode.EXPLICIT_PROVIDER,
        requested_provider_id=provider_id,
        selected_provider_id=provider_id,
        selected_implementation=DenseProviderImplementation.COLMAP_MVS,
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


class FakeWorkerRunner:
    def __init__(self, *, return_code: int = 0, status: str = "pass", mutate_path: Path | None = None, ply_points: int = 3, vis_points: int | None = None):
        self.return_code = return_code
        self.status = status
        self.mutate_path = mutate_path
        self.ply_points = ply_points
        self.vis_points = vis_points
        self.calls = 0

    def run(self, *, command, result_path, state_path, dense_workspace, total_views, timeout_s, poll_s, progress_callback):
        self.calls += 1
        dense_workspace.mkdir(parents=True, exist_ok=True)
        progress_callback("undistort", 0, {"stage": "undistort"})
        progress_callback("patch_match", max(1, total_views // 2), {"stage": "patch_match", "resources": {"process_peak_rss_gib": 0.25}})
        progress_callback("stereo_fusion", total_views, {"stage": "stereo_fusion"})
        if self.return_code == 0 and self.status == "pass":
            fused = dense_workspace / "fused.ply"
            write_minimal_ply(fused, self.ply_points)
            if self.vis_points is not None:
                with Path(f"{fused}.vis").open("r+b") as handle:
                    handle.seek(0)
                    handle.write(struct.pack("<Q", self.vis_points))
        if self.mutate_path is not None:
            self.mutate_path.write_bytes(self.mutate_path.read_bytes() + b"mutation")
        result_path.parent.mkdir(parents=True, exist_ok=True)
        result_path.write_text(json.dumps({"status": self.status, "error": "synthetic worker failure" if self.status != "pass" else ""}), encoding="utf-8")
        state_path.write_text(json.dumps({"stage": "complete" if self.status == "pass" else "failed"}), encoding="utf-8")
        return self.return_code
