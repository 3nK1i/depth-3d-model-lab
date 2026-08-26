from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np

from core.reconstruction.dense_provider_architecture import (
    DenseProviderImplementation,
    DenseProviderSelectionDecision,
    DenseProviderSelectionMode,
    DenseProviderSelectionRequest,
)
from core.reconstruction.video_depth_anything_dense_provider import (
    I43D1_PROVIDER_ID,
    VideoDepthAnythingEnvironmentProbe,
    VideoDepthAnythingProviderRunContext,
    VideoDepthAnythingWorkerResult,
)
from tests.i4_3c_helpers import make_da3_sfm_result


def good_vda_environment(tmp_path: Path, *, available: bool = True) -> VideoDepthAnythingEnvironmentProbe:
    root = tmp_path / "project"
    source = root / "third_party/Video-Depth-Anything"
    marker = source / ".eia_video_depth_anything_ready.json"
    checkpoint = root / "checkpoints/video-depth-anything/metric_video_depth_anything_vits.pth"
    python = tmp_path / "bin/python"
    source.mkdir(parents=True, exist_ok=True)
    marker.write_text('{"commit":"' + ("b" * 40) + '"}\n', encoding="utf-8")
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    checkpoint.write_bytes(b"checkpoint")
    python.parent.mkdir(parents=True, exist_ok=True)
    python.write_text("", encoding="utf-8")
    return VideoDepthAnythingEnvironmentProbe(
        python_executable=str(python.resolve()),
        project_root=str(root.resolve()),
        source_directory=str(source.resolve()),
        ready_marker=str(marker.resolve()),
        checkpoint_path=str(checkpoint.resolve()),
        source_revision="b" * 40,
        source_ready=available,
        checkpoint_ready=available,
        adapter_importable=available,
        upstream_importable=available,
        torch_available=available,
        cuda_available=available,
        device_names=("Synthetic RTX",) if available else (),
        error="" if available else "not_ready",
    )


def make_vda_context(tmp_path: Path, *, view_count: int = 3) -> tuple[VideoDepthAnythingProviderRunContext, VideoDepthAnythingEnvironmentProbe]:
    sfm = make_da3_sfm_result(tmp_path, view_count=view_count)
    env = good_vda_environment(tmp_path)
    worker = tmp_path / "worker.py"
    worker.write_text("", encoding="utf-8")
    context = VideoDepthAnythingProviderRunContext(
        sfm_result=sfm,
        reconstruction_frame_id="offline-sfm-reconstruction",
        output_dir=tmp_path / "vda-provider-output",
        python_executable=Path(env.python_executable),
        worker_script=worker,
        runtime_config={
            "source_dir": "third_party/Video-Depth-Anything",
            "checkpoint": "checkpoints/video-depth-anything/metric_video_depth_anything_vits.pth",
            "encoder": "vits",
            "model_label": "Metric Video Depth Anything Small",
            "input_size": 392,
            "max_resolution": 960,
        },
        device="cuda:0",
        precision="fp16",
    )
    return context, env


def explicit_vda_selection(view_count: int, *, provider_id: str = I43D1_PROVIDER_ID):
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
        selected_implementation=DenseProviderImplementation.VIDEO_DEPTH_ANYTHING,
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


class FakeVDAWorkerRunner:
    def __init__(
        self,
        *,
        source_revision: str = "b" * 40,
        depth_value: float = 1.0,
        output_shape: tuple[int, int] = (48, 64),
        mutate_path: Path | None = None,
        source_sequence_echo: bool = True,
    ) -> None:
        self.source_revision = source_revision
        self.depth_value = float(depth_value)
        self.output_shape = output_shape
        self.mutate_path = mutate_path
        self.source_sequence_echo = source_sequence_echo
        self.calls = 0
        self.last_manifest = None

    def run(self, manifest_path: Path, output_dir: Path) -> VideoDepthAnythingWorkerResult:
        import json
        self.calls += 1
        payload = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
        self.last_manifest = payload
        sequence = payload["sequence"]
        output_dir.mkdir(parents=True, exist_ok=True)
        n = len(sequence)
        h, w = self.output_shape
        depth = np.full((n, h, w), self.depth_value, dtype=np.float32)
        npz = output_dir / "vda_depth_evidence.npz"
        np.savez_compressed(npz, depth=depth)
        if self.mutate_path is not None:
            self.mutate_path.write_bytes(self.mutate_path.read_bytes() + b"mutation")
        result = {
            "schema_version": 1,
            "phase": "I4.3D1",
            "status": "pass",
            "model_id": "video-depth-anything",
            "source_revision": self.source_revision,
            "frame_count": n,
            "depth_npz_path": str(npz.resolve()),
            "frame_records": [
                {
                    "order": i,
                    "source_sequence_index": int(item["source_sequence_index"]) if self.source_sequence_echo else i,
                    "native_is_metric": True,
                    "intrinsics_source_native": "estimated",
                }
                for i, item in enumerate(sequence)
            ],
        }
        manifest = output_dir / "vda_result_manifest.json"
        manifest.write_text(json.dumps(result), encoding="utf-8")
        return VideoDepthAnythingWorkerResult(
            manifest_path=manifest,
            depth_npz_path=npz,
            depth=depth,
            metadata=result,
        )


def scrambled_temporal_metadata_sfm_result(sfm_result):
    images = list(sfm_result.workspace.images)
    n = len(images)
    # Keep workspace order contiguous, but deliberately make tuple/workspace order
    # differ from temporal source order. Timestamps follow source sequence order.
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
