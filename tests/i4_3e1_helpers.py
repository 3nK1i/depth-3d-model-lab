from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from core.reconstruction.dense_provider_architecture import (
    DenseProviderImplementation,
    DenseProviderSelectionDecision,
    DenseProviderSelectionMode,
    DenseProviderSelectionRequest,
    dense_provider_implementation_contracts,
)
from core.reconstruction.metric3d_v2_dense_provider import (
    I43E1_PROVIDER_ID,
    Metric3DV2EnvironmentProbe,
    Metric3DV2ProviderRunContext,
    Metric3DV2WorkerResult,
)
from tests.i4_3c_helpers import make_da3_sfm_result


def architecture_snapshot() -> dict:
    return {
        "schema_version": 2,
        "phase": "I4.3A",
        "status": "pass",
        "pass": True,
        "architecture": {
            "authoritative_camera_frame_owner": "pycolmap_sfm",
            "provider_implementation_contracts": [item.as_dict() for item in dense_provider_implementation_contracts()],
        },
    }


def d3_snapshot() -> dict:
    return {
        "schema_version": 1,
        "phase": "I4.3D3",
        "status": "pass",
        "pass": True,
        "decision": {
            "i4_3d_temporal_provider_architecture_closed": True,
            "i4_3e_authorized": True,
        },
        "authority": {
            "metric_scale_claimed": False,
            "measurement_authoritative": False,
            "certified_metrology": False,
            "surface_mesh_generated": False,
        },
    }


def good_metric3d_environment(tmp_path: Path, *, available: bool = True) -> Metric3DV2EnvironmentProbe:
    root = (tmp_path / "project").resolve()
    source = root / "third_party/Metric3D"
    marker = source / ".eia_metric3d_v2_ready.json"
    checkpoint = root / "checkpoints/metric3d-v2/metric_depth_vit_small_800k.pth"
    python = root / "bin/python"
    source.mkdir(parents=True, exist_ok=True)
    marker.write_text(json.dumps({"commit": "e" * 40}) + "\n", encoding="utf-8")
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    checkpoint.write_bytes(b"checkpoint")
    python.parent.mkdir(parents=True, exist_ok=True)
    python.write_text("", encoding="utf-8")
    return Metric3DV2EnvironmentProbe(
        python_executable=str(python),
        project_root=str(root),
        source_directory=str(source),
        ready_marker=str(marker),
        checkpoint_path=str(checkpoint),
        source_revision="e" * 40,
        source_ready=available,
        checkpoint_ready=available,
        adapter_importable=available,
        torch_available=available,
        cuda_available=available,
        device_names=("Synthetic RTX",) if available else (),
        error="" if available else "not_ready",
    )


def make_metric3d_context(tmp_path: Path, *, view_count: int = 3, camera_ids: tuple[str, ...] | None = None):
    sfm = make_da3_sfm_result(tmp_path, view_count=view_count, camera_ids=camera_ids)
    env = good_metric3d_environment(tmp_path)
    worker = tmp_path / "single_image_worker.py"
    worker.write_text("", encoding="utf-8")
    context = Metric3DV2ProviderRunContext(
        sfm_result=sfm,
        reconstruction_frame_id="offline-sfm-reconstruction",
        output_dir=tmp_path / "metric3d-provider-output",
        python_executable=Path(env.python_executable),
        worker_script=worker,
        runtime_config={
            "source_dir": "third_party/Metric3D",
            "checkpoint": "checkpoints/metric3d-v2/metric_depth_vit_small_800k.pth",
            "entrypoint": "metric3d_vit_small",
            "model_label": "Metric3D v2 ViT-S",
            "input_height": 616,
            "input_width": 1064,
            "virtual_fov_degrees": 60.0,
            "point_budget": 250000,
        },
        device="cuda:0",
        precision="fp16",
    )
    return context, env


def explicit_metric3d_selection(view_count: int, *, provider_id: str = I43E1_PROVIDER_ID, allow_fallback: bool = False):
    request = DenseProviderSelectionRequest(
        mode=DenseProviderSelectionMode.EXPLICIT_PROVIDER,
        available_view_count=view_count,
        authoritative_sfm_available=True,
        authoritative_sfm_frame_id="offline-sfm-reconstruction",
        requested_provider_id=provider_id,
        allow_fallback=allow_fallback,
    )
    decision = DenseProviderSelectionDecision(
        request_mode=DenseProviderSelectionMode.EXPLICIT_PROVIDER,
        requested_provider_id=provider_id,
        selected_provider_id=provider_id,
        selected_implementation=DenseProviderImplementation.METRIC3D_V2,
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


class FakeMetric3DWorkerRunner:
    def __init__(
        self,
        *,
        source_revision: str = "e" * 40,
        depth_value: float = 1.0,
        confidence_value: float = 0.8,
        output_shape: tuple[int, int] = (48, 64),
        mutate_path: Path | None = None,
        native_depth_scale: str = "metric",
        native_is_metric: bool = True,
        intrinsics_source_native: str = "estimated",
        confidence_kind_native: str = "confidence",
    ) -> None:
        self.source_revision = source_revision
        self.depth_value = float(depth_value)
        self.confidence_value = float(confidence_value)
        self.output_shape = output_shape
        self.mutate_path = mutate_path
        self.native_depth_scale = native_depth_scale
        self.native_is_metric = native_is_metric
        self.intrinsics_source_native = intrinsics_source_native
        self.confidence_kind_native = confidence_kind_native
        self.calls = 0
        self.last_manifest = None

    def run(self, manifest_path: Path, output_dir: Path) -> Metric3DV2WorkerResult:
        self.calls += 1
        payload = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
        self.last_manifest = payload
        output_dir.mkdir(parents=True, exist_ok=True)
        h, w = self.output_shape
        records = []
        for index, item in enumerate(payload["sequence"]):
            depth = np.full((h, w), self.depth_value, dtype=np.float32)
            confidence = np.full((h, w), self.confidence_value, dtype=np.float32)
            valid_mask = np.ones((h, w), dtype=bool)
            evidence = output_dir / f"frame_{index:05d}.npz"
            np.savez_compressed(evidence, depth=depth, confidence=confidence, valid_mask=valid_mask)
            records.append(
                {
                    "order": index,
                    "keyframe_id": item["keyframe_id"],
                    "frame_id": item["frame_id"],
                    "source_sequence_index": item["source_sequence_index"],
                    "evidence_npz_path": str(evidence),
                    "width": w,
                    "height": h,
                    "native_is_metric": self.native_is_metric,
                    "native_depth_scale": self.native_depth_scale,
                    "intrinsics_source_native": self.intrinsics_source_native,
                    "confidence_kind_native": self.confidence_kind_native,
                    "model_intrinsics_are_canonical_authority": False,
                    "model_metric_depth_establishes_metric_scale": False,
                }
            )
        if self.mutate_path is not None:
            self.mutate_path.write_bytes(self.mutate_path.read_bytes() + b"mutation")
        result = {
            "schema_version": 1,
            "phase": "I4.3E1",
            "status": "pass",
            "model_id": "metric3d-v2",
            "source_revision": self.source_revision,
            "frame_count": len(records),
            "frame_records": records,
        }
        manifest = output_dir / "single_image_depth_result_manifest.json"
        manifest.write_text(json.dumps(result), encoding="utf-8")
        return Metric3DV2WorkerResult(manifest, result)
