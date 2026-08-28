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
from core.reconstruction.unidepth_v2_dense_provider import (
    I43E2_PROVIDER_ID,
    UniDepthV2EnvironmentProbe,
    UniDepthV2ProviderRunContext,
    UniDepthV2WorkerResult,
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


def e1_snapshot() -> dict:
    return {
        "schema_version": 1,
        "phase": "I4.3E1",
        "status": "pass",
        "pass": True,
        "provider": {
            "implementation": "metric3d_v2",
            "qualification_state": "candidate",
        },
        "authority": {
            "provider_promoted": False,
            "metric_scale_claimed": False,
            "measurement_authoritative": False,
            "certified_metrology": False,
            "surface_mesh_generated": False,
        },
    }


def good_unidepth_environment(tmp_path: Path, *, available: bool = True) -> UniDepthV2EnvironmentProbe:
    root = (tmp_path / "project").resolve()
    source = root / "third_party/UniDepth"
    marker = source / ".eia_unidepth_v2_compat.json"
    checkpoint = root / "checkpoints/unidepth-v2/vits14"
    python = root / "bin/python"
    source.mkdir(parents=True, exist_ok=True)
    marker.write_text(
        json.dumps(
            {
                "commit": "8" * 40,
                "purpose": "V2-only imports; model and inference implementation unchanged",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    checkpoint.mkdir(parents=True, exist_ok=True)
    (checkpoint / "config.json").write_text("{}", encoding="utf-8")
    (checkpoint / "model.safetensors").write_bytes(b"weights")
    python.parent.mkdir(parents=True, exist_ok=True)
    python.write_text("", encoding="utf-8")
    return UniDepthV2EnvironmentProbe(
        python_executable=str(python),
        project_root=str(root),
        source_directory=str(source),
        compatibility_marker=str(marker),
        checkpoint_path=str(checkpoint),
        source_revision="8" * 40,
        checkpoint_expected_revision="0" * 40,
        variant="small",
        repository="lpiccinelli/unidepth-v2-vits14",
        source_ready=available,
        source_checkout_matches_marker=available,
        checkpoint_ready=available,
        adapter_importable=available,
        official_runtime_importable=available,
        official_runtime_source_matches_checkout=available,
        torch_available=available,
        cuda_available=available,
        device_names=("Synthetic RTX",) if available else (),
        error="" if available else "not_ready",
    )


def make_unidepth_context(
    tmp_path: Path,
    *,
    view_count: int = 3,
    camera_ids: tuple[str, ...] | None = None,
):
    sfm = make_da3_sfm_result(tmp_path, view_count=view_count, camera_ids=camera_ids)
    env = good_unidepth_environment(tmp_path)
    worker = tmp_path / "unidepth_worker.py"
    worker.write_text("", encoding="utf-8")
    context = UniDepthV2ProviderRunContext(
        sfm_result=sfm,
        reconstruction_frame_id="offline-sfm-reconstruction",
        output_dir=tmp_path / "unidepth-provider-output",
        python_executable=Path(env.python_executable),
        worker_script=worker,
        runtime_config={
            "checkpoint_dir": "checkpoints/unidepth-v2/vits14",
            "repository": "lpiccinelli/unidepth-v2-vits14",
            "variant": "vits14",
            "model_label": "UniDepthV2 ViT-S/14",
            "resolution_level": 2,
            "interpolation_mode": "bilinear",
            "point_budget": 250000,
            "warmup_size": 256,
            "warmup_runs": 1,
            "allow_tf32": False,
            "matmul_precision": "high",
            "hardware_tier": "efficient",
        },
        device="cuda:0",
        precision="fp16",
    )
    return context, env


def explicit_unidepth_selection(
    view_count: int,
    *,
    provider_id: str = I43E2_PROVIDER_ID,
    allow_fallback: bool = False,
):
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
        selected_implementation=DenseProviderImplementation.UNIDEPTH_V2,
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


class FakeUniDepthWorkerRunner:
    def __init__(
        self,
        *,
        source_revision: str = "8" * 40,
        checkpoint_expected_revision: str = "0" * 40,
        variant: str = "small",
        depth_value: float = 1.0,
        uncertainty_value: float = 0.25,
        output_shape: tuple[int, int] = (48, 64),
        mutate_path: Path | None = None,
        native_depth_scale: str = "metric",
        native_is_metric: bool = True,
        intrinsics_source_native: str = "model_predicted",
        confidence_kind_native: str = "uncertainty",
        direct_points_native: bool = True,
        direct_points_consumed_for_canonical_geometry: bool = False,
    ) -> None:
        self.source_revision = source_revision
        self.checkpoint_expected_revision = checkpoint_expected_revision
        self.variant = variant
        self.depth_value = float(depth_value)
        self.uncertainty_value = float(uncertainty_value)
        self.output_shape = output_shape
        self.mutate_path = mutate_path
        self.native_depth_scale = native_depth_scale
        self.native_is_metric = native_is_metric
        self.intrinsics_source_native = intrinsics_source_native
        self.confidence_kind_native = confidence_kind_native
        self.direct_points_native = direct_points_native
        self.direct_points_consumed_for_canonical_geometry = direct_points_consumed_for_canonical_geometry
        self.calls = 0
        self.last_manifest = None

    def run(self, manifest_path: Path, output_dir: Path) -> UniDepthV2WorkerResult:
        self.calls += 1
        payload = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
        self.last_manifest = payload
        output_dir.mkdir(parents=True, exist_ok=True)
        h, w = self.output_shape
        records = []
        for index, item in enumerate(payload["sequence"]):
            depth = np.full((h, w), self.depth_value, dtype=np.float32)
            uncertainty = np.full((h, w), self.uncertainty_value, dtype=np.float32)
            valid_mask = np.ones((h, w), dtype=bool)
            direct_points = np.zeros((max(1, h // 2), max(1, w // 2), 3), dtype=np.float32)
            direct_points[..., 2] = self.depth_value
            intrinsics = np.asarray(
                [[100.0, 0.0, w / 2.0], [0.0, 100.0, h / 2.0], [0.0, 0.0, 1.0]],
                dtype=np.float32,
            )
            evidence = output_dir / f"frame_{index:05d}.npz"
            np.savez_compressed(
                evidence,
                depth=depth,
                uncertainty=uncertainty,
                valid_mask=valid_mask,
                direct_points_model=direct_points,
                model_intrinsics=intrinsics,
            )
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
                    "direct_points_native": self.direct_points_native,
                    "direct_points_consumed_for_canonical_geometry": self.direct_points_consumed_for_canonical_geometry,
                    "model_intrinsics_are_canonical_authority": False,
                    "model_metric_depth_establishes_metric_scale": False,
                }
            )
        if self.mutate_path is not None:
            self.mutate_path.write_bytes(self.mutate_path.read_bytes() + b"mutation")
        result = {
            "schema_version": 1,
            "phase": "I4.3E2",
            "status": "pass",
            "model_id": "unidepth-v2",
            "source_revision": self.source_revision,
            "checkpoint_expected_revision": self.checkpoint_expected_revision,
            "variant": self.variant,
            "frame_count": len(records),
            "frame_records": records,
        }
        manifest = output_dir / "unidepth_v2_result_manifest.json"
        manifest.write_text(json.dumps(result), encoding="utf-8")
        return UniDepthV2WorkerResult(manifest, result)
