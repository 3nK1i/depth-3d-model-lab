from __future__ import annotations

import hashlib
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
from core.reconstruction.depth_pro_dense_provider import (
    I43E4_PROVIDER_ID,
    DepthProEnvironmentProbe,
    DepthProProviderRunContext,
    DepthProWorkerResult,
)
from tests.i4_3c_helpers import make_da3_sfm_result


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def architecture_snapshot() -> dict:
    return {
        "schema_version": 2,
        "phase": "I4.3A",
        "status": "pass",
        "pass": True,
        "architecture": {
            "authoritative_camera_frame_owner": "pycolmap_sfm",
            "provider_implementation_contracts": [
                item.as_dict() for item in dense_provider_implementation_contracts()
            ],
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


def predecessor_snapshot(phase: str, implementation: str) -> dict:
    return {
        "schema_version": 1,
        "phase": phase,
        "status": "pass",
        "pass": True,
        "provider": {
            "implementation": implementation,
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


def e1_snapshot() -> dict:
    return predecessor_snapshot("I4.3E1", "metric3d_v2")


def e2_snapshot() -> dict:
    return predecessor_snapshot("I4.3E2", "unidepth_v2")


def e3_snapshot() -> dict:
    return predecessor_snapshot("I4.3E3", "moge_2")


def good_depth_pro_environment(
    tmp_path: Path,
    *,
    available: bool = True,
) -> DepthProEnvironmentProbe:
    root = (tmp_path / "project").resolve()
    checkpoint = root / "checkpoints/depth_pro.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    checkpoint.write_bytes(b"synthetic-depth-pro-checkpoint")
    python = root / "bin/python"
    python.parent.mkdir(parents=True, exist_ok=True)
    python.write_text("", encoding="utf-8")
    revision = "a" * 40
    checkpoint_sha = _sha256(checkpoint)
    return DepthProEnvironmentProbe(
        python_executable=str(python),
        project_root=str(root),
        distribution_name="depth-pro",
        distribution_version="0.1.synthetic",
        distribution_root=str(root / "site-packages/depth_pro"),
        source_repository="https://github.com/apple/ml-depth-pro.git",
        source_revision=revision,
        source_revision_resolved=available,
        requirement_commit_pinned=False,
        official_runtime_importable=available,
        official_runtime_source_matches_distribution=available,
        checkpoint_path=str(checkpoint),
        checkpoint_size_bytes=checkpoint.stat().st_size,
        checkpoint_sha256=checkpoint_sha,
        checkpoint_ready=available,
        adapter_importable=available,
        torch_available=available,
        cuda_available=available,
        device_names=("Synthetic RTX",) if available else (),
        error="" if available else "not_ready",
    )


def make_depth_pro_context(
    tmp_path: Path,
    *,
    view_count: int = 3,
    camera_ids: tuple[str, ...] | None = None,
):
    sfm = make_da3_sfm_result(tmp_path, view_count=view_count, camera_ids=camera_ids)
    env = good_depth_pro_environment(tmp_path)
    worker = tmp_path / "depth_pro_worker.py"
    worker.write_text("", encoding="utf-8")
    context = DepthProProviderRunContext(
        sfm_result=sfm,
        reconstruction_frame_id="offline-sfm-reconstruction",
        output_dir=tmp_path / "depth-pro-provider-output",
        python_executable=Path(env.python_executable),
        worker_script=worker,
        runtime_config={
            "checkpoint": env.checkpoint_path,
            "model_label": "Depth Pro",
            "internal_resolution": 1536,
            "point_budget": 250000,
            "warmup_size": 256,
            "warmup_runs": 1,
            "matmul_precision": "high",
            "hardware_tier": "efficient",
        },
        device="cuda:0",
        precision="fp16",
    )
    return context, env


def explicit_depth_pro_selection(
    view_count: int,
    *,
    provider_id: str = I43E4_PROVIDER_ID,
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
        selected_implementation=DenseProviderImplementation.DEPTH_PRO,
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


class FakeDepthProWorkerRunner:
    def __init__(
        self,
        *,
        source_revision: str = "a" * 40,
        checkpoint_sha256: str | None = None,
        depth_value: float = 1.0,
        output_shape: tuple[int, int] = (48, 64),
        mutate_path: Path | None = None,
        native_depth_scale: str = "metric",
        native_is_metric: bool = True,
        intrinsics_source_native: str = "model_predicted",
        confidence_kind_native: str = "none",
        direct_points_native: bool = False,
        normals_native: bool = False,
        promote_native_authority: bool = False,
    ) -> None:
        self.source_revision = source_revision
        self.checkpoint_sha256 = checkpoint_sha256
        self.depth_value = float(depth_value)
        self.output_shape = output_shape
        self.mutate_path = mutate_path
        self.native_depth_scale = native_depth_scale
        self.native_is_metric = native_is_metric
        self.intrinsics_source_native = intrinsics_source_native
        self.confidence_kind_native = confidence_kind_native
        self.direct_points_native = direct_points_native
        self.normals_native = normals_native
        self.promote_native_authority = promote_native_authority
        self.calls = 0
        self.last_manifest = None

    def run(self, manifest_path: Path, output_dir: Path) -> DepthProWorkerResult:
        self.calls += 1
        payload = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
        self.last_manifest = payload
        output_dir.mkdir(parents=True, exist_ok=True)
        h, w = self.output_shape
        records = []
        for index, item in enumerate(payload["sequence"]):
            depth = np.full((h, w), self.depth_value, dtype=np.float32)
            valid_mask = np.ones((h, w), dtype=bool)
            model_intrinsics = np.array(
                [[500.0, 0.0, (w - 1) * 0.5], [0.0, 500.0, (h - 1) * 0.5], [0.0, 0.0, 1.0]],
                dtype=np.float32,
            )
            evidence = output_dir / f"frame_{index:05d}.npz"
            np.savez_compressed(
                evidence,
                depth=depth,
                valid_mask=valid_mask,
                model_intrinsics=model_intrinsics,
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
                    "normals_native": self.normals_native,
                    "model_intrinsics_are_canonical_authority": self.promote_native_authority,
                    "model_metric_depth_establishes_metric_scale": self.promote_native_authority,
                }
            )
        if self.mutate_path is not None:
            self.mutate_path.write_bytes(self.mutate_path.read_bytes() + b"mutation")
        result = {
            "schema_version": 1,
            "phase": "I4.3E4",
            "status": "pass",
            "model_id": "depth-pro",
            "source_revision": self.source_revision,
            "checkpoint_path": payload["checkpoint_path"],
            "checkpoint_sha256": self.checkpoint_sha256 or payload["checkpoint_sha256"],
            "checkpoint_size_bytes": payload["checkpoint_size_bytes"],
            "frame_count": len(records),
            "frame_records": records,
            "authority": {
                "camera_pose_authority_owner": "pycolmap_sfm",
                "model_intrinsics_are_canonical_authority": self.promote_native_authority,
                "model_metric_depth_establishes_metric_scale": self.promote_native_authority,
            },
        }
        manifest = output_dir / "depth_pro_result_manifest.json"
        manifest.write_text(json.dumps(result), encoding="utf-8")
        return DepthProWorkerResult(manifest, result)
