from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path

import numpy as np

from core.reconstruction.depth_anything_v2_dense_provider import (
    I43E5_PROVIDER_ID,
    DepthAnythingV2EnvironmentProbe,
    DepthAnythingV2ProviderRunContext,
    DepthAnythingV2WorkerResult,
)
from core.reconstruction.dense_provider_architecture import (
    DenseProviderImplementation,
    DenseProviderSelectionDecision,
    DenseProviderSelectionMode,
    DenseProviderSelectionRequest,
    dense_provider_implementation_contracts,
)
from core.reconstruction.sfm_contracts import SfmLandmarkEstimate, SfmObservation
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


def e4_snapshot() -> dict:
    return predecessor_snapshot("I4.3E4", "depth_pro")


def good_depth_anything_v2_environment(
    tmp_path: Path,
    *,
    available: bool = True,
    variant: str = "small",
) -> DepthAnythingV2EnvironmentProbe:
    root = (tmp_path / "project").resolve()
    source = root / "third_party/Depth-Anything-V2"
    source.mkdir(parents=True, exist_ok=True)
    marker = source / ".depth3d_depth_anything_v2_ready.json"
    marker.write_text("{}\n", encoding="utf-8")
    python = root / "bin/python"
    python.parent.mkdir(parents=True, exist_ok=True)
    python.write_text("", encoding="utf-8")
    filename = {
        "small": "depth_anything_v2_vits.pth",
        "base": "depth_anything_v2_vitb.pth",
        "large": "depth_anything_v2_vitl.pth",
    }[variant]
    repository = {
        "small": "depth-anything/Depth-Anything-V2-Small",
        "base": "depth-anything/Depth-Anything-V2-Base",
        "large": "depth-anything/Depth-Anything-V2-Large",
    }[variant]
    checkpoint = root / "checkpoints/depth-anything-v2" / filename
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    checkpoint.write_bytes(b"synthetic-depth-anything-v2-checkpoint")
    revision = "a" * 40
    return DepthAnythingV2EnvironmentProbe(
        python_executable=str(python),
        project_root=str(root),
        source_directory=str(source),
        source_marker=str(marker),
        source_repository="https://github.com/DepthAnything/Depth-Anything-V2.git",
        source_revision=revision,
        source_checkout_matches_marker=available,
        installer_ref_commit_pinned=False,
        official_runtime_importable=available,
        official_runtime_source_matches_checkout=available,
        variant=variant,
        encoder={"small": "vits", "base": "vitb", "large": "vitl"}[variant],
        checkpoint_repository=repository,
        checkpoint_declared_revision="main",
        checkpoint_revision_commit_pinned=False,
        checkpoint_path=str(checkpoint),
        checkpoint_size_bytes=checkpoint.stat().st_size,
        checkpoint_sha256=_sha256(checkpoint),
        checkpoint_ready=available,
        adapter_importable=available,
        torch_available=available,
        cuda_available=available,
        device_names=("Synthetic RTX",) if available else (),
        error="" if available else "not_ready",
    )


def _vary_sfm_depths(sfm):
    cameras = {item.keyframe_id: item for item in sfm.cameras}
    varied: list[SfmLandmarkEstimate] = []
    for index, landmark in enumerate(sfm.landmarks):
        x, y, _ = np.asarray(landmark.position_reconstruction, dtype=np.float64)
        z = 1.45 + 0.055 * index
        position = np.asarray([x, y, z], dtype=np.float64)
        observations: list[SfmObservation] = []
        for observation in landmark.observations:
            camera = cameras[observation.keyframe_id]
            cam_x = float(camera.T_reconstruction_from_camera[0, 3])
            px = 100.0 * ((x - cam_x) / z) + 32.0
            py = 100.0 * (y / z) + 24.0
            observations.append(replace(observation, feature_xy=(float(px), float(py))))
        varied.append(
            replace(
                landmark,
                position_reconstruction=position,
                observations=tuple(observations),
            )
        )
    return replace(sfm, landmarks=tuple(varied))


def make_depth_anything_v2_context(
    tmp_path: Path,
    *,
    view_count: int = 3,
    camera_ids: tuple[str, ...] | None = None,
):
    sfm = _vary_sfm_depths(
        make_da3_sfm_result(tmp_path, view_count=view_count, camera_ids=camera_ids)
    )
    env = good_depth_anything_v2_environment(tmp_path)
    worker = tmp_path / "depth_anything_v2_worker.py"
    worker.write_text("", encoding="utf-8")
    context = DepthAnythingV2ProviderRunContext(
        sfm_result=sfm,
        reconstruction_frame_id="offline-sfm-reconstruction",
        output_dir=tmp_path / "depth-anything-v2-provider-output",
        python_executable=Path(env.python_executable),
        worker_script=worker,
        runtime_config={
            "source_dir": env.source_directory,
            "checkpoint": env.checkpoint_path,
            "encoder": "vits",
            "model_label": "Depth Anything V2 Small",
            "input_size": 518,
            "virtual_fov_degrees": 60.0,
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


def explicit_depth_anything_v2_selection(
    view_count: int,
    *,
    provider_id: str = I43E5_PROVIDER_ID,
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
        selected_implementation=DenseProviderImplementation.DEPTH_ANYTHING_V2,
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


class FakeDepthAnythingV2WorkerRunner:
    def __init__(
        self,
        sfm_result,
        *,
        source_revision: str = "a" * 40,
        checkpoint_sha256: str | None = None,
        output_shape: tuple[int, int] = (48, 64),
        mutate_path: Path | None = None,
        native_depth_scale: str = "relative",
        native_is_metric: bool = False,
        depth_representation_native: str = "relative_distance_from_inverse_depth",
        intrinsics_source_native: str = "estimated",
        confidence_kind_native: str = "none",
        direct_points_native: bool = False,
        normals_native: bool = False,
        adapter_projected_points_discarded: bool = True,
        promote_native_authority: bool = False,
        corrupt_alignment: bool = False,
    ) -> None:
        self.sfm_result = sfm_result
        self.source_revision = source_revision
        self.checkpoint_sha256 = checkpoint_sha256
        self.output_shape = output_shape
        self.mutate_path = mutate_path
        self.native_depth_scale = native_depth_scale
        self.native_is_metric = native_is_metric
        self.depth_representation_native = depth_representation_native
        self.intrinsics_source_native = intrinsics_source_native
        self.confidence_kind_native = confidence_kind_native
        self.direct_points_native = direct_points_native
        self.normals_native = normals_native
        self.adapter_projected_points_discarded = adapter_projected_points_discarded
        self.promote_native_authority = promote_native_authority
        self.corrupt_alignment = corrupt_alignment
        self.calls = 0
        self.last_manifest = None

    def _relative_depth_for_keyframe(self, keyframe_id: str, h: int, w: int) -> np.ndarray:
        # Synthetic relation: inv_z_sfm = a * inv_dav2 + b.
        slope = -0.35 if self.corrupt_alignment else 0.72
        intercept = 0.08
        relative = np.full((h, w), 2.0, dtype=np.float32)
        camera = next(item for item in self.sfm_result.cameras if item.keyframe_id == keyframe_id)
        from core.reconstruction.sfm_coordinates import invert_rigid_matrix
        T = invert_rigid_matrix(camera.T_reconstruction_from_camera)
        for landmark in self.sfm_result.landmarks:
            obs = next((item for item in landmark.observations if item.keyframe_id == keyframe_id), None)
            if obs is None:
                continue
            camera_point = T[:3, :3] @ landmark.position_reconstruction + T[:3, 3]
            z = float(camera_point[2])
            target_inv = 1.0 / z
            source_inv = (target_inv - intercept) / slope
            if source_inv <= 0.0:
                continue
            value = 1.0 / source_inv
            x, y = obs.feature_xy
            x0 = int(np.floor(x))
            y0 = int(np.floor(y))
            for yy in (y0, min(y0 + 1, h - 1)):
                for xx in (x0, min(x0 + 1, w - 1)):
                    if 0 <= xx < w and 0 <= yy < h:
                        relative[yy, xx] = value
        return relative

    def run(self, manifest_path: Path, output_dir: Path) -> DepthAnythingV2WorkerResult:
        self.calls += 1
        payload = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
        self.last_manifest = payload
        output_dir.mkdir(parents=True, exist_ok=True)
        h, w = self.output_shape
        records = []
        for index, item in enumerate(payload["sequence"]):
            relative_depth = self._relative_depth_for_keyframe(item["keyframe_id"], h, w)
            valid_mask = np.ones((h, w), dtype=bool)
            estimated_intrinsics = np.array(
                [[500.0, 0.0, (w - 1) * 0.5], [0.0, 500.0, (h - 1) * 0.5], [0.0, 0.0, 1.0]],
                dtype=np.float32,
            )
            evidence = output_dir / f"frame_{index:05d}.npz"
            np.savez_compressed(
                evidence,
                relative_depth=relative_depth,
                valid_mask=valid_mask,
                estimated_intrinsics=estimated_intrinsics,
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
                    "depth_representation_native": self.depth_representation_native,
                    "intrinsics_source_native": self.intrinsics_source_native,
                    "confidence_kind_native": self.confidence_kind_native,
                    "direct_points_native": self.direct_points_native,
                    "normals_native": self.normals_native,
                    "adapter_projected_points_discarded": self.adapter_projected_points_discarded,
                    "estimated_model_intrinsics_are_canonical_authority": self.promote_native_authority,
                    "adapter_projected_points_are_canonical_geometry": self.promote_native_authority,
                    "relative_depth_establishes_metric_scale": self.promote_native_authority,
                }
            )
        if self.mutate_path is not None:
            self.mutate_path.write_bytes(self.mutate_path.read_bytes() + b"mutation")
        result = {
            "schema_version": 1,
            "phase": "I4.3E5",
            "status": "pass",
            "model_id": "depth-anything-v2",
            "source_revision": self.source_revision,
            "variant": payload["variant"],
            "encoder": payload["encoder"],
            "checkpoint_repository": payload["checkpoint_repository"],
            "checkpoint_declared_revision": payload["checkpoint_declared_revision"],
            "checkpoint_path": payload["checkpoint_path"],
            "checkpoint_sha256": self.checkpoint_sha256 or payload["checkpoint_sha256"],
            "checkpoint_size_bytes": payload["checkpoint_size_bytes"],
            "frame_count": len(records),
            "frame_records": records,
            "authority": {
                "camera_pose_authority_owner": "pycolmap_sfm",
                "estimated_model_intrinsics_are_canonical_authority": self.promote_native_authority,
                "adapter_projected_points_are_canonical_geometry": self.promote_native_authority,
                "relative_depth_establishes_metric_scale": self.promote_native_authority,
            },
        }
        manifest = output_dir / "depth_anything_v2_result_manifest.json"
        manifest.write_text(json.dumps(result), encoding="utf-8")
        return DepthAnythingV2WorkerResult(manifest, result)
