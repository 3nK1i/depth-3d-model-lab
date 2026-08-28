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
    dense_provider_implementation_contracts,
)
from core.reconstruction.marigold_dense_provider import (
    I43E6_PROVIDER_ID,
    MarigoldEnvironmentProbe,
    MarigoldProviderRunContext,
    MarigoldWorkerResult,
)
from core.reconstruction.sfm_contracts import SfmLandmarkEstimate, SfmObservation
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
        "decision": {"i4_3d_temporal_provider_architecture_closed": True, "i4_3e_authorized": True},
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
        "provider": {"implementation": implementation, "qualification_state": "candidate"},
        "authority": {
            "provider_promoted": False,
            "metric_scale_claimed": False,
            "measurement_authoritative": False,
            "certified_metrology": False,
            "surface_mesh_generated": False,
        },
    }


def e1_snapshot() -> dict: return predecessor_snapshot("I4.3E1", "metric3d_v2")
def e2_snapshot() -> dict: return predecessor_snapshot("I4.3E2", "unidepth_v2")
def e3_snapshot() -> dict: return predecessor_snapshot("I4.3E3", "moge_2")
def e4_snapshot() -> dict: return predecessor_snapshot("I4.3E4", "depth_pro")
def e5_snapshot() -> dict: return predecessor_snapshot("I4.3E5", "depth_anything_v2")


def good_marigold_environment(tmp_path: Path, *, available: bool = True, profile: str = "efficient") -> MarigoldEnvironmentProbe:
    root = (tmp_path / "project").resolve()
    python = root / "bin/python"
    python.parent.mkdir(parents=True, exist_ok=True)
    python.write_text("", encoding="utf-8")
    checkpoint = root / "checkpoints/marigold/depth-v1-1"
    checkpoint.mkdir(parents=True, exist_ok=True)
    (checkpoint / ".ready").write_text("{}\n", encoding="utf-8")
    (checkpoint / "model_index.json").write_text("{}\n", encoding="utf-8")
    return MarigoldEnvironmentProbe(
        python_executable=str(python),
        project_root=str(root),
        distribution_name="diffusers",
        distribution_version="0.35.1",
        distribution_root=str(root / "site-packages/diffusers"),
        official_runtime_importable=available,
        official_runtime_source_matches_distribution=available,
        runtime_requirement_exact_pinned=False,
        profile=profile,
        checkpoint_repository="prs-eth/marigold-depth-v1-1",
        checkpoint_requested_revision="main",
        checkpoint_resolved_revision="a" * 40,
        checkpoint_revision_pinned=available,
        checkpoint_path=str(checkpoint),
        checkpoint_marker_sha256="b" * 64,
        model_index_sha256="c" * 64,
        checkpoint_file_count=20,
        checkpoint_size_bytes=2_000_000_000,
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
        varied.append(replace(landmark, position_reconstruction=position, observations=tuple(observations)))
    return replace(sfm, landmarks=tuple(varied))


def make_marigold_context(tmp_path: Path, *, view_count: int = 3, camera_ids: tuple[str, ...] | None = None, profile: str = "efficient"):
    sfm = _vary_sfm_depths(make_da3_sfm_result(tmp_path, view_count=view_count, camera_ids=camera_ids))
    env = good_marigold_environment(tmp_path, profile=profile)
    worker = tmp_path / "marigold_worker.py"
    worker.write_text("", encoding="utf-8")
    context = MarigoldProviderRunContext(
        sfm_result=sfm,
        reconstruction_frame_id="offline-sfm-reconstruction",
        output_dir=tmp_path / "marigold-provider-output",
        python_executable=Path(env.python_executable),
        worker_script=worker,
        runtime_config={
            "repository": env.checkpoint_repository,
            "checkpoint_dir": env.checkpoint_path,
            "model_label": "Marigold Test",
            "prediction_type": "depth",
            "processing_resolution": 512,
            "num_inference_steps": 1,
            "ensemble_size": 1,
            "output_uncertainty": False,
            "model_cpu_offload": True,
            "virtual_fov_degrees": 60.0,
            "point_budget": 250000,
            "warmup_size": 192,
            "warmup_runs": 1,
            "matmul_precision": "high",
            "hardware_tier": profile,
            "profile": profile,
        },
        device="cuda:0",
        precision="fp16",
    )
    return context, env


def explicit_marigold_selection(view_count: int, *, provider_id: str = I43E6_PROVIDER_ID, allow_fallback: bool = False):
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
        selected_implementation=DenseProviderImplementation.MARIGOLD,
        considered_provider_ids=(provider_id,),
        automatic=False,
        engineer_override_applied=False,
        fallback_used=False,
        reason_codes=("explicit_engineer_selection",),
    )
    return request, decision


class CollectingProgressSink:
    def __init__(self) -> None: self.events = []
    def publish(self, event) -> None: self.events.append(event)


class FakeMarigoldWorkerRunner:
    def __init__(
        self,
        sfm_result,
        *,
        prediction_type: str = "depth",
        uncertainty: bool = False,
        output_shape: tuple[int, int] = (48, 64),
        mutate_path: Path | None = None,
        native_depth_scale: str = "relative",
        native_is_metric: bool = False,
        depth_representation_native: str = "affine_invariant_relative_depth_or_disparity",
        intrinsics_source_native: str = "estimated",
        confidence_kind_native: str | None = None,
        direct_points_native: bool = False,
        normals_native: bool = False,
        adapter_projected_points_discarded: bool = True,
        promote_native_authority: bool = False,
        corrupt_alignment: bool = False,
        checkpoint_revision: str | None = None,
    ) -> None:
        self.sfm_result = sfm_result
        self.prediction_type = prediction_type
        self.uncertainty = uncertainty
        self.output_shape = output_shape
        self.mutate_path = mutate_path
        self.native_depth_scale = native_depth_scale
        self.native_is_metric = native_is_metric
        self.depth_representation_native = depth_representation_native
        self.intrinsics_source_native = intrinsics_source_native
        self.confidence_kind_native = confidence_kind_native or ("uncertainty" if uncertainty else "none")
        self.direct_points_native = direct_points_native
        self.normals_native = normals_native
        self.adapter_projected_points_discarded = adapter_projected_points_discarded
        self.promote_native_authority = promote_native_authority
        self.corrupt_alignment = corrupt_alignment
        self.checkpoint_revision = checkpoint_revision
        self.calls = 0
        self.last_manifest = None

    def _relative_depth_for_keyframe(self, keyframe_id: str, h: int, w: int) -> np.ndarray:
        from core.reconstruction.sfm_coordinates import invert_rigid_matrix
        relative = np.full((h, w), 1.0, dtype=np.float32)
        camera = next(item for item in self.sfm_result.cameras if item.keyframe_id == keyframe_id)
        T = invert_rigid_matrix(camera.T_reconstruction_from_camera)
        slope = -0.35 if self.corrupt_alignment else 0.72
        intercept = 0.08
        for landmark in self.sfm_result.landmarks:
            obs = next((item for item in landmark.observations if item.keyframe_id == keyframe_id), None)
            if obs is None: continue
            camera_point = T[:3, :3] @ landmark.position_reconstruction + T[:3, 3]
            z = float(camera_point[2])
            if self.prediction_type == "depth":
                value = (z - intercept) / slope
            else:
                source_inv = ((1.0 / z) - intercept) / slope
                value = 1.0 / source_inv if source_inv > 0.0 else -1.0
            if value <= 0.0: continue
            x, y = obs.feature_xy
            x0 = int(np.floor(x)); y0 = int(np.floor(y))
            for yy in (y0, min(y0 + 1, h - 1)):
                for xx in (x0, min(x0 + 1, w - 1)):
                    if 0 <= xx < w and 0 <= yy < h:
                        relative[yy, xx] = value
        return relative

    def run(self, manifest_path: Path, output_dir: Path) -> MarigoldWorkerResult:
        self.calls += 1
        payload = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
        self.last_manifest = payload
        output_dir.mkdir(parents=True, exist_ok=True)
        h, w = self.output_shape
        records = []
        for index, item in enumerate(payload["sequence"]):
            relative_depth = self._relative_depth_for_keyframe(item["keyframe_id"], h, w)
            valid_mask = np.ones((h, w), dtype=bool)
            estimated_intrinsics = np.array([[500.0, 0.0, (w - 1) * 0.5], [0.0, 500.0, (h - 1) * 0.5], [0.0, 0.0, 1.0]], dtype=np.float32)
            evidence = output_dir / f"frame_{index:05d}.npz"
            arrays = {"relative_depth": relative_depth, "valid_mask": valid_mask, "estimated_intrinsics": estimated_intrinsics}
            if self.uncertainty:
                arrays["uncertainty"] = np.full((h, w), 0.25, dtype=np.float32)
            np.savez_compressed(evidence, **arrays)
            records.append({
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
                "prediction_type": self.prediction_type,
                "intrinsics_source_native": self.intrinsics_source_native,
                "confidence_kind_native": self.confidence_kind_native,
                "uncertainty_available": self.uncertainty,
                "direct_points_native": self.direct_points_native,
                "normals_native": self.normals_native,
                "adapter_projected_points_discarded": self.adapter_projected_points_discarded,
                "estimated_model_intrinsics_are_canonical_authority": self.promote_native_authority,
                "adapter_projected_points_are_canonical_geometry": self.promote_native_authority,
                "relative_depth_establishes_metric_scale": self.promote_native_authority,
            })
        if self.mutate_path is not None:
            self.mutate_path.write_bytes(self.mutate_path.read_bytes() + b"mutation")
        result = {
            "schema_version": 1,
            "phase": "I4.3E6",
            "status": "pass",
            "model_id": "marigold",
            "runtime_distribution": payload["runtime_distribution"],
            "runtime_distribution_version": payload["runtime_distribution_version"],
            "profile": payload["profile"],
            "checkpoint_repository": payload["checkpoint_repository"],
            "checkpoint_resolved_revision": self.checkpoint_revision or payload["checkpoint_resolved_revision"],
            "checkpoint_path": payload["checkpoint_path"],
            "checkpoint_marker_sha256": payload["checkpoint_marker_sha256"],
            "model_index_sha256": payload["model_index_sha256"],
            "frame_count": len(records),
            "frame_records": records,
            "authority": {
                "camera_pose_authority_owner": "pycolmap_sfm",
                "estimated_model_intrinsics_are_canonical_authority": self.promote_native_authority,
                "adapter_projected_points_are_canonical_geometry": self.promote_native_authority,
                "relative_depth_establishes_metric_scale": self.promote_native_authority,
            },
        }
        manifest = output_dir / "marigold_result_manifest.json"
        manifest.write_text(json.dumps(result), encoding="utf-8")
        return MarigoldWorkerResult(manifest, result)
