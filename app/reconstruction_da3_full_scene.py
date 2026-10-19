"""Full-scene recorded-video DA3 build contracts for Phase 9.6E2.

The production ``depth`` process owns orchestration only. Heavy Depth Anything 3
inference remains in the qualified external ``depth3d-da3`` interpreter. Full-scene
coverage is produced from quality-filtered keyframes and bounded overlapping
three-view windows; it never turns DA3 into canonical, metric-scale,
measurement, QC, or certified-metrology authority.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from pathlib import Path
import os
import sys
from typing import Sequence
from uuid import uuid4

import numpy as np

from app.reconstruction_processing_jobs import (
    DigitalTwinProcessingCommand,
    DigitalTwinProcessingJobSpec,
    DigitalTwinProcessingKind,
)
from core.reconstruction.da3 import DA3_EFFICIENT_MODEL_ID
from core.reconstruction.errors import ReconstructionValidationError

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DA3_PYTHON = Path.home() / "anaconda3/envs/depth3d-da3/bin/python"
DEFAULT_DA3_REPOSITORY = Path.home() / "Desktop/depth3d-da3-backends/depth-anything-3"

def _default_da3_python_path() -> Path:
    raw = os.environ.get("DEPTH3D_RECONSTRUCTION_DA3_PYTHON", "").strip()
    if not raw:
        raw = os.environ.get("EIA_RECONSTRUCTION_DA3_PYTHON", "").strip()
    return Path(raw).expanduser().resolve() if raw else DEFAULT_DA3_PYTHON.expanduser().resolve()

def _default_da3_repository_path() -> Path:
    raw = os.environ.get("DEPTH3D_RECONSTRUCTION_DA3_REPOSITORY", "").strip()
    if not raw:
        raw = os.environ.get("EIA_RECONSTRUCTION_DA3_REPOSITORY", "").strip()
    return Path(raw).expanduser().resolve() if raw else DEFAULT_DA3_REPOSITORY.expanduser().resolve()

DA3_FULL_SCENE_WORKER = PROJECT_ROOT / "scripts" / "run_phase9_6e2_full_scene_da3_job.py"
DA3_FULL_SCENE_PROTOCOL = "depth3d.digital-twin.da3-full-scene.v1"


@dataclass(frozen=True, slots=True)
class DA3FullSceneProfile:
    """Conservative RTX-2060-first profile for full-video temporal coverage."""

    model_id: str = DA3_EFFICIENT_MODEL_ID
    target_fps_hz: float = 5.0
    maximum_keyframes: int = 96
    minimum_spacing_seconds: float = 0.50
    forced_spacing_seconds: float = 2.0
    minimum_visual_change: float = 0.055
    minimum_quality_score: float = 0.12
    window_size: int = 3
    window_overlap: int = 2
    process_res: int = 392
    pixel_stride: int = 6
    per_window_timeout_seconds: float = 180.0

    def __post_init__(self) -> None:
        rate = float(self.target_fps_hz)
        if not isfinite(rate) or not 0.5 <= rate <= 15.0:
            raise ReconstructionValidationError("Full-scene target_fps_hz must be within [0.5, 15].")
        maximum = int(self.maximum_keyframes)
        if not 3 <= maximum <= 250:
            raise ReconstructionValidationError("Full-scene maximum_keyframes must be within [3, 250].")
        minimum_spacing = float(self.minimum_spacing_seconds)
        forced_spacing = float(self.forced_spacing_seconds)
        if not isfinite(minimum_spacing) or minimum_spacing <= 0.0:
            raise ReconstructionValidationError("minimum_spacing_seconds must be positive.")
        if not isfinite(forced_spacing) or forced_spacing < minimum_spacing:
            raise ReconstructionValidationError("forced_spacing_seconds must be >= minimum_spacing_seconds.")
        change = float(self.minimum_visual_change)
        quality = float(self.minimum_quality_score)
        if not 0.0 <= change <= 1.0 or not 0.0 <= quality <= 1.0:
            raise ReconstructionValidationError("Full-scene visual-change/quality thresholds must be in [0,1].")
        window = int(self.window_size)
        overlap = int(self.window_overlap)
        if window != 3 or overlap != 2:
            raise ReconstructionValidationError(
                "Phase 9.6E2 is qualified only for bounded three-view windows with two-view overlap."
            )
        resolution = int(self.process_res)
        stride = int(self.pixel_stride)
        if not 196 <= resolution <= 1008:
            raise ReconstructionValidationError("DA3 process_res must be within [196, 1008].")
        if not 1 <= stride <= 32:
            raise ReconstructionValidationError("DA3 full-scene pixel_stride must be within [1, 32].")
        timeout = float(self.per_window_timeout_seconds)
        if not isfinite(timeout) or timeout <= 0.0:
            raise ReconstructionValidationError("per_window_timeout_seconds must be positive.")
        object.__setattr__(self, "target_fps_hz", rate)
        object.__setattr__(self, "maximum_keyframes", maximum)
        object.__setattr__(self, "minimum_spacing_seconds", minimum_spacing)
        object.__setattr__(self, "forced_spacing_seconds", forced_spacing)
        object.__setattr__(self, "minimum_visual_change", change)
        object.__setattr__(self, "minimum_quality_score", quality)
        object.__setattr__(self, "window_size", window)
        object.__setattr__(self, "window_overlap", overlap)
        object.__setattr__(self, "process_res", resolution)
        object.__setattr__(self, "pixel_stride", stride)
        object.__setattr__(self, "per_window_timeout_seconds", timeout)

    @property
    def profile_label(self) -> str:
        return f"Full Scene • ≤{self.maximum_keyframes} keyframes • 3-view windows"


def plan_overlapping_windows(frame_count: int, *, window_size: int = 3, overlap: int = 2) -> tuple[tuple[int, ...], ...]:
    count = int(frame_count)
    window = int(window_size)
    shared = int(overlap)
    if count < window:
        raise ReconstructionValidationError("Full-scene DA3 requires at least three selected keyframes.")
    if window < 2 or not 1 <= shared < window:
        raise ReconstructionValidationError("Invalid full-scene DA3 window/overlap configuration.")
    step = window - shared
    starts = list(range(0, count - window + 1, step))
    final_start = count - window
    if starts[-1] != final_start:
        starts.append(final_start)
    return tuple(tuple(range(start, start + window)) for start in starts)


def _w2c_to_c2w(extrinsic: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    matrix = np.asarray(extrinsic, dtype=np.float64)
    if matrix.shape == (3, 4):
        rotation_w2c = matrix[:, :3]
        translation_w2c = matrix[:, 3]
    elif matrix.shape == (4, 4):
        rotation_w2c = matrix[:3, :3]
        translation_w2c = matrix[:3, 3]
    else:
        raise ReconstructionValidationError("DA3 extrinsic must be 3x4 or 4x4.")
    rotation_c2w = rotation_w2c.T
    center = -rotation_c2w @ translation_w2c
    return rotation_c2w, center


def _c2w_to_w2c(rotation_c2w: np.ndarray, center: np.ndarray) -> np.ndarray:
    rotation = np.asarray(rotation_c2w, dtype=np.float64)
    centre = np.asarray(center, dtype=np.float64).reshape(3)
    if rotation.shape != (3, 3):
        raise ReconstructionValidationError("Camera rotation must be 3x3.")
    rotation_w2c = rotation.T
    translation = -rotation_w2c @ centre
    output = np.eye(4, dtype=np.float32)
    output[:3, :3] = rotation_w2c.astype(np.float32)
    output[:3, 3] = translation.astype(np.float32)
    return output


def _nearest_rotation(matrix: np.ndarray) -> np.ndarray:
    u, _s, vt = np.linalg.svd(np.asarray(matrix, dtype=np.float64))
    rotation = u @ vt
    if np.linalg.det(rotation) < 0.0:
        u[:, -1] *= -1.0
        rotation = u @ vt
    return rotation


def estimate_overlap_similarity(
    global_extrinsics_w2c: Sequence[np.ndarray],
    local_extrinsics_w2c: Sequence[np.ndarray],
    global_depths: Sequence[np.ndarray],
    local_depths: Sequence[np.ndarray],
) -> tuple[float, np.ndarray, np.ndarray]:
    """Estimate local→global arbitrary-gauge similarity from two shared views.

    Scale comes from robust overlapping depth ratios; rotation/translation come
    from the same cameras' poses. This does not create physical/metric scale.
    """

    if not (len(global_extrinsics_w2c) == len(local_extrinsics_w2c) == len(global_depths) == len(local_depths) == 2):
        raise ReconstructionValidationError("E2 overlap alignment requires exactly two shared views.")
    ratios: list[np.ndarray] = []
    rotation_candidates: list[np.ndarray] = []
    global_centres: list[np.ndarray] = []
    local_centres: list[np.ndarray] = []
    for g_ext, l_ext, g_depth, l_depth in zip(
        global_extrinsics_w2c, local_extrinsics_w2c, global_depths, local_depths
    ):
        gd = np.asarray(g_depth, dtype=np.float64)
        ld = np.asarray(l_depth, dtype=np.float64)
        if gd.shape != ld.shape or gd.ndim != 2:
            raise ReconstructionValidationError("Overlapping DA3 depth maps must share one HxW shape.")
        valid = np.isfinite(gd) & np.isfinite(ld) & (gd > 1e-6) & (ld > 1e-6)
        values = (gd[valid] / ld[valid])
        if values.size < 64:
            raise ReconstructionValidationError("Insufficient valid overlapping depth samples for DA3 gauge alignment.")
        lo, hi = np.quantile(values, [0.10, 0.90])
        clipped = values[(values >= lo) & (values <= hi)]
        ratios.append(clipped if clipped.size else values)
        g_rot, g_center = _w2c_to_c2w(np.asarray(g_ext))
        l_rot, l_center = _w2c_to_c2w(np.asarray(l_ext))
        rotation_candidates.append(g_rot @ l_rot.T)
        global_centres.append(g_center)
        local_centres.append(l_center)
    scale = float(np.median(np.concatenate(ratios)))
    if not isfinite(scale) or not 0.05 <= scale <= 20.0:
        raise ReconstructionValidationError("DA3 overlap produced an unstable arbitrary-gauge scale ratio.")
    rotation = _nearest_rotation(sum(rotation_candidates) / len(rotation_candidates))
    translations = [g - scale * (rotation @ l) for g, l in zip(global_centres, local_centres)]
    translation = np.mean(np.stack(translations, axis=0), axis=0)
    if not np.isfinite(rotation).all() or not np.isfinite(translation).all():
        raise ReconstructionValidationError("DA3 overlap similarity contains non-finite values.")
    return scale, rotation, translation


def transform_chunk_to_global_gauge(
    extrinsics_w2c: np.ndarray,
    depth: np.ndarray,
    *,
    scale: float,
    rotation: np.ndarray,
    translation: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    extrinsics = np.asarray(extrinsics_w2c)
    depths = np.asarray(depth, dtype=np.float32)
    if extrinsics.ndim != 3 or depths.ndim != 3 or extrinsics.shape[0] != depths.shape[0]:
        raise ReconstructionValidationError("DA3 chunk pose/depth frame counts do not match.")
    output_poses = []
    for extrinsic in extrinsics:
        local_rotation, local_center = _w2c_to_c2w(extrinsic)
        global_rotation = np.asarray(rotation, dtype=np.float64) @ local_rotation
        global_center = float(scale) * (np.asarray(rotation, dtype=np.float64) @ local_center) + np.asarray(translation, dtype=np.float64)
        output_poses.append(_c2w_to_w2c(global_rotation, global_center))
    return np.stack(output_poses, axis=0), np.ascontiguousarray(depths * float(scale), dtype=np.float32)


def build_full_scene_da3_job(
    *,
    video_path: str | Path,
    output_root: str | Path,
    profile: DA3FullSceneProfile | None = None,
    da3_python: str | Path | None = None,
    da3_repository: str | Path | None = None,
) -> tuple[DigitalTwinProcessingJobSpec, DigitalTwinProcessingCommand]:
    video = Path(video_path).expanduser().resolve()
    if not video.is_file():
        raise ReconstructionValidationError(f"Full-scene DA3 recorded video does not exist: {video}")
    worker = DA3_FULL_SCENE_WORKER.resolve()
    if not worker.is_file():
        raise ReconstructionValidationError(f"Full-scene DA3 orchestration worker is missing: {worker}")
    active_profile = profile or DA3FullSceneProfile()
    external_python = Path(da3_python or _default_da3_python_path()).expanduser().resolve()
    external_repo = Path(da3_repository or _default_da3_repository_path()).expanduser().resolve()
    if not external_python.is_file():
        raise ReconstructionValidationError(f"Qualified DA3 Python is missing: {external_python}")
    if not external_repo.is_dir():
        raise ReconstructionValidationError(f"Qualified DA3 checkout is missing: {external_repo}")
    job_id = str(uuid4())
    root = Path(output_root).expanduser().resolve()
    output_dir = root / f"da3-full-scene-{job_id}"
    if output_dir.exists() and (not output_dir.is_dir() or any(output_dir.iterdir())):
        raise ReconstructionValidationError("Full-scene DA3 output directory must be empty before start.")
    spec = DigitalTwinProcessingJobSpec(
        job_id=job_id,
        kind=DigitalTwinProcessingKind.DENSE_GEOMETRY,
        title="Build Full-Scene DA3",
        backend_label="Depth Anything 3",
        runtime_label="external Depth Anything 3",
        output_dir=output_dir,
        expected_protocol=DA3_FULL_SCENE_PROTOCOL,
        source_label=video.name,
        profile_label=active_profile.profile_label,
        cancellable=True,
    )
    args = (
        str(worker),
        "--job-id", job_id,
        "--video", str(video),
        "--output-dir", str(output_dir),
        "--da3-python", str(external_python),
        "--da3-repo", str(external_repo),
        "--model", active_profile.model_id,
        "--target-fps", str(active_profile.target_fps_hz),
        "--max-keyframes", str(active_profile.maximum_keyframes),
        "--minimum-spacing-seconds", str(active_profile.minimum_spacing_seconds),
        "--forced-spacing-seconds", str(active_profile.forced_spacing_seconds),
        "--minimum-visual-change", str(active_profile.minimum_visual_change),
        "--minimum-quality-score", str(active_profile.minimum_quality_score),
        "--process-res", str(active_profile.process_res),
        "--pixel-stride", str(active_profile.pixel_stride),
        "--per-window-timeout", str(active_profile.per_window_timeout_seconds),
    )
    command = DigitalTwinProcessingCommand(
        program=Path(sys.executable),
        arguments=args,
        working_directory=PROJECT_ROOT,
        environment_overrides=(("PYTHONUNBUFFERED", "1"),),
    )
    return spec, command


__all__ = [
    "DA3_FULL_SCENE_PROTOCOL",
    "DA3FullSceneProfile",
    "build_full_scene_da3_job",
    "estimate_overlap_similarity",
    "plan_overlapping_windows",
    "transform_chunk_to_global_gauge",
]
