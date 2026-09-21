#!/usr/bin/env python3
"""Dependency-isolated I6.6B progressive sparse-SfM emission helpers.

The production PyCOLMAP worker imports this sibling module under the external
reconstruction interpreter.  Keep it stdlib-only: no application/core imports,
Qt, OpenCV, Torch, NumPy, or PyCOLMAP dependency is allowed here.

The helper mirrors the frozen I6.6A JSON contract so the worker can emit
provisional visualization payloads without importing the application package.
The application-side I6.6A parser remains the authority that validates these
payloads when I6.6C begins consuming them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import math
from typing import Iterable

PROGRESSIVE_SFM_EVENT_PREFIX = "EIA_SFM_PROGRESSIVE "
PROGRESSIVE_SFM_SCHEMA_VERSION = 1
PROGRESSIVE_SFM_PHASE = "I6.6A_PROGRESSIVE_SFM_SNAPSHOT_AND_AUTHORITY_CONTRACT"
PROGRESSIVE_SFM_BACKEND_ID = "pycolmap-offline-sfm"
PROGRESSIVE_SFM_BACKEND_VERSION = "1"
PROGRESSIVE_SFM_CAMERA_AUTHORITY_OWNER = "PyCOLMAP"
PROGRESSIVE_SFM_POSE_CONVENTION = "T_reconstruction_from_camera"
PROGRESSIVE_SFM_SCALE_STATUS = "arbitrary"
PROGRESSIVE_SFM_ALIGNMENT_STATUS = "arbitrary"
PROGRESSIVE_SFM_SNAPSHOT_AUTHORITY = "provisional_visualization_only"
PROGRESSIVE_SFM_SAMPLING_PROFILE = "bounded_preview_v1"

# Operational visualization/transport policy only.  These do not alter
# feature extraction, matching, mapper, BA, model selection, or admission.
PROGRESSIVE_TARGET_MAPPING_SNAPSHOTS = 12
PROGRESSIVE_MIN_NATIVE_FRAME_INTERVAL = 2
PROGRESSIVE_MIN_EMIT_INTERVAL_S = 0.75
PROGRESSIVE_MAX_CAMERA_SAMPLES = 64
PROGRESSIVE_MAX_LANDMARK_SAMPLES = 4096


def mapping_snapshot_frame_interval(input_views: int) -> int:
    """Choose a bounded native snapshot cadence targeting about 12 updates.

    PyCOLMAP already registers the initial pair before its per-image snapshot
    cadence begins, hence the ``input_views - 2`` term.  This cadence controls
    only temporary visualization snapshots written by PyCOLMAP.
    """

    if isinstance(input_views, bool) or int(input_views) != input_views:
        raise ValueError("input_views must be an integer.")
    count = int(input_views)
    if count < 3:
        raise ValueError("Progressive offline SfM requires at least three input views.")
    remaining = max(1, count - 2)
    return max(
        PROGRESSIVE_MIN_NATIVE_FRAME_INTERVAL,
        int(math.ceil(remaining / PROGRESSIVE_TARGET_MAPPING_SNAPSHOTS)),
    )


def evenly_sample_ids(values: Iterable[int], limit: int) -> tuple[int, ...]:
    """Return deterministic bounded IDs while preserving range endpoints."""

    if isinstance(limit, bool) or int(limit) != limit or int(limit) < 1:
        raise ValueError("limit must be a positive integer.")
    maximum = int(limit)
    ordered = tuple(sorted({int(value) for value in values}))
    if len(ordered) <= maximum:
        return ordered
    if maximum == 1:
        return (ordered[0],)
    last = len(ordered) - 1
    indices = tuple((index * last) // (maximum - 1) for index in range(maximum))
    sampled = tuple(ordered[index] for index in indices)
    if len(sampled) != len(set(sampled)):
        raise RuntimeError("Deterministic bounded sampling produced duplicate IDs.")
    return sampled


def _finite_or_none(value) -> float | None:
    number = float(value)
    return number if math.isfinite(number) else None


def _matrix34(value) -> tuple[tuple[float, float, float, float], ...]:
    matrix = value.matrix()
    rows = []
    for row in range(3):
        values = tuple(float(matrix[row][col]) for col in range(4))
        if not all(math.isfinite(item) for item in values):
            raise ValueError("PyCOLMAP pose contains a non-finite value.")
        rows.append(values)
    return tuple(rows)


def _invert_camera_from_reconstruction(value) -> list[list[float]]:
    """Convert PyCOLMAP camera-from-world 3x4 to camera-to-reconstruction 4x4."""

    rows = _matrix34(value)
    rotation = tuple(tuple(rows[r][c] for c in range(3)) for r in range(3))
    translation = tuple(rows[r][3] for r in range(3))
    rotation_t = tuple(tuple(rotation[c][r] for c in range(3)) for r in range(3))
    inverse_translation = tuple(
        -sum(rotation_t[r][c] * translation[c] for c in range(3))
        for r in range(3)
    )
    return [
        [rotation_t[0][0], rotation_t[0][1], rotation_t[0][2], inverse_translation[0]],
        [rotation_t[1][0], rotation_t[1][1], rotation_t[1][2], inverse_translation[1]],
        [rotation_t[2][0], rotation_t[2][1], rotation_t[2][2], inverse_translation[2]],
        [0.0, 0.0, 0.0, 1.0],
    ]


def _authority_payload() -> dict[str, object]:
    return {
        "snapshot_authority": PROGRESSIVE_SFM_SNAPSHOT_AUTHORITY,
        "camera_pose_authority_owner": PROGRESSIVE_SFM_CAMERA_AUTHORITY_OWNER,
        "pose_convention": PROGRESSIVE_SFM_POSE_CONVENTION,
        "scale_status": PROGRESSIVE_SFM_SCALE_STATUS,
        "alignment_status": PROGRESSIVE_SFM_ALIGNMENT_STATUS,
        "provisional": True,
        "final_result_admitted": False,
        "persistent_canonical_scene": False,
        "canonical_geometry_authoritative": False,
        "metric_scale_claimed": False,
        "measurement_authoritative": False,
        "certified_metrology": False,
    }


def _canonical_sha256(payload: dict[str, object]) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def build_progressive_sfm_snapshot_payload(
    reconstruction,
    *,
    session_id: str,
    request_fingerprint_sha256: str,
    revision: int,
    timestamp_utc: str,
    elapsed_ns: int,
    stage: str,
    stage_detail: str,
    input_views: int,
    progress_fraction: float | None,
    candidate_model_id: str = "",
    candidate_model_count: int | None = None,
    warnings: Iterable[str] = (),
    max_camera_samples: int = PROGRESSIVE_MAX_CAMERA_SAMPLES,
    max_landmark_samples: int = PROGRESSIVE_MAX_LANDMARK_SAMPLES,
) -> dict[str, object]:
    """Convert a current PyCOLMAP-like reconstruction into an I6.6A payload."""

    image_ids = tuple(int(value) for value in reconstruction.reg_image_ids())
    point_ids = tuple(int(value) for value in reconstruction.point3D_ids())
    registered_cameras = int(reconstruction.num_reg_images())
    sparse_landmarks = int(reconstruction.num_points3D())
    observations = int(reconstruction.compute_num_observations())
    if sparse_landmarks == 0 and observations != 0:
        raise ValueError("PyCOLMAP reported observations without sparse landmarks.")

    sampled_image_ids = evenly_sample_ids(image_ids, max_camera_samples)
    sampled_point_ids = evenly_sample_ids(point_ids, max_landmark_samples)

    camera_samples: list[dict[str, object]] = []
    for image_id in sampled_image_ids:
        image = reconstruction.image(image_id)
        camera_samples.append(
            {
                "backend_image_id": image_id,
                "backend_camera_id": int(image.camera_id),
                "source_image_name": str(image.name),
                "T_reconstruction_from_camera": _invert_camera_from_reconstruction(
                    image.cam_from_world()
                ),
            }
        )

    landmark_samples: list[dict[str, object]] = []
    for point_id in sampled_point_ids:
        point = reconstruction.point3D(point_id)
        xyz = tuple(float(value) for value in point.xyz)
        if len(xyz) != 3 or not all(math.isfinite(value) for value in xyz):
            raise ValueError(f"Sparse point {point_id} contains invalid coordinates.")
        raw_color = tuple(int(value) for value in point.color)
        if len(raw_color) != 3 or any(value < 0 or value > 255 for value in raw_color):
            raise ValueError(f"Sparse point {point_id} contains invalid RGB values.")
        track_length = len(point.track.elements)
        landmark_samples.append(
            {
                "backend_point3d_id": point_id,
                "position_reconstruction": list(xyz),
                "color_rgb": list(raw_color),
                "track_length": int(track_length),
            }
        )

    geometry: dict[str, object] | None = None
    if camera_samples or landmark_samples:
        geometry = {
            "sampling_profile": PROGRESSIVE_SFM_SAMPLING_PROFILE,
            "camera_samples": camera_samples,
            "landmark_samples": landmark_samples,
        }

    warning_values = [str(value).strip() for value in warnings if str(value).strip()]
    warning_values = list(dict.fromkeys(warning_values))

    identity: dict[str, object] = {
        "schema_version": PROGRESSIVE_SFM_SCHEMA_VERSION,
        "phase": PROGRESSIVE_SFM_PHASE,
        "session_id": str(session_id),
        "revision": int(revision),
        "timestamp_utc": str(timestamp_utc),
        "elapsed_ns": int(elapsed_ns),
        "stage": str(stage),
        "stage_detail": str(stage_detail),
        "backend_id": PROGRESSIVE_SFM_BACKEND_ID,
        "backend_version": PROGRESSIVE_SFM_BACKEND_VERSION,
        "request_fingerprint_sha256": str(request_fingerprint_sha256),
        "statistics": {
            "input_views": int(input_views),
            "registered_cameras": registered_cameras,
            "sparse_landmarks": sparse_landmarks,
            "observations": observations,
            "mean_reprojection_error_px": _finite_or_none(
                reconstruction.compute_mean_reprojection_error()
            ),
        },
        "progress_fraction": (
            None if progress_fraction is None else float(progress_fraction)
        ),
        "candidate_model_id": str(candidate_model_id),
        "candidate_model_count": (
            None if candidate_model_count is None else int(candidate_model_count)
        ),
        "geometry": geometry,
        "warnings": warning_values,
        "authority": _authority_payload(),
    }
    payload = dict(identity)
    payload["snapshot_sha256"] = _canonical_sha256(identity)
    return payload


def progressive_event_line(payload: dict[str, object]) -> str:
    return PROGRESSIVE_SFM_EVENT_PREFIX + json.dumps(
        payload, sort_keys=True, allow_nan=False, separators=(",", ":")
    )


@dataclass(slots=True)
class ProgressiveSnapshotThrottle:
    """Deduplicate native snapshots and enforce a minimum worker emit interval."""

    min_interval_s: float = PROGRESSIVE_MIN_EMIT_INTERVAL_S
    _last_emit_s: float | None = None
    _seen_tokens: set[str] = field(default_factory=set)

    def __post_init__(self) -> None:
        self.min_interval_s = float(self.min_interval_s)
        if not math.isfinite(self.min_interval_s) or self.min_interval_s < 0.0:
            raise ValueError("min_interval_s must be finite and non-negative.")

    def should_emit(self, token: str, *, now_s: float, force: bool = False) -> bool:
        token_value = str(token).strip()
        if not token_value:
            raise ValueError("snapshot token must not be empty.")
        now = float(now_s)
        if not math.isfinite(now):
            raise ValueError("now_s must be finite.")
        if token_value in self._seen_tokens:
            return False
        self._seen_tokens.add(token_value)
        if not force and self._last_emit_s is not None:
            if now - self._last_emit_s < self.min_interval_s:
                return False
        self._last_emit_s = now
        return True
