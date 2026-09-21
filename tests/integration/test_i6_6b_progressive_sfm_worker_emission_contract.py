from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

from core.reconstruction.reconstruction_progressive_sfm_contract import (
    ProgressiveSfmSnapshot,
    validate_progressive_sfm_snapshot_transition,
)

ROOT = Path(__file__).resolve().parents[2]
HELPER_PATH = ROOT / "scripts" / "reconstruction_progressive_sfm_emission.py"


def _load_helper():
    spec = importlib.util.spec_from_file_location("i6_6b_progressive_integration", HELPER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class _Rigid:
    def __init__(self, tx: float):
        self._matrix = (
            (1.0, 0.0, 0.0, tx),
            (0.0, 1.0, 0.0, 0.0),
            (0.0, 0.0, 1.0, 0.0),
        )
    def matrix(self): return self._matrix


class _Image:
    def __init__(self, image_id: int):
        self.camera_id = 1
        self.name = f"frame_{image_id:04d}.png"
        self._pose = _Rigid(float(image_id))
    def cam_from_world(self): return self._pose


class _Track:
    def __init__(self, n: int): self.elements = tuple(range(n))


class _Point:
    def __init__(self, point_id: int):
        self.xyz = (float(point_id), 1.0, 2.0)
        self.color = (1, 2, 3)
        self.track = _Track(4)


class _Rec:
    def __init__(self, cameras: int, points: int):
        self._cameras = cameras
        self._points = points
    def reg_image_ids(self): return tuple(range(1, self._cameras + 1))
    def point3D_ids(self): return tuple(range(1, self._points + 1))
    def image(self, image_id): return _Image(int(image_id))
    def point3D(self, point_id): return _Point(int(point_id))
    def num_reg_images(self): return self._cameras
    def num_points3D(self): return self._points
    def compute_num_observations(self): return self._points * 4
    def compute_mean_reprojection_error(self): return 0.5


def _payload(helper, revision: int, cameras: int, points: int, elapsed_ns: int):
    return helper.build_progressive_sfm_snapshot_payload(
        _Rec(cameras, points),
        session_id="11111111-2222-4333-8444-555555555555",
        request_fingerprint_sha256="d" * 64,
        revision=revision,
        timestamp_utc=f"2026-09-17T13:00:{revision:02d}+00:00",
        elapsed_ns=elapsed_ns,
        stage="mapping",
        stage_detail="registered_image_snapshot",
        input_views=128,
        progress_fraction=min(0.90, 0.68 + revision * 0.01),
        max_camera_samples=64,
        max_landmark_samples=256,
    )


def test_worker_payload_series_obeys_i66a_transition_contract_with_revision_gaps() -> None:
    helper = _load_helper()
    first = ProgressiveSfmSnapshot.from_payload(_payload(helper, 1, 12, 100, 10))
    second = ProgressiveSfmSnapshot.from_payload(_payload(helper, 4, 18, 160, 20))
    validate_progressive_sfm_snapshot_transition(first, second)


def test_worker_payload_bounded_geometry_never_exceeds_policy_caps() -> None:
    helper = _load_helper()
    snapshot = ProgressiveSfmSnapshot.from_payload(_payload(helper, 1, 120, 5000, 10))
    assert len(snapshot.geometry.camera_samples) <= 64
    assert len(snapshot.geometry.landmark_samples) <= 256
    assert len(snapshot.geometry.camera_samples) <= snapshot.statistics.registered_cameras
    assert len(snapshot.geometry.landmark_samples) <= snapshot.statistics.sparse_landmarks


def test_mapping_counts_are_allowed_to_decrease_across_candidate_models() -> None:
    helper = _load_helper()
    first = ProgressiveSfmSnapshot.from_payload(_payload(helper, 1, 30, 800, 10))
    second = ProgressiveSfmSnapshot.from_payload(_payload(helper, 2, 8, 120, 20))
    validate_progressive_sfm_snapshot_transition(first, second)
