from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import pytest

from core.reconstruction.reconstruction_progressive_sfm_contract import ProgressiveSfmSnapshot

ROOT = Path(__file__).resolve().parents[2]
HELPER_PATH = ROOT / "scripts" / "reconstruction_progressive_sfm_emission.py"


def _load_helper():
    spec = importlib.util.spec_from_file_location("i6_6b_progressive_emission", HELPER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class _Rigid:
    def __init__(self, tx: float):
        self._matrix = [
            [1.0, 0.0, 0.0, tx],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
        ]

    def matrix(self):
        return self._matrix


class _Image:
    def __init__(self, image_id: int):
        self.camera_id = 7
        self.name = f"image_{image_id:04d}.jpg"
        self._rigid = _Rigid(float(image_id))

    def cam_from_world(self):
        return self._rigid


class _Track:
    def __init__(self, length: int):
        self.elements = [object() for _ in range(length)]


class _Point:
    def __init__(self, point_id: int):
        self.xyz = (float(point_id), 0.5, -0.25)
        self.color = (10, 20, 30)
        self.track = _Track(3)


class _Reconstruction:
    def __init__(self, *, cameras: int = 6, points: int = 20):
        self._image_ids = tuple(range(1, cameras + 1))
        self._point_ids = tuple(range(100, 100 + points))

    def reg_image_ids(self):
        return self._image_ids

    def point3D_ids(self):
        return self._point_ids

    def image(self, image_id: int):
        return _Image(image_id)

    def point3D(self, point_id: int):
        return _Point(point_id)

    def num_reg_images(self):
        return len(self._image_ids)

    def num_points3D(self):
        return len(self._point_ids)

    def compute_num_observations(self):
        return len(self._point_ids) * 3

    def compute_mean_reprojection_error(self):
        return 0.75


def test_mapping_snapshot_interval_targets_bounded_number_of_native_snapshots() -> None:
    helper = _load_helper()
    assert helper.mapping_snapshot_frame_interval(3) == 2
    assert helper.mapping_snapshot_frame_interval(12) == 2
    assert helper.mapping_snapshot_frame_interval(128) == 11
    assert helper.mapping_snapshot_frame_interval(500) == 42


def test_mapping_snapshot_interval_rejects_invalid_input_count() -> None:
    helper = _load_helper()
    with pytest.raises(ValueError):
        helper.mapping_snapshot_frame_interval(2)


def test_even_sample_ids_is_deterministic_bounded_and_keeps_endpoints() -> None:
    helper = _load_helper()
    values = list(range(100))
    sampled = helper.evenly_sample_ids(values, 8)
    assert len(sampled) == 8
    assert sampled[0] == 0
    assert sampled[-1] == 99
    assert sampled == helper.evenly_sample_ids(reversed(values), 8)
    assert len(sampled) == len(set(sampled))


def test_even_sample_ids_returns_all_values_when_under_limit() -> None:
    helper = _load_helper()
    assert helper.evenly_sample_ids([4, 1, 3], 8) == (1, 3, 4)


def test_even_sample_ids_rejects_nonpositive_limit() -> None:
    helper = _load_helper()
    with pytest.raises(ValueError):
        helper.evenly_sample_ids([1, 2, 3], 0)


def test_build_payload_roundtrips_through_i66a_contract_and_inverts_pose() -> None:
    helper = _load_helper()
    payload = helper.build_progressive_sfm_snapshot_payload(
        _Reconstruction(cameras=6, points=20),
        session_id="11111111-2222-4333-8444-555555555555",
        request_fingerprint_sha256="a" * 64,
        revision=3,
        timestamp_utc="2026-09-17T13:00:00+00:00",
        elapsed_ns=123,
        stage="mapping",
        stage_detail="registered_image_snapshot",
        input_views=12,
        progress_fraction=0.74,
        max_camera_samples=4,
        max_landmark_samples=5,
    )
    snapshot = ProgressiveSfmSnapshot.from_payload(payload)
    assert snapshot.revision == 3
    assert snapshot.statistics.registered_cameras == 6
    assert snapshot.statistics.sparse_landmarks == 20
    assert len(snapshot.geometry.camera_samples) == 4
    assert len(snapshot.geometry.landmark_samples) == 5
    # camera_from_reconstruction has +1 X translation, so inverse camera centre is -1.
    assert snapshot.geometry.camera_samples[0].T_reconstruction_from_camera[0, 3] == pytest.approx(-1.0)
    assert snapshot.provisional is True
    assert snapshot.final_result_admitted is False


def test_payload_sampling_is_deterministic_and_json_safe() -> None:
    helper = _load_helper()
    kwargs = dict(
        session_id="11111111-2222-4333-8444-555555555555",
        request_fingerprint_sha256="b" * 64,
        revision=1,
        timestamp_utc="2026-09-17T13:00:00+00:00",
        elapsed_ns=1,
        stage="mapping",
        stage_detail="registered_image_snapshot",
        input_views=50,
        progress_fraction=0.7,
        max_camera_samples=3,
        max_landmark_samples=7,
    )
    first = helper.build_progressive_sfm_snapshot_payload(_Reconstruction(cameras=10, points=100), **kwargs)
    second = helper.build_progressive_sfm_snapshot_payload(_Reconstruction(cameras=10, points=100), **kwargs)
    assert first == second
    json.dumps(first, allow_nan=False)


def test_payload_never_claims_final_canonical_or_metric_authority() -> None:
    helper = _load_helper()
    payload = helper.build_progressive_sfm_snapshot_payload(
        _Reconstruction(),
        session_id="11111111-2222-4333-8444-555555555555",
        request_fingerprint_sha256="c" * 64,
        revision=1,
        timestamp_utc="2026-09-17T13:00:00+00:00",
        elapsed_ns=1,
        stage="finalizing",
        stage_detail="selected_model_before_final_result_admission",
        input_views=12,
        progress_fraction=0.94,
        candidate_model_id="0",
        candidate_model_count=2,
    )
    authority = payload["authority"]
    assert authority["provisional"] is True
    assert authority["final_result_admitted"] is False
    assert authority["persistent_canonical_scene"] is False
    assert authority["canonical_geometry_authoritative"] is False
    assert authority["metric_scale_claimed"] is False
    assert authority["measurement_authoritative"] is False
    assert authority["certified_metrology"] is False


def test_throttle_accepts_new_token_only_after_time_window_unless_forced() -> None:
    helper = _load_helper()
    throttle = helper.ProgressiveSnapshotThrottle(min_interval_s=0.75)
    assert throttle.should_emit("a", now_s=10.0) is True
    assert throttle.should_emit("a", now_s=11.0) is False
    assert throttle.should_emit("b", now_s=10.4) is False
    assert throttle.should_emit("c", now_s=10.75) is True
    assert throttle.should_emit("d", now_s=10.80, force=True) is True


def test_progressive_event_line_uses_distinct_prefix_and_is_json() -> None:
    helper = _load_helper()
    payload = {"schema_version": 1, "revision": 2}
    line = helper.progressive_event_line(payload)
    assert line.startswith("DEPTH3D_SFM_PROGRESSIVE ")
    restored = json.loads(line[len("DEPTH3D_SFM_PROGRESSIVE "):])
    assert restored == payload
