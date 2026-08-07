from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[2]
WORKER_PATH = ROOT / "scripts" / "reconstruction_pycolmap_worker.py"


def _load_worker_module():
    spec = importlib.util.spec_from_file_location("phase8_3d_worker_test_module", WORKER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class _FakeReconstruction:
    def __init__(self, *, registered: int, points: int, observations: int, error: float):
        self._registered = registered
        self._points = points
        self._observations = observations
        self._error = error

    def num_reg_images(self):
        return self._registered

    def num_points3D(self):
        return self._points

    def compute_num_observations(self):
        return self._observations

    def compute_mean_reprojection_error(self):
        return self._error


def test_worker_model_selection_prefers_registration_then_observations_then_points() -> None:
    worker = _load_worker_module()
    models = {
        3: _FakeReconstruction(registered=8, points=1000, observations=1500, error=0.2),
        1: _FakeReconstruction(registered=9, points=500, observations=1000, error=1.2),
        2: _FakeReconstruction(registered=9, points=800, observations=1200, error=1.0),
    }
    selected_id, _, summaries = worker.select_best_model(models)
    assert selected_id == 2
    assert [item["model_id"] for item in summaries] == [2, 1, 3]


def test_worker_model_selection_is_deterministic_on_complete_tie() -> None:
    worker = _load_worker_module()
    models = {
        9: _FakeReconstruction(registered=4, points=100, observations=250, error=0.5),
        2: _FakeReconstruction(registered=4, points=100, observations=250, error=0.5),
    }
    selected_id, _, _ = worker.select_best_model(models)
    assert selected_id == 2


def test_resume_checkpoint_reuses_only_matching_fingerprint_and_database(tmp_path) -> None:
    worker = _load_worker_module()
    checkpoint = tmp_path / "checkpoint.json"
    database = tmp_path / "database.db"
    database.write_bytes(b"db")
    fingerprint = "a" * 64

    worker.write_stage_checkpoint(
        checkpoint,
        request_fingerprint=fingerprint,
        pycolmap_version="4.1.1",
        image_count=12,
        completed_stage="matching",
    )
    payload = worker.load_resume_checkpoint(checkpoint)
    assert worker.resolve_resume_stage(
        payload,
        request_fingerprint=fingerprint,
        pycolmap_version="4.1.1",
        database_path=database,
        image_count=12,
    ) == "matching"
    assert worker.resolve_resume_stage(
        payload,
        request_fingerprint="b" * 64,
        pycolmap_version="4.1.1",
        database_path=database,
        image_count=12,
    ) == "none"
    assert worker.resolve_resume_stage(
        payload,
        request_fingerprint=fingerprint,
        pycolmap_version="4.0.0",
        database_path=database,
        image_count=12,
    ) == "none"


def test_complete_checkpoint_reuses_matching_database_but_reruns_mapping(tmp_path) -> None:
    worker = _load_worker_module()
    database = tmp_path / "database.db"
    database.write_bytes(b"db")
    payload = {
        "schema_version": worker.SFM_WORKER_CHECKPOINT_SCHEMA_VERSION,
        "request_fingerprint": "c" * 64,
        "pycolmap_version": "4.1.1",
        "image_count": 5,
        "completed_stage": "complete",
    }
    assert worker.resolve_resume_stage(
        payload,
        request_fingerprint="c" * 64,
        pycolmap_version="4.1.1",
        database_path=database,
        image_count=5,
    ) == "matching"


def test_atomic_worker_checkpoint_never_leaves_temporary_file(tmp_path) -> None:
    worker = _load_worker_module()
    path = tmp_path / "stage.json"
    worker.atomic_write_object(path, {"ok": True})
    assert json.loads(path.read_text()) == {"ok": True}
    assert not path.with_name(path.name + ".tmp").exists()
