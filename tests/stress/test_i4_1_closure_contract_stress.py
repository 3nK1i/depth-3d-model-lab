from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path

import pytest

from core.reconstruction.errors import ReconstructionValidationError
from core.reconstruction.offline_dense_i3_promotion import build_i3_closure_record


ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / "scripts" / "run_i4_1_object_orbit_campaign.py"


def _load_runner():
    spec = importlib.util.spec_from_file_location("run_i4_1_object_orbit_campaign_stress", RUNNER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _closure(index: int = 0) -> dict:
    scene_id = f"pdrgb-stress-{index:04d}"
    automated = {
        "phase": "I3.2",
        "automated_pass": True,
        "scene_id": scene_id,
        "scene_fingerprint_sha256": f"{index:064x}"[-64:],
        "authority": {
            "metric_scale_claimed": False,
            "measurement_authoritative": False,
            "surface_mesh_generated": False,
        },
    }
    ui = {"phase": "I3.2-UI", "pass": True, "scene_id": scene_id}
    return build_i3_closure_record(automated, ui, fixed_view_review_pass=True)


def _write(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")


def test_i4_1_valid_closure_validation_is_repeatable_and_stateless(tmp_path: Path) -> None:
    runner = _load_runner()
    path = tmp_path / "closure.json"
    payload = _closure(1)
    _write(path, payload)
    for _ in range(250):
        assert runner._validate_i3_closure(path)["decision"]["i4_connector_orbit_authorized"] is True


def test_i4_1_many_distinct_valid_producer_closures_are_accepted(tmp_path: Path) -> None:
    runner = _load_runner()
    path = tmp_path / "closure.json"
    for index in range(100):
        payload = _closure(index)
        _write(path, payload)
        restored = runner._validate_i3_closure(path)
        assert restored["scene_id"] == payload["scene_id"]


def test_i4_1_authorization_mutation_matrix_fails_closed(tmp_path: Path) -> None:
    runner = _load_runner()
    path = tmp_path / "closure.json"
    mutations = (
        (("decision", "i3_closure_authorized"), False),
        (("decision", "i4_connector_orbit_authorized"), False),
        (("decision", "mesh_authorization"), True),
        (("authority", "canonical_geometry_authoritative"), False),
        (("authority", "canonical_persistence_promoted"), False),
        (("authority", "metric_scale_claimed"), True),
        (("authority", "measurement_authoritative"), True),
        (("authority", "surface_mesh_generated"), True),
        (("checks", "automated_requalification"), "failed"),
        (("checks", "fixed_view_review"), "failed"),
        (("checks", "reconstruction_ui_sanity"), "failed"),
    )
    for index, (keys, value) in enumerate(mutations):
        payload = copy.deepcopy(_closure(index))
        payload[keys[0]][keys[1]] = value
        _write(path, payload)
        with pytest.raises(ReconstructionValidationError):
            runner._validate_i3_closure(path)


def test_i4_1_top_level_authorization_cannot_bypass_nested_denial(tmp_path: Path) -> None:
    runner = _load_runner()
    path = tmp_path / "closure.json"
    for index in range(50):
        payload = _closure(index)
        payload["i3_closure_authorized"] = True
        payload["i4_connector_orbit_authorized"] = True
        payload["mesh_authorization"] = False
        payload["decision"]["i4_connector_orbit_authorized"] = False
        _write(path, payload)
        with pytest.raises(ReconstructionValidationError, match="decision.i4_connector_orbit_authorized"):
            runner._validate_i3_closure(path)


def test_i4_1_malformed_or_partial_closure_files_fail_closed(tmp_path: Path) -> None:
    runner = _load_runner()
    path = tmp_path / "closure.json"
    malformed = (
        "",
        "[]",
        "{",
        "null",
        json.dumps({"phase": "I3", "status": "pass", "pass": True}),
    )
    for text in malformed * 20:
        path.write_text(text, encoding="utf-8")
        with pytest.raises(ReconstructionValidationError):
            runner._validate_i3_closure(path)
