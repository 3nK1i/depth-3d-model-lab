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
    spec = importlib.util.spec_from_file_location("run_i4_1_object_orbit_campaign", RUNNER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _producer_closure(*, scene_id: str = "pdrgb-i3-contract") -> dict:
    automated = {
        "phase": "I3.2",
        "automated_pass": True,
        "scene_id": scene_id,
        "scene_fingerprint_sha256": "a" * 64,
        "authority": {
            "metric_scale_claimed": False,
            "measurement_authoritative": False,
            "surface_mesh_generated": False,
        },
    }
    ui = {"phase": "I3.2-UI", "pass": True, "scene_id": scene_id}
    return build_i3_closure_record(automated, ui, fixed_view_review_pass=True)


def _write(tmp_path: Path, payload: dict) -> Path:
    path = tmp_path / "i3_2_closure.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def test_i4_1_accepts_exact_i3_2_producer_closure_schema(tmp_path: Path) -> None:
    runner = _load_runner()
    closure = _producer_closure()
    restored = runner._validate_i3_closure(_write(tmp_path, closure))
    assert restored == closure
    assert restored["decision"]["i3_closure_authorized"] is True
    assert restored["decision"]["i4_connector_orbit_authorized"] is True
    assert restored["decision"]["mesh_authorization"] is False


def test_i4_1_rejects_flattened_authorization_even_when_values_are_true(tmp_path: Path) -> None:
    runner = _load_runner()
    closure = _producer_closure()
    decision = closure.pop("decision")
    closure.update(decision)
    with pytest.raises(ReconstructionValidationError, match="nested I3 closure decision"):
        runner._validate_i3_closure(_write(tmp_path, closure))


def test_i4_1_rejects_incomplete_i3_final_review_contract(tmp_path: Path) -> None:
    runner = _load_runner()
    closure = _producer_closure()
    closure["checks"]["fixed_view_review"] = "pending"
    with pytest.raises(ReconstructionValidationError, match="fixed_view_review=pass"):
        runner._validate_i3_closure(_write(tmp_path, closure))


def test_i4_1_rejects_premature_metric_or_measurement_authority(tmp_path: Path) -> None:
    runner = _load_runner()
    for field in ("metric_scale_claimed", "measurement_authoritative", "surface_mesh_generated"):
        closure = _producer_closure(scene_id=f"scene-{field}")
        closure["authority"][field] = True
        with pytest.raises(ReconstructionValidationError, match="premature metric/measurement/mesh authority"):
            runner._validate_i3_closure(_write(tmp_path, closure))


def test_i4_1_rejects_mesh_pre_authorization_in_decision(tmp_path: Path) -> None:
    runner = _load_runner()
    closure = copy.deepcopy(_producer_closure())
    closure["decision"]["mesh_authorization"] = True
    with pytest.raises(ReconstructionValidationError, match="pre-authorized mesh"):
        runner._validate_i3_closure(_write(tmp_path, closure))
