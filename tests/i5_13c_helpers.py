from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

from tests.i5_13b_helpers import admission as make_i5_13a_admission, receipt as make_i5_13b_receipt
from core.reconstruction.i5_auto_policy_engine import build_i5_13b_engine_freeze


def make_binding(name: str, ch: str = "a") -> dict[str, str]:
    return {"path": f"/tmp/{name}", "sha256": ch * 64}


def make_i5_13b_freeze() -> dict[str, Any]:
    a = make_i5_13a_admission()
    receipt = make_i5_13b_receipt()
    return build_i5_13b_engine_freeze(
        i5_13a_admission=a,
        i5_13a_binding=make_binding("i5_13a.json", "1"),
        post_i5_13a_checkpoint_binding=make_binding("post_i5_13a.tar.gz", "2"),
        engine_module_binding=make_binding("i5_auto_policy_engine.py", "3"),
        engine_receipt=receipt,
        engine_receipt_binding=make_binding("i5_13b_receipt.json", "4"),
    )


def make_i5_13c_receipt() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "phase": "I5.13C_REPLAY_RECEIPT",
        "status": "pass",
        "pass": True,
        "scope": {
            "focused_i5_13c_tests": True,
            "i5_13b_regression": True,
            "dense_selection_contract_regression": True,
            "deterministic_replay_stress": True,
            "import_probe": True,
            "runner_help_probe": True,
            "provider_execution": False,
            "sfm_execution": False,
            "mvs_execution": False,
            "scenario_rematerialization": False,
            "production_auto_integration": False,
            "production_default_modified": False,
            "provider_promoted": False,
        },
        "counts": {
            "focused_tests": 1,
            "predecessor_unit_tests": 1,
            "predecessor_integration_tests": 1,
            "predecessor_stress_tests": 1,
            "predecessor_total_tests": 3,
            "combined_exercised_tests": 4,
        },
        "return_codes": {
            "focused": 0,
            "predecessor_unit": 0,
            "predecessor_integration": 0,
            "predecessor_stress": 0,
            "import_probe": 0,
            "runner_help_probe": 0,
        },
    }


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
