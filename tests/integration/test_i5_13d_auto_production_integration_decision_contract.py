from __future__ import annotations

from pathlib import Path

from tests.i5_13d_helpers import make_binding, make_i5_13c_qualification, make_i5_13d_receipt
from core.reconstruction.i5_auto_policy_production_qualification import (
    build_i5_13d_production_integration_decision,
    inspect_i5_13d_production_integration_surface,
)


def _build():
    root = Path(__file__).resolve().parents[2]
    inventory = inspect_i5_13d_production_integration_surface(root)
    return build_i5_13d_production_integration_decision(
        i5_13c_qualification=make_i5_13c_qualification(),
        i5_13c_binding=make_binding("c.json", "1"),
        post_i5_13c_checkpoint_binding=make_binding("post_c.tar.gz", "2"),
        integration_inventory=inventory,
        integration_anchor_bindings={
            "dense_provider_architecture": make_binding("dense_provider_architecture.py", "3"),
            "adaptive_production_planning": make_binding("adaptive_production_planning.py", "4"),
            "reconstruction_page": make_binding("reconstruction_page.py", "5"),
            "i5_13b_engine_module": make_binding("i5_auto_policy_engine.py", "6"),
        },
        decision_receipt=make_i5_13d_receipt(),
        decision_receipt_binding=make_binding("receipt.json", "7"),
    )


def test_integration_contract_closes_i5_13_without_production_auto():
    out = _build()
    assert out["status"] == "auto_policy_production_integration_decision_complete"
    assert out["qualification"]["offline_replay_qualified"] is True
    assert out["qualification"]["production_auto_qualified"] is False
    assert out["decision"]["i5_13_complete"] is True
    assert out["decision"]["production_auto_authorized"] is False


def test_integration_contract_preserves_explicit_production_path():
    out = _build()
    inv = out["production_integration_inventory"]
    assert inv["adaptive_production_planning_mode"] == "explicit_provider_only"
    assert inv["production_ui_provider_selection_mode"] == "inspection_only"
    assert out["scope"]["production_auto_integration"] is False


def test_integration_contract_authorizes_final_i5_review():
    out = _build()
    assert out["decision"]["i5_14_authorized"] is True
    assert out["next"].startswith("I5.14")
