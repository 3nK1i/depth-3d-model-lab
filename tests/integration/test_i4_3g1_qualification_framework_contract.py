from __future__ import annotations

from pathlib import Path

from core.reconstruction.cross_provider_qualification import build_i4_3g1_framework_snapshot
from tests.i4_3g1_helpers import passing_i4_3a_snapshot, passing_i4_3f3_snapshot

ROOT = Path(__file__).resolve().parents[2]


def test_g1_framework_is_execution_free_and_does_not_touch_provider_workers() -> None:
    source = (ROOT / "core/reconstruction/cross_provider_qualification.py").read_text(encoding="utf-8")
    forbidden = ("subprocess.run(", "Popen(", ".run_provider(", "execute_dense_provider(")
    assert all(token not in source for token in forbidden)


def test_g1_binding_script_accepts_architecture_and_f3_predecessors() -> None:
    source = (ROOT / "scripts/run_i4_3g1_qualification_framework.py").read_text(encoding="utf-8")
    assert "--i4-3a-snapshot" in source
    assert "--i4-3f3-snapshot" in source
    assert "i4_3g1_qualification_framework_snapshot.json" in source


def test_g1_snapshot_freezes_fairness_modes_without_selecting_provider() -> None:
    snapshot = build_i4_3g1_framework_snapshot(
        i4_3a_snapshot=passing_i4_3a_snapshot(), i4_3f3_snapshot=passing_i4_3f3_snapshot()
    )
    assert snapshot["fairness"]["production_native_comparison_primary"] is True
    assert snapshot["fairness"]["matched_view_diagnostic_secondary"] is True
    assert snapshot["fairness"]["historical_evidence_never_reported_as_new_execution"] is True
    assert snapshot["authority"]["provider_selected"] is False
    assert snapshot["authority"]["provider_promoted"] is False


def test_g1_complete_room_temple_house_low_texture_weak_overlap_are_all_frozen() -> None:
    snapshot = build_i4_3g1_framework_snapshot(
        i4_3a_snapshot=passing_i4_3a_snapshot(), i4_3f3_snapshot=passing_i4_3f3_snapshot()
    )
    ids = {item["scenario_id"] for item in snapshot["scenario_matrix"]}
    assert {
        "g2_complete_room_mvs_da3",
        "g2_temple_ring_mvs_da3",
        "g2_house_mvs_da3_historical",
        "g2_low_texture_mvs_da3",
        "g2_weak_overlap_mvs_da3",
    } <= ids


def test_g1_next_step_is_g2_not_auto_or_hybrid() -> None:
    snapshot = build_i4_3g1_framework_snapshot(
        i4_3a_snapshot=passing_i4_3a_snapshot(), i4_3f3_snapshot=passing_i4_3f3_snapshot()
    )
    assert snapshot["next"].startswith("I4.3G2")
    assert snapshot["decision"]["auto_policy_implemented"] is False
    assert snapshot["decision"]["hybrid_authorized"] is False
