from __future__ import annotations

import json

from core.reconstruction.i5_rolling_depth_robustness_replay import (
    I57B_SCENARIO_IDS, build_i5_7b_recovered_control_record,
    build_i5_7b_rolling_depth_robustness_replay,
)
from tests.i5_7b_helpers import (
    build_args, provider_summary, recovery_artifact_bindings, recovery_provenance, scenario_record,
)


def _canonical(payload: dict) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)


def test_replay_builder_is_deterministic_across_repeated_runs():
    first = _canonical(build_i5_7b_rolling_depth_robustness_replay(**build_args()))
    for _ in range(20):
        assert _canonical(build_i5_7b_rolling_depth_robustness_replay(**build_args())) == first


def test_every_single_stress_rejection_remains_degraded_not_retry_or_global_failure():
    for sid in I57B_SCENARIO_IDS[1:]:
        args = build_args()
        args["scenario_records"][sid] = scenario_record(sid, "rejected")
        out = build_i5_7b_rolling_depth_robustness_replay(**args)
        assert out["quality_status"] == "degraded"
        assert out["pass"] is True
        assert out["scope"]["retry_until_pass"] is False
        assert out["decision"]["i5_7c_authorized"] is True


def test_control_rejection_dominates_quality_status_without_provider_promotion():
    for stress in ("pass", "degraded", "rejected"):
        args = build_args()
        args["scenario_records"]["castle_control"] = scenario_record("castle_control", "rejected")
        args["scenario_records"]["castle_dynamic_proxy"] = scenario_record("castle_dynamic_proxy", stress)
        out = build_i5_7b_rolling_depth_robustness_replay(**args)
        assert out["quality_status"] == "rejected"
        assert out["authority"]["provider_promoted"] is False
        assert out["decision"]["universal_provider_winner_declared"] is False


def test_recovered_control_keeps_missing_performance_ratios_null_without_blocking_closure():
    args = build_args()
    args["scenario_records"]["castle_control"] = build_i5_7b_recovered_control_record(
        provider_summary=provider_summary("pass", views=19),
        artifact_bindings=recovery_artifact_bindings(),
        recovery_provenance=recovery_provenance(),
    )
    out = build_i5_7b_rolling_depth_robustness_replay(**args)
    assert out["pass"] is True
    assert out["decision"]["i5_7c_authorized"] is True
    for sid in I57B_SCENARIO_IDS:
        obs = out["control_relative_observations"][sid]
        assert obs["wall_time_ratio_to_control"] is None
        assert obs["peak_vram_ratio_to_control"] is None
        assert obs["peak_system_ram_ratio_to_control"] is None
    assert out["scope"]["retry_until_pass"] is False
