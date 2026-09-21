from __future__ import annotations

from core.reconstruction.i5_failure_robustness import I511AFailureAdmissionPolicy, repository_failure_robustness_inventory


def test_repeated_policy_construction_is_deterministic() -> None:
    payloads = [I511AFailureAdmissionPolicy().as_dict() for _ in range(100)]
    assert all(item == payloads[0] for item in payloads)


def test_inventory_is_deterministic_and_does_not_probe_resources() -> None:
    payloads = [repository_failure_robustness_inventory() for _ in range(50)]
    assert all(item == payloads[0] for item in payloads)
    assert payloads[0]["inventory_is_static"] is True
    assert payloads[0]["resource_pressure_performed"] is False


def test_fault_scenario_set_stays_narrow_under_repetition() -> None:
    for _ in range(100):
        p = I511AFailureAdmissionPolicy()
        assert len(p.fault_scenarios) == 3
        assert "cancel" not in p.fault_scenarios
        assert "timeout" not in p.fault_scenarios


def test_real_resource_exhaustion_remains_disabled() -> None:
    for _ in range(100):
        p = I511AFailureAdmissionPolicy()
        assert p.actual_gpu_exhaustion is False
        assert p.actual_root_filesystem_fill is False
        assert p.actual_user_data_corruption is False
