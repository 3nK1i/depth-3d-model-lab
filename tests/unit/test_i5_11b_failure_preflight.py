from __future__ import annotations

from pathlib import Path
import json

import pytest

from core.reconstruction.errors import ReconstructionValidationError
from core.reconstruction.i5_failure_robustness_preflight import (
    I511B_FAULT_SCENARIOS,
    I511B_MIN_FREE_BYTES_FOR_C,
    I511B_PHASE,
    build_i5_11b_preflight,
    discover_live_i5_11_fault_workers,
    observe_disk_space,
    probe_failure_classifier_without_fault_execution,
    validate_i5_11a_for_i5_11b,
)
from tests.i5_11b_helpers import (
    I511A_SHA,
    POST_I510D_DELTA_SHA,
    binding,
    mutate,
    synthetic_anchors,
    synthetic_classifier_probe,
    synthetic_disk_observation,
    synthetic_i511a_admission,
    synthetic_i511b_receipt,
)


def _build(**overrides) -> dict:
    args = {
        "i5_11a_admission": synthetic_i511a_admission(),
        "i5_11a_binding": binding("/evidence/i5_11a.json", I511A_SHA),
        "post_i5_10d_delta_binding": binding("/checkpoints/post_i5_10d.tar.gz", POST_I510D_DELTA_SHA),
        "preflight_receipt": synthetic_i511b_receipt(),
        "preflight_receipt_binding": binding("/evidence/i5_11b_receipt.json", "b" * 64),
        "repository_anchors": synthetic_anchors(),
        "classifier_probe": synthetic_classifier_probe(),
        "disk_observation": synthetic_disk_observation(),
        "live_fault_workers": [],
    }
    args.update(overrides)
    return build_i5_11b_preflight(**args)


def test_phase_and_fault_order_are_frozen() -> None:
    assert I511B_PHASE == "I5.11B"
    assert I511B_FAULT_SCENARIOS == ("cuda_oom", "disk_enospc", "crash_orphan_descendant")



def test_a_validation_accepts_sorted_json_round_trip() -> None:
    original = synthetic_i511a_admission()
    reloaded = json.loads(json.dumps(original, sort_keys=True))
    validated = validate_i5_11a_for_i5_11b(reloaded)
    assert set(validated["fault_matrix"]) == set(I511B_FAULT_SCENARIOS)

def test_a_validation_rejects_campaign_fault_order_drift() -> None:
    payload = synthetic_i511a_admission()
    payload["i5_11c_campaign_contract"]["fault_scenarios"] = [
        "crash_orphan_descendant",
        "cuda_oom",
        "disk_enospc",
    ]
    with pytest.raises(ReconstructionValidationError, match="fault_scenarios"):
        validate_i5_11a_for_i5_11b(payload)


def test_classifier_probe_is_pure_and_preserves_enospc_gap() -> None:
    probe = probe_failure_classifier_without_fault_execution()
    assert probe["fault_worker_executed"] is False
    assert probe["cuda_oom"]["observed_kind"] == "cuda_out_of_memory"
    assert probe["cuda_oom"]["first_class"] is True
    assert probe["disk_enospc"]["observed_kind"] == "unknown_model_error"
    assert probe["disk_enospc"]["gap_preserved"] is True
    assert probe["disk_enospc"]["raw_signature_preserved"] is True


def test_disk_probe_does_not_create_pressure(tmp_path: Path) -> None:
    before = list(tmp_path.iterdir())
    probe = observe_disk_space(tmp_path)
    after = list(tmp_path.iterdir())
    assert before == after == []
    assert probe["pressure_performed"] is False
    assert probe["files_created_for_probe"] is False
    assert probe["minimum_free_bytes_for_i5_11c"] == I511B_MIN_FREE_BYTES_FOR_C


def test_proc_discovery_finds_only_named_fault_workers(tmp_path: Path) -> None:
    (tmp_path / "100").mkdir()
    (tmp_path / "100" / "cmdline").write_bytes(b"python\x00scripts/i5_11_bounded_fault_worker.py\x00--mode\x00cuda_oom")
    (tmp_path / "101").mkdir()
    (tmp_path / "101" / "cmdline").write_bytes(b"python\x00something_else.py")
    found = discover_live_i5_11_fault_workers(tmp_path)
    assert [item["pid"] for item in found] == [100]


def test_a_validation_rejects_real_disk_fill_drift() -> None:
    payload = mutate(synthetic_i511a_admission(), "i5_11b_preflight_requirements", "real_disk_fill", value=True)
    with pytest.raises(ReconstructionValidationError, match="real_disk_fill"):
        validate_i5_11a_for_i5_11b(payload)


def test_build_authorizes_only_c_and_remains_execution_free() -> None:
    out = _build()
    assert out["pass"] is True
    assert out["scope"]["execution_free_preflight"] is True
    assert out["scope"]["fault_injection"] is False
    assert out["scope"]["supervised_fault_worker_execution"] is False
    assert out["decision"]["i5_11b_complete"] is True
    assert out["decision"]["i5_11c_authorized"] is True
    assert out["decision"]["i5_11c_started"] is False
    assert out["decision"]["i5_11d_authorized"] is False


def test_fault_plan_has_one_fault_and_one_recovery_each() -> None:
    out = _build()
    for scenario in I511B_FAULT_SCENARIOS:
        plan = out["fault_plan"][scenario]
        assert plan["attempts"] == 1
        assert plan["recovery_attempts"] == 1
        assert plan["expected_fault_state"] == "crashed"
        assert plan["expected_recovery_state"] == "completed"
        assert plan["real_gpu_exhaustion"] is False
        assert plan["real_disk_fill"] is False


def test_fault_plan_freezes_safe_signatures() -> None:
    out = _build()
    assert out["fault_plan"]["cuda_oom"]["expected_fault_signature"] == "simulated cuda out-of-memory"
    assert out["fault_plan"]["disk_enospc"]["expected_fault_signature"] == "No space left on device"
    assert out["fault_plan"]["crash_orphan_descendant"]["expected_fault_exit_code"] == 17


def test_low_disk_refuses_c_authorization() -> None:
    with pytest.raises(ReconstructionValidationError, match="safety floor"):
        _build(disk_observation=synthetic_disk_observation(free_bytes=2 * 1024**3))


def test_live_fault_worker_refuses_freeze() -> None:
    with pytest.raises(ReconstructionValidationError, match="live fault workers"):
        _build(live_fault_workers=[{"pid": 123, "cmdline": "python i5_11_bounded_fault_worker.py"}])


def test_classifier_gap_cannot_be_silently_changed() -> None:
    probe = synthetic_classifier_probe()
    probe["disk_enospc"]["gap_preserved"] = False
    with pytest.raises(ReconstructionValidationError, match="gap_preserved"):
        _build(classifier_probe=probe)


def test_campaign_contract_forbids_retuning_and_real_exhaustion() -> None:
    contract = _build()["campaign_contract"]
    assert contract["real_resource_exhaustion"] is False
    assert contract["retry_until_pass"] is False
    assert contract["result_driven_retuning"] is False
    assert contract["production_failure_classifier_modified"] is False


def test_abort_rules_fail_closed() -> None:
    rules = _build()["abort_rules"]
    assert rules["abort_on_process_group_leak"] is True
    assert rules["abort_on_workspace_cleanup_failure"] is True
    assert rules["abort_on_recovery_failure"] is True
    assert rules["no_retry_after_any_terminal_scenario"] is True


def test_evidence_schema_separates_fault_robustness_from_quality() -> None:
    schema = _build()["i5_11c_evidence_schema"]
    assert schema["expected_fault_is_not_reconstruction_quality"] is True
    assert schema["failed_recovery_is_robustness_failure"] is True
    assert "runtime_integrity_failure" in schema["campaign_terminal_states"]
