import pytest
from core.reconstruction.errors import ReconstructionValidationError
from core.reconstruction.i5_mvs_robustness import validate_i5_6a_for_i5_6b, build_i5_6b_mvs_robustness_replay
from core.reconstruction.i5_difficult_handheld_robustness import I51EvidenceBinding
from tests.i5_6b_helpers import admission, receipt, scenario_record


def b(): return I51EvidenceBinding("/x.json", "1" * 64)


def test_rejects_permanent_cap_and_auto_promotion():
    x = admission(); x["execution_policy"]["permanent_global_view_cap"] = 12
    with pytest.raises(ReconstructionValidationError): validate_i5_6a_for_i5_6b(x)
    x = admission(); x["decision"]["auto_policy_implemented"] = True
    with pytest.raises(ReconstructionValidationError): validate_i5_6a_for_i5_6b(x)


def test_control_rejection_makes_runtime_quality_rejected_not_contract_failure():
    a=admission(); records={sid:scenario_record(sid) for sid in a["scenario_matrix"]}
    records["castle_control"] = scenario_record("castle_control", status="rejected")
    out=build_i5_6b_mvs_robustness_replay(
        i5_6a_admission=a,i5_6a_binding=b(),replay_receipt=receipt(),replay_receipt_binding=b(),
        runtime_preflight={"mvs_runtime_ready":True,"cuda_devices":1},scenario_records=records)
    assert out["quality_status"] == "rejected"
    assert out["pass"] is True


def test_recovery_refuses_false_completed_worker_state(tmp_path):
    from core.reconstruction.i5_mvs_robustness import recover_i5_6b_interrupted_native_scenario
    from tests.i5_6b_helpers import recovery_fixture
    a,p,r,s,f,v,d,n=recovery_fixture(tmp_path)
    s["stage_completed"] = True
    with pytest.raises(ReconstructionValidationError):
        recover_i5_6b_interrupted_native_scenario(
            admission_node=a,adaptive_plan=p,worker_request=r,worker_state=s,
            fused_ply=f,native_visibility=v,depth_maps_dir=d,normal_maps_dir=n,artifact_bindings={})


def test_recovery_refuses_mvs_config_relaxation(tmp_path):
    from core.reconstruction.i5_mvs_robustness import recover_i5_6b_interrupted_native_scenario
    from tests.i5_6b_helpers import recovery_fixture
    a,p,r,s,f,v,d,n=recovery_fixture(tmp_path)
    r["config"]["geometric_consistency"] = False
    with pytest.raises(ReconstructionValidationError):
        recover_i5_6b_interrupted_native_scenario(
            admission_node=a,adaptive_plan=p,worker_request=r,worker_state=s,
            fused_ply=f,native_visibility=v,depth_maps_dir=d,normal_maps_dir=n,artifact_bindings={})
