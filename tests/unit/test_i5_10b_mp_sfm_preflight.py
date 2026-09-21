from __future__ import annotations

from pathlib import Path
import pytest

from core.reconstruction.errors import ReconstructionValidationError
from core.reconstruction.i5_mp_sfm_preflight import (
    I510B_PHASE,
    I510B_I510A_FROZEN_SHA256,
    build_i5_10b_candidate_command_template,
    build_i5_10b_mp_sfm_preflight,
    revalidate_i5_10b_source,
    validate_i5_10a_for_i5_10b,
    validate_i5_4_chain_for_i5_10b,
    validate_i5_10b_runtime_probe,
)
from tests.i5_10b_helpers import (
    CHECKPOINT_SHA, I510A_SHA, I510B_RECEIPT_SHA, I54A_SHA, I54B_SHA, I54C_SHA,
    clone, evidence_binding, historical_chain, passing_runtime_probe,
    synthetic_i510a_admission, synthetic_i510b_receipt,
)


def _build(tmp_path: Path):
    a,b,c,dataset = historical_chain(tmp_path)
    source = revalidate_i5_10b_source(a, dataset)
    return build_i5_10b_mp_sfm_preflight(
        i5_10a_admission=synthetic_i510a_admission(),
        i5_10a_binding=evidence_binding('/evidence/i5_10a.json', I510A_SHA),
        i5_4a_admission=a,
        i5_4a_binding=evidence_binding('/evidence/i5_4a.json', I54A_SHA),
        i5_4b_replay=b,
        i5_4b_binding=evidence_binding('/evidence/i5_4b.json', I54B_SHA),
        i5_4c_closure=c,
        i5_4c_binding=evidence_binding('/evidence/i5_4c.json', I54C_SHA),
        source_revalidation=source,
        runtime_probe=passing_runtime_probe(),
        post_i5_10a_checkpoint_binding=evidence_binding('/checkpoints/post_i5_10a.tar.gz', CHECKPOINT_SHA),
        preflight_receipt=synthetic_i510b_receipt(),
        preflight_receipt_binding=evidence_binding('/evidence/i5_10b_receipt.json', I510B_RECEIPT_SHA),
        repository_anchors=None,
    )


def test_phase_and_frozen_a_hash():
    assert I510B_PHASE == 'I5.10B'
    assert I510B_I510A_FROZEN_SHA256 == I510A_SHA


def test_validates_i510a_boundary():
    out = validate_i5_10a_for_i5_10b(synthetic_i510a_admission())
    assert out['decision']['i5_10b_authorized'] is True
    assert out['decision']['real_mp_sfm_execution_started'] is False


def test_rejects_i510a_if_c_already_authorized():
    bad = synthetic_i510a_admission(); bad['decision']['i5_10c_authorized'] = True
    with pytest.raises(ReconstructionValidationError): validate_i5_10a_for_i5_10b(bad)


def test_historical_chain_is_exact_rejected_baseline(tmp_path):
    a,b,c,_ = historical_chain(tmp_path)
    out = validate_i5_4_chain_for_i5_10b(a,b,c, i5_4a_sha256=I54A_SHA, i5_4b_sha256=I54B_SHA)
    assert out['baseline_matching_strategy'] == 'sequential'
    assert out['baseline_sequential_overlap'] == 10
    assert out['input_images'] == 60
    assert out['registered_images'] == 0
    assert out['primary_bottleneck'] == 'incremental_mapping_no_sparse_model'


def test_source_revalidation_hashes_full_source_and_exact_window(tmp_path):
    a,_,_,dataset = historical_chain(tmp_path)
    out = revalidate_i5_10b_source(a, dataset)
    assert out['full_source_identity_verified'] is True
    assert out['exact_60_image_window_verified'] is True
    assert out['selected_source_sha256_verified'] is True
    assert out['window_image_count'] == 60


def test_source_revalidation_rejects_byte_drift(tmp_path):
    a,_,_,dataset = historical_chain(tmp_path)
    first = dataset / a['stress_plan']['forward_motion_window']['names'][0]
    first.write_bytes(first.read_bytes() + b'drift')
    with pytest.raises(ReconstructionValidationError): revalidate_i5_10b_source(a, dataset)


def test_runtime_probe_requires_recommended_version_and_apis():
    out = validate_i5_10b_runtime_probe(passing_runtime_probe())
    assert out['pycolmap_version'] == '4.1.1'
    assert out['apis']['match_exhaustive'] is True
    assert out['device_mode'] == 'cuda'


def test_runtime_probe_records_explicit_cpu_fallback():
    probe = passing_runtime_probe(); probe['cuda_devices']=0; probe['cuda_ready']=False
    out = validate_i5_10b_runtime_probe(probe)
    assert out['device_mode'] == 'cpu_fallback_explicit'
    assert out['cpu_fallback'] is True


def test_runtime_probe_rejects_wrong_version():
    probe = passing_runtime_probe(); probe['pycolmap_version']='4.0.1'; probe['recommended_version']=False
    with pytest.raises(ReconstructionValidationError): validate_i5_10b_runtime_probe(probe)


def test_runtime_probe_rejects_missing_exhaustive_api():
    probe = passing_runtime_probe(); probe['apis']['match_exhaustive']=False; probe['required_apis_available']=False
    with pytest.raises(ReconstructionValidationError): validate_i5_10b_runtime_probe(probe)


def test_candidate_command_freezes_only_matching_change():
    cmd = build_i5_10b_candidate_command_template('/opt/sfm/python', use_gpu=True)
    text = ' '.join(cmd)
    assert '--matching exhaustive' in text
    assert '--profile efficient' in text
    assert '--max-keyframes 60' in text
    assert '--image-sequence-fps 10.0' in text
    assert '--sequential-overlap' not in text
    assert '--cpu' not in text


def test_candidate_command_cpu_fallback_is_explicit():
    cmd = build_i5_10b_candidate_command_template('/opt/sfm/python', use_gpu=False)
    assert '--cpu' in cmd


def test_preflight_is_execution_free_and_authorizes_c(tmp_path):
    out = _build(tmp_path)
    assert out['pass'] is True
    assert out['scope']['execution_free'] is True
    assert out['scope']['real_mp_sfm_execution'] is False
    assert out['decision']['i5_10b_complete'] is True
    assert out['decision']['i5_10c_authorized'] is True
    assert out['decision']['i5_10c_started'] is False


def test_preflight_freezes_1770_exhaustive_pairs(tmp_path):
    out = _build(tmp_path)
    assert out['candidate_plan']['input_images'] == 60
    assert out['candidate_plan']['exhaustive_pair_count'] == 1770
    assert out['candidate_plan']['attempt_count'] == 1
    assert out['candidate_plan']['baseline_rerun'] is False


def test_preflight_binds_no_git_checkpoint(tmp_path):
    out = _build(tmp_path)
    assert out['evidence_bindings']['post_i5_10a_source_checkpoint']['sha256'] == CHECKPOINT_SHA
    assert out['source_control_policy']['git_or_gitea_required'] is False
    assert out['source_control_policy']['immutable_checkpoint_sha_required'] is True


def test_preflight_preserves_authority_and_no_metric_claim(tmp_path):
    out = _build(tmp_path)
    auth = out['authority']
    assert auth['camera_pose_authority_owner'] == 'pycolmap_sfm'
    assert auth['metric_scale_claimed'] is False
    assert auth['measurement_authoritative'] is False
    assert auth['candidate_reconstruction_is_evaluation_only'] is True


def test_preflight_does_not_start_later_roadmap(tmp_path):
    d = _build(tmp_path)['decision']
    assert d['i5_10d_started'] is False
    assert d['i5_11_started'] is False
    assert d['i5_12_started'] is False
    assert d['i5_13_started'] is False
    assert d['i5_14_started'] is False
    assert d['auto_policy_implemented'] is False
    assert d['hybrid_authorized'] is False


def test_preflight_freezes_resource_policy_and_c_evidence_schema(tmp_path):
    out = _build(tmp_path)
    resource = out['resource_policy']
    assert resource['input_images'] == 60
    assert resource['exhaustive_pair_count'] == 1770
    assert resource['adaptive_image_reduction'] is False
    assert resource['i5_11_owns_failure_injection'] is True
    schema = out['i5_10c_evidence_schema']
    assert schema['attempt_directory_must_be_new_and_immutable'] is True
    assert 'candidate_command' in schema['required_execution_records']
    assert schema['no_retry_artifact_overwrite'] is True
