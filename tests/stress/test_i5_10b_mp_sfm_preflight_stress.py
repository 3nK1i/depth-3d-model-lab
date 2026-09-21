from __future__ import annotations

from pathlib import Path
import pytest

from core.reconstruction.errors import ReconstructionValidationError
from core.reconstruction.i5_mp_sfm_preflight import (
    build_i5_10b_mp_sfm_preflight,
    revalidate_i5_10b_source,
    validate_i5_4_chain_for_i5_10b,
    validate_i5_10b_receipt,
)
from tests.i5_10b_helpers import (
    CHECKPOINT_SHA, I510A_SHA, I510B_RECEIPT_SHA, I54A_SHA, I54B_SHA, I54C_SHA,
    clone, evidence_binding, historical_chain, passing_runtime_probe,
    synthetic_i510a_admission, synthetic_i510b_receipt,
)


def test_receipt_rejects_real_execution_flag():
    receipt = synthetic_i510b_receipt(); receipt['scope']['real_mp_sfm_execution'] = True
    with pytest.raises(ReconstructionValidationError): validate_i5_10b_receipt(receipt)


def test_chain_rejects_sparse_model_rewrite(tmp_path: Path):
    a,b,c,_ = historical_chain(tmp_path); b['replay']['mapping_produced_sparse_model'] = True
    with pytest.raises(ReconstructionValidationError):
        validate_i5_4_chain_for_i5_10b(a,b,c, i5_4a_sha256=I54A_SHA, i5_4b_sha256=I54B_SHA)


def test_chain_rejects_a_binding_mismatch(tmp_path: Path):
    a,b,c,_ = historical_chain(tmp_path); b['evidence_bindings']['i5_4a_admission']['sha256'] = 'd'*64
    with pytest.raises(ReconstructionValidationError):
        validate_i5_4_chain_for_i5_10b(a,b,c, i5_4a_sha256=I54A_SHA, i5_4b_sha256=I54B_SHA)


def test_chain_rejects_b_binding_mismatch(tmp_path: Path):
    a,b,c,_ = historical_chain(tmp_path); c['evidence_bindings']['i5_4b_real_replay']['sha256'] = 'e'*64
    with pytest.raises(ReconstructionValidationError):
        validate_i5_4_chain_for_i5_10b(a,b,c, i5_4a_sha256=I54A_SHA, i5_4b_sha256=I54B_SHA)


def test_build_rejects_wrong_frozen_i510a_binding(tmp_path: Path):
    a,b,c,dataset = historical_chain(tmp_path)
    source = revalidate_i5_10b_source(a,dataset)
    with pytest.raises(ReconstructionValidationError):
        build_i5_10b_mp_sfm_preflight(
            i5_10a_admission=synthetic_i510a_admission(), i5_10a_binding=evidence_binding('/e/a.json','f'*64),
            i5_4a_admission=a, i5_4a_binding=evidence_binding('/e/4a.json',I54A_SHA),
            i5_4b_replay=b, i5_4b_binding=evidence_binding('/e/4b.json',I54B_SHA),
            i5_4c_closure=c, i5_4c_binding=evidence_binding('/e/4c.json',I54C_SHA),
            source_revalidation=source, runtime_probe=passing_runtime_probe(),
            post_i5_10a_checkpoint_binding=evidence_binding('/c/post.tar.gz',CHECKPOINT_SHA),
            preflight_receipt=synthetic_i510b_receipt(), preflight_receipt_binding=evidence_binding('/e/r.json',I510B_RECEIPT_SHA),
            repository_anchors=None,
        )
