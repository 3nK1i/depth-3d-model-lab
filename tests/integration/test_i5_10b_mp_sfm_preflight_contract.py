from __future__ import annotations

from pathlib import Path

from core.reconstruction.i5_mp_sfm_preflight import (
    I510B_REPOSITORY_ANCHORS,
    build_i5_10b_candidate_command_template,
    verify_i5_10b_repository_anchors,
)


def test_current_repository_anchor_hashes_match_preflight_design():
    root = Path(__file__).resolve().parents[2]
    verified = verify_i5_10b_repository_anchors(root)
    assert set(verified) == set(I510B_REPOSITORY_ANCHORS)
    assert all(x['verified'] for x in verified.values())


def test_candidate_command_uses_existing_production_smoke_runner():
    cmd = build_i5_10b_candidate_command_template('/env/sfm/bin/python', use_gpu=True)
    assert cmd[0] == '${CURRENT_PYTHON}'
    assert cmd[1] == 'scripts/run_reconstruction_pycolmap_smoke.py'
    assert cmd[cmd.index('--matching') + 1] == 'exhaustive'


def test_worker_currently_contains_exhaustive_matching_call():
    root = Path(__file__).resolve().parents[2]
    text = (root / 'scripts/reconstruction_pycolmap_worker.py').read_text(encoding='utf-8')
    assert 'pycolmap.match_exhaustive(' in text
    assert 'pycolmap.match_sequential(' in text


def test_preflight_module_contains_no_sfm_execution_api_call():
    root = Path(__file__).resolve().parents[2]
    text = (root / 'core/reconstruction/i5_mp_sfm_preflight.py').read_text(encoding='utf-8')
    assert 'run_sparse_reconstruction(' not in text
    assert 'incremental_mapping(' not in text
    assert 'match_exhaustive(' not in text
