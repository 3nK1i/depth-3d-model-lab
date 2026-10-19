from __future__ import annotations

from pathlib import Path

from core.reconstruction.i5_mp_sfm_preflight import (
    I510B_REPOSITORY_ANCHORS,
    build_i5_10b_candidate_command_template,
)


def test_i510b_repository_anchors_preserve_historical_preflight_evidence():
    assert I510B_REPOSITORY_ANCHORS["core/reconstruction/sfm_pycolmap.py"] == (
        "bdcc8a9cd2a885c51000f795ebe2edbc9a43449441fdbfae42c1d3e2f9455f59"
    )
    assert I510B_REPOSITORY_ANCHORS["scripts/reconstruction_pycolmap_worker.py"] == (
        "75277ced2640d61b7ebadb1541b1b28ae5b6dc8f5ba23183fb20e442cdd97dac"
    )
    assert I510B_REPOSITORY_ANCHORS["scripts/run_reconstruction_pycolmap_smoke.py"] == (
        "4585bc467166f8e9d4d689f97cb04b0fbefddb33c76d27ad5f4457192741180b"
    )
    assert all(
        len(value) == 64
        and all(ch in "0123456789abcdef" for ch in value)
        for value in I510B_REPOSITORY_ANCHORS.values()
    )


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
