"""Prevent a diagnostics-only dense test from ever becoming a production job."""
from __future__ import annotations

from pathlib import Path
import tempfile

import pytest

from scripts.run_i6_8f_dense_telemetry_fixture import _fixture_root, _id, _parse

SID = '11111111-2222-4333-8444-555555555555'
RID = 'aaaaaaaa-bbbb-4ccc-8ddd-eeeeeeeeeeee'


def test_only_top_level_private_empty_temp_folder_may_receive_fixture_sidecars(tmp_path):
    with tempfile.TemporaryDirectory(prefix='i68f_dense_fixture_') as folder:
        assert _fixture_root(Path(folder)) == Path(folder)
        (Path(folder) / 'existing').write_text('keep')
        with pytest.raises(ValueError, match='empty'):
            _fixture_root(Path(folder))
    with pytest.raises(ValueError):
        _fixture_root(tmp_path)
    with pytest.raises((ValueError, FileNotFoundError)):
        _fixture_root(Path('/tmp/eia_depth_reconstruction'))


def test_test_flag_identity_and_input_restrictions(tmp_path):
    with pytest.raises(SystemExit):
        _parse(['--output-dir', str(tmp_path), '--session-id', SID, '--dense-run-id', RID])
    with pytest.raises(ValueError):
        _id('not-a-uuid')
    with pytest.raises(ValueError):
        _parse(['--test-only-no-mvs', '--output-dir', str(tmp_path),
                '--session-id', SID, '--dense-run-id', RID])


def test_fixture_never_invokes_expensive_dense_provider(monkeypatch):
    from scripts import run_i6_8f_dense_telemetry_fixture as fixture
    source = Path(fixture.__file__).read_text(encoding='utf-8')
    assert 'worker.execute_explicit_colmap_dense_after_sparse = fake_mvs_executor' in source
    assert 'raise RuntimeError(FIXTURE_FAILURE)' in source
    assert 'final_dense_admission' not in source
    assert 'image_undistortion' in source and 'patch_match_stereo' in source
