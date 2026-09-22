"""Fail-closed I6.8F real-worker fixture/preflight: no actual workers in tests."""
from pathlib import Path
import subprocess
import sys

import pytest

from scripts import run_i6_8f_real_worker_qualification as real


def create_images(folder, count=12, *, dimensions=(640, 480)):
    PIL = pytest.importorskip('PIL.Image')
    folder.mkdir()
    for n in range(count):
        PIL.new('RGB', dimensions).save(folder / f'templeR{n+1}.png')


def test_selects_eight_adjacent_naturally_sorted_unmodified_real_paths(tmp_path):
    source = tmp_path / 'original'
    create_images(source)
    selected = real.select_source_images(source, count=8)
    assert [p.name for p in selected] == [f'templeR{n}.png' for n in range(3, 11)]
    assert len(list(source.iterdir())) == 12


def test_preflight_rejects_nonexistent_or_undersized_or_oversized_sources(tmp_path):
    with pytest.raises(ValueError, match='must exist'):
        real.select_source_images(tmp_path / 'absent')
    source = tmp_path / 'small'
    create_images(source, 3)
    with pytest.raises(ValueError, match='need 8'):
        real.select_source_images(source, count=8)
    source2 = tmp_path / 'huge'
    create_images(source2, 8, dimensions=(1600, 1200))
    with pytest.raises(ValueError, match='size outside'):
        real.select_source_images(source2, count=8)


def test_rejects_input_symlinks_and_invalid_count(tmp_path):
    source = tmp_path / 'images'
    create_images(source, 8)
    (source / 'templeR3.png').unlink()
    (source / 'templeR3.png').symlink_to(source / 'templeR1.png')
    with pytest.raises(ValueError, match='ordinary file'):
        real.select_source_images(source, count=8)
    with pytest.raises(ValueError, match='8..24'):
        real.select_source_images(source, count=25)


def test_external_runtime_is_checked_not_assumed(tmp_path, monkeypatch):
    py = tmp_path / 'bin'
    py.write_text('test')
    def fake_run(*args, **kwargs):
        return subprocess.CompletedProcess(args[0], 0, stdout='4.1.1\n', stderr='')
    monkeypatch.setattr(real.subprocess, 'run', fake_run)
    assert real.sfm_runtime_check(py) == '4.1.1'
    def bad_run(*args, **kwargs):
        return subprocess.CompletedProcess(args[0], 1, stdout='', stderr='not found')
    monkeypatch.setattr(real.subprocess, 'run', bad_run)
    with pytest.raises(RuntimeError, match='required'):
        real.sfm_runtime_check(py)


def test_fail_closed_resource_gate_before_any_job(tmp_path, monkeypatch):
    monkeypatch.setattr(real, 'mem_available_bytes', lambda: 2 * 1024**3)
    with pytest.raises(RuntimeError, match='RAM'):
        real.safety_check(tmp_path)
    monkeypatch.setattr(real, 'mem_available_bytes', lambda: 20 * 1024**3)
    monkeypatch.setattr(real.shutil, 'disk_usage', lambda _: type('D', (), {'free': 1*1024**3})())
    with pytest.raises(RuntimeError, match='disk'):
        real.safety_check(tmp_path)


def test_does_not_modify_source_or_run_without_explicit_run(tmp_path, monkeypatch, capsys):
    source = tmp_path / 'images'
    create_images(source, 8)
    monkeypatch.setattr(real, 'sfm_runtime_check', lambda _: '4.1.1')
    monkeypatch.setattr(real, 'safety_check', lambda _: (10.0, 20.0, 0.0))
    monkeypatch.setattr(sys, 'argv', ['probe', '--images-dir', str(source), '--image-count', '8'])
    assert real.main() == 0
    assert 'ACTUAL_RECONSTRUCTION_STARTED=NO' in capsys.readouterr().out
    assert len(list(tmp_path.iterdir())) == 1


def test_wall_clock_group_termination_targets_only_own_process():
    child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(20)'],
                             start_new_session=True)
    try:
        real.terminate_owned_group(child)
        assert child.poll() is not None
    finally:
        if child.poll() is None:
            child.kill()
            child.wait(timeout=2)


def test_actual_run_is_opt_in_and_bounded_by_declared_limits():
    import inspect
    source = inspect.getsource(real.real_worker_probe)
    assert "'--cpu'" in source
    assert 'start_new_session=True' in source
    assert 'JobTelemetryQtRelay' in source
    assert 'run_reconstruction_offline_job.py' in source
    assert 'JobTelemetryPanelState' in source
    assert 'MAX_RUN_SECONDS' in source
    assert 'terminate_owned_group' in source
    assert 'reconstruction_pycolmap_dense_provider_mvs_worker' not in source
    assert real.IMAGE_COUNT == 24
    assert real.MIN_IMAGE_COUNT == 8
    assert "str(len(images))" in source
    assert "resource_sidecar_seen_while_worker_alive" in source
    assert "worker_wall_seconds" in source
    assert real.MAX_RUN_SECONDS == 120
    assert real.MAX_TREE_RSS_GIB == 5.0
    assert real.MAX_SCRATCH_BYTES == 1024**3


def test_default_fixture_expands_to_twenty_four_but_preserves_bounds(tmp_path):
    source = tmp_path / 'original'
    create_images(source, 32)
    selected = real.select_source_images(source)
    assert len(selected) == 24
    assert selected[0].name == 'templeR5.png'
    assert selected[-1].name == 'templeR28.png'
    assert len(list(source.iterdir())) == 32
    with pytest.raises(ValueError, match='8..24'):
        real.select_source_images(source, count=7)
    with pytest.raises(ValueError, match='8..24'):
        real.select_source_images(source, count=25)


def test_24_image_preflight_still_does_not_launch_a_worker(tmp_path, monkeypatch, capsys):
    source = tmp_path / 'input'
    create_images(source, 24)
    monkeypatch.setattr(real, 'sfm_runtime_check', lambda _: '4.1.1')
    monkeypatch.setattr(real, 'safety_check', lambda _: (10.0, 20.0, 0.0))
    monkeypatch.setattr(sys, 'argv', ['probe', '--images-dir', str(source)])
    assert real.main() == 0
    output = capsys.readouterr().out
    assert 'SELECTED_IMAGE_COUNT=24' in output
    assert 'ACTUAL_RECONSTRUCTION_STARTED=NO' in output
    assert len(list(tmp_path.iterdir())) == 1
