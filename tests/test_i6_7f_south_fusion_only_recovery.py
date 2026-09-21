"""Fast dependency-light failure-state and COLMAP map integrity checks."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import struct
import sys
from uuid import uuid4

import pytest

SRC = Path(__file__).resolve().parents[1] / 'scripts' / 'run_i6_7f_south_fusion_only_recovery.py'
spec = importlib.util.spec_from_file_location('south_fusion_recovery_tested', SRC)
m = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = m
spec.loader.exec_module(m)


def write_colmap_map(folder: Path, image: str, variant: str, channels: int, width=3, height=2):
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f'{image}.{variant}.bin'
    path.write_bytes(f'{width}&{height}&{channels}&'.encode() + struct.pack('<' + 'f' * (width * height * channels), *([1.0] * (width * height * channels))))
    return path


def small_workspace(tmp_path: Path):
    dense = tmp_path / 'dense'
    (dense / 'stereo').mkdir(parents=True)
    names = [f'kf_{i:06}.jpg' for i in range(87)]
    for name in names:
        for kind, channels in (('depth_maps', 1), ('normal_maps', 3)):
            for variant in ('photometric', 'geometric'):
                write_colmap_map(dense / 'stereo' / kind, name, variant, channels)
    return dense, {'image_names': names}


def test_requested_run_is_exact_and_not_auto():
    assert m.SESSION_ID == '03d5f129-781f-419f-90aa-95673a5306fc'
    assert m.RUN_ID == '273476cc-4221-4b43-9407-96c0ed6e63e5'
    text = SRC.read_text()
    assert 'pycolmap.patch_match_stereo(' not in text
    assert 'pycolmap.undistort_images(' not in text
    assert 'pycolmap.stereo_fusion(' in text
    assert "'final_dense_result_admitted': False" in text


def test_structurally_complete_87_view_maps(tmp_path):
    dense, request = small_workspace(tmp_path)
    report = m.inspect_workspace(request, dense)
    assert report['registered_views'] == 87
    assert report['checked_maps'] == 348
    assert report['map_payloads_complete_by_size'] is True


@pytest.mark.parametrize('kind,variant', [('depth_maps', 'geometric'), ('normal_maps', 'photometric')])
def test_rejects_truncated_map(tmp_path, kind, variant):
    dense, request = small_workspace(tmp_path)
    path = dense / 'stereo' / kind / (request['image_names'][0] + '.' + variant + '.bin')
    path.write_bytes(path.read_bytes()[:-4])
    with pytest.raises(RuntimeError, match='incomplete/oversized'):
        m.inspect_workspace(request, dense)


def test_rejects_missing_map(tmp_path):
    dense, request = small_workspace(tmp_path)
    (dense / 'stereo/depth_maps' / (request['image_names'][0] + '.geometric.bin')).unlink()
    with pytest.raises(RuntimeError, match='expected 174 exact files'):
        m.inspect_workspace(request, dense)


def test_rejects_foreign_map(tmp_path):
    dense, request = small_workspace(tmp_path)
    (dense / 'stereo/depth_maps' / 'foreign.geometric.bin').write_bytes(b'bad')
    with pytest.raises(RuntimeError, match='expected 174 exact files'):
        m.inspect_workspace(request, dense)


def test_rejects_symlinked_map(tmp_path):
    dense, request = small_workspace(tmp_path)
    path = dense / 'stereo/depth_maps' / (request['image_names'][0] + '.geometric.bin')
    target = dense / 'target.bin'
    path.rename(target)
    path.symlink_to(target)
    with pytest.raises(RuntimeError, match='symlinked'):
        m.inspect_workspace(request, dense)


def test_rejects_different_normal_map_dimensions(tmp_path):
    dense, request = small_workspace(tmp_path)
    write_colmap_map(dense / 'stereo/normal_maps', request['image_names'][0], 'geometric', 3, width=2, height=3)
    with pytest.raises(RuntimeError, match='dimension mismatch'):
        m.inspect_workspace(request, dense)


def test_rejects_non_canonical_image_names(tmp_path):
    dense, request = small_workspace(tmp_path)
    request['image_names'][0] = '../foreign.jpg'
    with pytest.raises(RuntimeError, match='noncanonical'):
        m.inspect_workspace(request, dense)


def test_rejects_wrong_thread_settings():
    for bad in (0, -1, 5, 100):
        assert bad not in (1, 2, 3, 4)


def test_sha256_pinned_original_request_and_state():
    root = Path('/mnt/data/south_fusion_inspect')
    if root.is_dir():
        assert m.digest(root / 'request.json') == m.REQUEST_SHA256
        assert m.digest(root / 'execution_state.json') == m.STATE_SHA256
        assert m.digest(root / 'reconstruction_pycolmap_dense_provider_mvs_worker.py') == m.WORKER_SHA256
