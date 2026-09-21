from __future__ import annotations

from dataclasses import replace
import hashlib
from pathlib import Path
import struct

import pytest

from core.reconstruction.errors import ReconstructionValidationError
from core.reconstruction.reconstruction_dense_artifact_preflight import (
    I67E1_CHUNK_POINTS, inspect_full_dense_ply,
)
from scripts.run_i6_7e1_full_dense_artifact_probe import verify_report


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def binary_ply(path: Path, count: int, *, nan_at: int = -1, rgb: int = 255) -> None:
    path.write_bytes((
        'ply\nformat binary_little_endian 1.0\n'
        f'element vertex {count}\n'
        'property float x\nproperty float y\nproperty float z\n'
        'property uchar red\nproperty uchar green\nproperty uchar blue\n'
        'end_header\n'
    ).encode('ascii') + b''.join(
        struct.pack('<fffBBB', float('nan') if i == nan_at else float(i),
                    0.5, -float(i), rgb, 10, 20)
        for i in range(count)
    ))


def ascii_ply(path: Path, text: str) -> None:
    path.write_text(
        'ply\nformat ascii 1.0\nelement vertex 2\n'
        'property float x\nproperty float y\nproperty float z\n'
        'property uchar red\nproperty uchar green\nproperty uchar blue\n'
        'end_header\n' + text, encoding='ascii'
    )


def check(path: Path, *, count: int | None = None):
    return inspect_full_dense_ply(path, expected_sha256=digest(path),
                                  expected_root=path.parent, expected_vertex_count=count)


def test_scans_full_binary_population_without_treating_preview_limit_as_geometry_limit(tmp_path):
    path = tmp_path / 'fused.ply'
    binary_ply(path, I67E1_CHUNK_POINTS + 7)
    result = check(path, count=I67E1_CHUNK_POINTS + 7)
    assert result.scanned_vertex_count == I67E1_CHUNK_POINTS + 7
    assert result.vertex_count == I67E1_CHUNK_POINTS + 7
    assert result.bounds_min == (0.0, 0.5, -(I67E1_CHUNK_POINTS + 6))
    assert result.bounds_max == (float(I67E1_CHUNK_POINTS + 6), 0.5, 0.0)
    assert result.has_rgb and result.ply_format == 'binary_little_endian'
    assert result.as_dict()['final_dense_result_admitted'] is False
    assert result.as_dict()['metric_scale_claimed'] is False
    assert result.as_dict()['persistent_dense_scene_created'] is False


def test_ascii_full_population(tmp_path):
    path = tmp_path / 'fused.ply'
    ascii_ply(path, '1 2 3 0 100 255\n4 5 6 12 16 250\n')
    result = check(path, count=2)
    assert result.bounds_min == (1.0, 2.0, 3.0)
    assert result.bounds_max == (4.0, 5.0, 6.0)


@pytest.mark.parametrize('rows', [
    '1 2 3 0 100 255\n4 5 6 300 16 250\n',
    '1 2 3 0 100 255\n4 NaN 6 10 16 250\n',
    '1 2 3 0 100 255\n4 5 6 10 16\n',
    '1 2 3 0 100 255\n',
])
def test_ascii_rejects_invalid_complete_population(tmp_path, rows):
    path = tmp_path / 'fused.ply'
    ascii_ply(path, rows)
    with pytest.raises(ReconstructionValidationError):
        check(path)


def test_binary_rejects_nan_in_unsampled_region(tmp_path):
    path = tmp_path / 'fused.ply'
    binary_ply(path, I67E1_CHUNK_POINTS + 11, nan_at=I67E1_CHUNK_POINTS + 3)
    with pytest.raises(ReconstructionValidationError, match='non-finite'):
        check(path)


def test_binary_rejects_truncation_or_changed_sha_or_count(tmp_path):
    path = tmp_path / 'fused.ply'
    binary_ply(path, 7)
    with pytest.raises(ReconstructionValidationError, match='count disagrees'):
        check(path, count=8)
    with pytest.raises(ReconstructionValidationError, match='SHA256 mismatch'):
        inspect_full_dense_ply(path, expected_root=tmp_path, expected_sha256='a'*64)
    path.write_bytes(path.read_bytes()[:-1])
    with pytest.raises(ReconstructionValidationError, match='truncated'):
        check(path)


def test_symlinks_and_external_geometry_rejected(tmp_path):
    root = tmp_path / 'admitted'; root.mkdir()
    external = tmp_path / 'external.ply'; binary_ply(external, 1)
    with pytest.raises(ReconstructionValidationError, match='escapes admitted'):
        inspect_full_dense_ply(external, expected_root=root, expected_sha256=digest(external))
    link = root / 'linked.ply'; link.symlink_to(external)
    with pytest.raises(ReconstructionValidationError, match='symlinked'):
        inspect_full_dense_ply(link, expected_root=root, expected_sha256=digest(external))


def report_for(path: Path, *, count: int):
    return {
        'pass': True, 'status': 'pass',
        'provider': {'selection_mode': 'explicit_provider', 'implementation': 'colmap_mvs',
                     'provider_id': 'colmap-mvs-external',
                     'result': {'status': 'completed', 'implementation': 'colmap_mvs',
                                'provider_id': 'colmap-mvs-external', 'selected_view_count': 24,
                                'canonical_frame_id': 'sfm-frame', 'canonical_geometry_ref': str(path),
                                'authority': {'camera_pose_authority_owner': 'pycolmap_sfm',
                                              'single_authoritative_sfm_frame_preserved': True,
                                              'canonical_geometry_authoritative': True}}},
        'source_sfm': {'registered_views': 24, 'reconstruction_frame_id': 'sfm-frame'},
        'artifacts': {'canonical_geometry': str(path), 'canonical_geometry_sha256': digest(path)},
        'progressive': {'current_dense_points': count, 'preview_point_count': 4096},
        'authority': {field: False for field in (
            'final_dense_result_admitted_by_progressive_path', 'persistent_dense_scene_created',
            'production_auto_activated', 'metric_scale_claimed', 'measurement_authoritative',
            'certified_metrology_claimed', 'surface_mesh_generated')},
    }


def test_report_is_identity_guard_not_an_admission(tmp_path):
    path = tmp_path / 'provider_colmap_mvs' / 'dense' / 'fused.ply'
    path.parent.mkdir(parents=True)
    binary_ply(path, 4097)
    report = report_for(path, count=4097)
    source, count, sha, output = verify_report(report, tmp_path / 'report.json')
    inspected = inspect_full_dense_ply(source, expected_sha256=sha,
                                       expected_root=output, expected_vertex_count=count)
    assert inspected.vertex_count == 4097 and not inspected.is_final_admitted
    report['provider']['selection_mode'] = 'auto'
    with pytest.raises(ReconstructionValidationError, match='explicit'):
        verify_report(report, tmp_path / 'report.json')


def test_report_rejects_foreign_frame_wrong_count_and_false_authority(tmp_path):
    path = tmp_path / 'provider_colmap_mvs' / 'dense' / 'fused.ply'
    path.parent.mkdir(parents=True)
    binary_ply(path, 4097)
    report = report_for(path, count=4097)
    report['provider']['result']['canonical_frame_id'] = 'other'
    with pytest.raises(ReconstructionValidationError, match='frame'):
        verify_report(report, tmp_path / 'report.json')
    report['provider']['result']['canonical_frame_id'] = 'sfm-frame'
    report['progressive']['current_dense_points'] = 4098
    source, count, sha, output = verify_report(report, tmp_path / 'report.json')
    with pytest.raises(ReconstructionValidationError, match='count disagrees'):
        inspect_full_dense_ply(source, expected_sha256=sha,
                               expected_root=output, expected_vertex_count=count)
    report['authority']['production_auto_activated'] = True
    with pytest.raises(ReconstructionValidationError, match='authority'):
        verify_report(report, tmp_path / 'report.json')


def test_binary_big_endian_color_and_geometry(tmp_path):
    path = tmp_path / 'fused.ply'
    path.write_bytes((
        'ply\nformat binary_big_endian 1.0\nelement vertex 2\n'
        'property float x\nproperty float y\nproperty float z\n'
        'property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n'
    ).encode() + struct.pack('>fffBBBfffBBB', 1, 2, 3, 1, 2, 3, 4, 5, 6, 4, 5, 6))
    result = check(path, count=2)
    assert result.bounds_min == (1.0, 2.0, 3.0)
    assert result.bounds_max == (4.0, 5.0, 6.0)
    assert result.ply_format == 'binary_big_endian'


def test_binary_rejects_out_of_range_rgb_even_if_sha_matches(tmp_path):
    path = tmp_path / 'fused.ply'
    path.write_bytes((
        'ply\nformat binary_little_endian 1.0\nelement vertex 1\n'
        'property float x\nproperty float y\nproperty float z\n'
        'property float red\nproperty float green\nproperty float blue\nend_header\n'
    ).encode() + struct.pack('<ffffff', 1, 2, 3, 300, 10, 20))
    with pytest.raises(ReconstructionValidationError, match='RGB'):
        check(path)


def test_header_rejects_missing_properties_duplicate_and_overlong_header(tmp_path):
    path = tmp_path / 'fused.ply'
    path.write_bytes(b'ply\nformat binary_little_endian 1.0\nelement vertex 1\n'
                     b'property float x\nproperty float x\nproperty float y\n'
                     b'property float z\nend_header\n' + struct.pack('<ffff', 1, 2, 3, 4))
    with pytest.raises(ReconstructionValidationError, match='duplicated'):
        check(path)
    path.write_bytes(b'ply\n' + b'comment ' + b'a' * 70000 + b'\nend_header\n')
    with pytest.raises(ReconstructionValidationError, match='header exceeds'):
        check(path)


def test_cli_read_only_synthetic_full_dense_source(tmp_path, capsys, monkeypatch):
    import json
    from scripts.run_i6_7e1_full_dense_artifact_probe import main
    path = tmp_path / 'provider_colmap_mvs' / 'dense' / 'fused.ply'
    path.parent.mkdir(parents=True)
    binary_ply(path, 4097)
    report_path = tmp_path / 'i6_7b_progressive_dense_emission_report.json'
    report_path.write_text(json.dumps(report_for(path, count=4097)), encoding='utf-8')
    monkeypatch.setattr('sys.argv', ['probe', '--source-report', str(report_path),
                         '--expected-source-report-sha256', digest(report_path)])
    assert main() == 0
    stdout = capsys.readouterr().out
    assert 'FULL_SCANNED_VERTICES=4097' in stdout
    assert 'FINAL_DENSE_RESULT_ADMITTED=NO' in stdout
    assert 'FULL_DENSITY_GPU_VIEWER_ENABLED=NO' in stdout
