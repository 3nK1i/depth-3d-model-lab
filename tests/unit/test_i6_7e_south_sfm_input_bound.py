"""South Building regression: accepted SfM JSON may exceed offline metadata cap."""
from types import SimpleNamespace

import pytest

from app.reconstruction_dense_production_worker import _read_json
from core.reconstruction.errors import ReconstructionValidationError


class _SizedJson:
    """Tiny in-memory JSON with a simulated on-disk size; no 94 MiB test fixture."""
    def __init__(self, size):
        self.size = size
        self.read_calls = 0

    def is_file(self):
        return True

    def is_symlink(self):
        return False

    def stat(self):
        return SimpleNamespace(st_size=self.size)

    def read_text(self, *, encoding):
        assert encoding == 'utf-8'
        self.read_calls += 1
        return '{"registered_keyframes": 87}'


@pytest.mark.parametrize('size', [64 * 1024 * 1024 + 1, 94_421_034, 128 * 1024 * 1024])
def test_genuine_sfm_metadata_can_be_read_within_separate_128_mib_bound(size):
    path = _SizedJson(size)
    assert _read_json(path, 'SfM result') == {'registered_keyframes': 87}
    assert path.read_calls == 1


@pytest.mark.parametrize('label, size', [
    ('SfM result', 128 * 1024 * 1024 + 1),
    ('offline result', 64 * 1024 * 1024 + 1),
    ('unrecognized evidence', 64 * 1024 * 1024 + 1),
])
def test_oversize_input_is_rejected_before_json_read(label, size):
    path = _SizedJson(size)
    with pytest.raises(ReconstructionValidationError, match=f'{label} exceeds bounded input size'):
        _read_json(path, label)
    assert path.read_calls == 0


def test_offline_evidence_remains_bounded_and_readable():
    path = _SizedJson(64 * 1024 * 1024)
    assert _read_json(path, 'offline result') == {'registered_keyframes': 87}
    assert path.read_calls == 1
