"""Checkpoint downloader tests using an in-memory response."""

from __future__ import annotations

import importlib.util
import io
from pathlib import Path


_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "download_depth_pro_checkpoint.py"
_SPEC = importlib.util.spec_from_file_location("depth_pro_downloader", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


class _Response(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False


def test_downloader_is_atomic_and_skips_existing_file(tmp_path, monkeypatch) -> None:
    output = tmp_path / "checkpoints" / "depth_pro.pt"
    monkeypatch.setattr(_MODULE, "_MINIMUM_EXPECTED_BYTES", 4)
    calls = []

    def fake_urlopen(url, timeout):
        calls.append((url, timeout))
        return _Response(b"weights")

    monkeypatch.setattr(_MODULE.urllib.request, "urlopen", fake_urlopen)
    assert _MODULE.download_checkpoint(output) == output.resolve()
    assert output.read_bytes() == b"weights"
    assert not output.with_suffix(".pt.part").exists()
    assert len(calls) == 1

    assert _MODULE.download_checkpoint(output) == output.resolve()
    assert len(calls) == 1
