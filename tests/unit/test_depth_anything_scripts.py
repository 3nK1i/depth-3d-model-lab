"""Installer and checkpoint downloader tests for both Depth Anything families."""

from __future__ import annotations

import json
from pathlib import Path
import sys

from scripts.download_depth_anything_models import (
    CHECKPOINTS,
    auto_variant,
    download_checkpoint,
    write_ready_marker,
)
from scripts.install_depth_anything_models import (
    SOURCES,
    prepare_source,
    verify_source_import,
)


class _Cuda:
    def __init__(self, gib: float, available: bool = True) -> None:
        self.gib = gib
        self.available = available

    def is_available(self):
        return self.available

    def get_device_properties(self, _index):
        return type("Props", (), {"total_memory": int(self.gib * 1024**3)})()


class _Torch:
    def __init__(self, gib: float, available: bool = True) -> None:
        self.cuda = _Cuda(gib, available)


def test_auto_variant_matches_gpu_classes() -> None:
    assert auto_variant(_Torch(6.0)) == "small"
    assert auto_variant(_Torch(12.0)) == "base"
    assert auto_variant(_Torch(32.0)) == "large"
    assert auto_variant(_Torch(0.0, False)) == "small"


def test_downloads_each_family_and_writes_ready_marker(tmp_path) -> None:
    calls = []

    def fake_download(**kwargs):
        calls.append(kwargs)
        destination = Path(kwargs["local_dir"]) / kwargs["filename"]
        destination.write_bytes(b"weights")
        return str(destination)

    for family in ("image", "video"):
        output = tmp_path / family
        destination = download_checkpoint(
            family, "small", output, hf_download_fn=fake_download, minimum_bytes=1
        )
        spec = CHECKPOINTS[family]["small"]
        assert destination.name == spec.filename
        assert calls[-1]["repo_id"] == spec.repo_id
        marker = write_ready_marker(family, output, ["small"])
        payload = json.loads(marker.read_text())
        assert payload["family"] == family
        assert payload["variants"] == ["small"]
        assert payload["checkpoints"][0]["filename"] == spec.filename


def test_prepare_source_validates_both_official_layouts(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(
        "scripts.install_depth_anything_models.subprocess.check_output",
        lambda *args, **kwargs: "abc123\n",
    )
    for family, spec in SOURCES.items():
        source = tmp_path / family
        (source / ".git").mkdir(parents=True)
        for relative in spec.required_files:
            path = source / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("# test\n")
        prepared = prepare_source(spec, source_dir=source)
        payload = json.loads((prepared / spec.marker).read_text())
        assert payload["family"] == family
        assert payload["commit"] == "abc123"
        assert payload["requested_ref"] == "main"


def test_verify_source_import_uses_isolated_interpreter(tmp_path, monkeypatch) -> None:
    calls = []

    def fake_run(command, *, cwd=None):
        calls.append((command, cwd))

    monkeypatch.setattr("scripts.install_depth_anything_models._run", fake_run)
    verify_source_import(SOURCES["video"], tmp_path)
    command, cwd = calls[-1]
    assert command[:2] == [sys.executable, "-c"]
    assert str(tmp_path.resolve()) in command[2]
    assert "video_depth_anything.video_depth_stream" in command[2]
    assert cwd is not None
