"""Tests for the Metric3D v2 installer and checkpoint downloader."""

from __future__ import annotations

import json
from pathlib import Path

from scripts.download_metric3d_v2_models import (
    REPOSITORY,
    VARIANTS,
    auto_variant,
    download_variant,
    write_ready_marker,
)
from scripts.install_metric3d_v2 import OFFICIAL_COMMIT, READY_MARKER, prepare_source


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


def test_auto_variant_matches_rtx2060_and_rtx5090_classes() -> None:
    assert auto_variant(_Torch(6.0)) == "small"
    assert auto_variant(_Torch(16.0)) == "small"
    assert auto_variant(_Torch(32.0)) == "large"
    assert auto_variant(_Torch(0.0, available=False)) == "small"


def test_download_writes_checkpoint_and_ready_marker(tmp_path) -> None:
    calls = []

    def fake_download(**kwargs):
        calls.append(kwargs)
        destination = Path(kwargs["local_dir"]) / kwargs["filename"]
        destination.write_bytes(b"weights")
        return str(destination)

    destination = download_variant(
        "small",
        tmp_path,
        hf_download_fn=fake_download,
        minimum_bytes=1,
    )
    assert destination.name == VARIANTS["small"]
    assert calls[0]["repo_id"] == REPOSITORY
    assert calls[0]["revision"] == "main"
    marker = write_ready_marker(tmp_path, ["small"])
    payload = json.loads(marker.read_text())
    assert payload["downloaded_variants"] == ["small"]
    assert payload["files"] == [VARIANTS["small"]]


def test_prepare_source_validates_layout_and_writes_pinned_marker(
    tmp_path, monkeypatch
) -> None:
    source = tmp_path / "Metric3D"
    (source / ".git").mkdir(parents=True)
    for relative in (
        "hubconf.py",
        "LICENSE",
        "mono/configs/HourglassDecoder/vit.raft5.small.py",
        "mono/configs/HourglassDecoder/vit.raft5.large.py",
    ):
        path = source / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# test\n", encoding="utf-8")
    monkeypatch.setattr(
        "scripts.install_metric3d_v2.subprocess.check_output",
        lambda *args, **kwargs: "abc123\n",
    )

    prepared = prepare_source(source)
    assert prepared == source.resolve()
    marker = json.loads((source / READY_MARKER).read_text())
    assert marker["commit"] == "abc123"
    assert marker["ref"] == OFFICIAL_COMMIT
    assert marker["models"] == ["metric3d_vit_small", "metric3d_vit_large"]
