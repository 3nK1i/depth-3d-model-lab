"""Tests for the isolated UniDepthV2 installer and model downloader."""

from __future__ import annotations

import json
from pathlib import Path

from scripts.download_unidepth_v2_models import (
    auto_variant,
    download_variant,
    write_ready_marker,
)
from scripts.install_unidepth_v2 import COMPAT_MARKER, OFFICIAL_COMMIT, prepare_source


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


def test_auto_variant_matches_target_gpu_classes() -> None:
    assert auto_variant(_Torch(6.0)) == "small"
    assert auto_variant(_Torch(12.0)) == "base"
    assert auto_variant(_Torch(32.0)) == "large"
    assert auto_variant(_Torch(0.0, available=False)) == "small"


def test_download_is_local_atomic_enough_and_writes_ready_marker(tmp_path) -> None:
    calls = []

    def fake_snapshot_download(**kwargs):
        calls.append(kwargs)
        destination = Path(kwargs["local_dir"])
        (destination / "config.json").write_text("{}")
        (destination / "model.safetensors").write_bytes(b"weights")
        return str(destination)

    destination = download_variant(
        "small", tmp_path, snapshot_download_fn=fake_snapshot_download
    )
    assert destination.name == "vits14"
    assert calls[0]["repo_id"] == "lpiccinelli/unidepth-v2-vits14"
    assert calls[0]["revision"] == "038c238f06c87b6c2f5b3749fd51fbf442b1f218"
    marker = write_ready_marker(tmp_path, ["small"])
    assert json.loads(marker.read_text())["downloaded_variants"] == ["small"]


def test_prepare_source_applies_v2_only_import_compatibility(
    tmp_path, monkeypatch
) -> None:
    source = tmp_path / "UniDepth"
    (source / ".git").mkdir(parents=True)
    models_init = source / "unidepth" / "models" / "__init__.py"
    unidepth_v2_init = source / "unidepth" / "models" / "unidepthv2" / "__init__.py"
    layers_init = source / "unidepth" / "layers" / "__init__.py"
    models_init.parent.mkdir(parents=True)
    unidepth_v2_init.parent.mkdir(parents=True)
    layers_init.parent.mkdir(parents=True)
    models_init.write_text("from .unidepthv1 import UniDepthV1\n")
    unidepth_v2_init.write_text("from .unidepthv2_old import UniDepthV2old\n")
    layers_init.write_text("from .nystrom_attention import NystromBlock\n")
    monkeypatch.setattr(
        "scripts.install_unidepth_v2.subprocess.check_output",
        lambda *args, **kwargs: "abc123\n",
    )

    prepared = prepare_source(source)
    assert prepared == source.resolve()
    assert "UniDepthV1" not in models_init.read_text()
    assert "from .nystrom_attention" not in layers_init.read_text()
    assert "UniDepthV2" in models_init.read_text()
    assert "UniDepthV2old" not in unidepth_v2_init.read_text()
    marker = json.loads((source / COMPAT_MARKER).read_text())
    assert marker["commit"] == "abc123"
    assert marker["ref"] == OFFICIAL_COMMIT
