"""Deterministic Phase 6 benchmark asset tests."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from core.benchmark.assets import (
    ensure_benchmark_assets,
    load_asset_frames,
    validate_benchmark_assets,
)


def test_benchmark_assets_are_deterministic_and_valid(tmp_path: Path) -> None:
    first = ensure_benchmark_assets(tmp_path / "assets")
    first_checksums = dict(first.checksums)
    second = ensure_benchmark_assets(tmp_path / "assets", overwrite=True)

    assert second.checksums == first_checksums
    manifest = json.loads(second.manifest.read_text(encoding="utf-8"))
    assert manifest["color_order"] == "BGR"
    assert manifest["assets"]["landscape"]["shape"] == [384, 640, 3]
    assert manifest["assets"]["portrait"]["shape"] == [640, 384, 3]
    assert manifest["assets"]["temporal"]["shape"] == [12, 384, 640, 3]

    landscape = load_asset_frames(second, "landscape")
    temporal = load_asset_frames(second, "temporal")
    assert len(landscape) == 1
    assert len(temporal) == 12
    assert landscape[0].dtype == np.uint8
    assert not np.array_equal(temporal[0], temporal[-1])


def test_benchmark_asset_corruption_is_detected(tmp_path: Path) -> None:
    manifest = ensure_benchmark_assets(tmp_path / "assets")
    array = np.load(manifest.landscape, allow_pickle=False)
    array[0, 0, 0] ^= 1
    np.save(manifest.landscape, array, allow_pickle=False)

    with pytest.raises(ValueError, match="Checksum mismatch"):
        validate_benchmark_assets(manifest.root)
