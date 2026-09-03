from __future__ import annotations

from pathlib import Path

import pytest

from core.reconstruction.i5_low_overlap_weak_parallax import I52AStressPlanPolicy, build_ordered_source_identity, build_stress_plan
from tests.i5_2a_helpers import make_ordered_dataset


@pytest.mark.parametrize("count", [12, 19, 30, 64, 121])
def test_i52a_plan_scales_deterministically_without_unbounded_slice_growth(tmp_path: Path, count: int) -> None:
    identity = build_ordered_source_identity(make_ordered_dataset(tmp_path / f"d{count}", count=count))
    plan = build_stress_plan(identity, policy=I52AStressPlanPolicy())
    low = plan["low_overlap_proxy"]
    weak = plan["weak_parallax_proxy"]
    assert 6 <= low["image_count"] <= 24
    assert 6 <= weak["image_count"] <= 24
    assert low["indices"] == sorted(set(low["indices"]))
    assert weak["indices"] == sorted(set(weak["indices"]))
    assert low["indices"][0] == 0
    assert low["indices"][-1] == count - 1
    assert weak["source_span_fraction"] < low["source_span_fraction"]


def test_i52a_source_identity_changes_when_one_pixel_asset_changes(tmp_path: Path) -> None:
    dataset = make_ordered_dataset(tmp_path / "scene")
    before = build_ordered_source_identity(dataset)
    path = dataset / "castle05.png"
    data = bytearray(path.read_bytes())
    data[-8] ^= 1
    path.write_bytes(data)
    after = build_ordered_source_identity(dataset)
    assert before["asset_set_sha256"] != after["asset_set_sha256"]
