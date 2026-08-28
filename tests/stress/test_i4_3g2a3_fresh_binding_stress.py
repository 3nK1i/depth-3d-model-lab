from __future__ import annotations

from pathlib import Path

from core.reconstruction.mvs_da3_fresh_scenario_binding import build_exact_source_identity


def test_large_exact_subset_manifest_is_stable(tmp_path: Path) -> None:
    root = tmp_path / "large"
    root.mkdir()
    for index in range(500):
        (root / f"frame_{index:04d}.jpg").write_bytes((f"payload-{index}-" * 12).encode())
    names = tuple(f"frame_{index:04d}.jpg" for index in range(0, 500, 3))
    first = build_exact_source_identity(root, names=names)
    second = build_exact_source_identity(root, names=names)
    assert first["image_count"] == len(names)
    assert first["asset_set_sha256"] == second["asset_set_sha256"]
