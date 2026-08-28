from __future__ import annotations

from pathlib import Path

from core.reconstruction.mvs_da3_qualification import build_source_identity, discover_image_directories


def test_large_manifest_identity_is_stable(tmp_path: Path) -> None:
    root = tmp_path / "large"
    root.mkdir()
    for idx in range(300):
        (root / f"frame_{idx:04d}.jpg").write_bytes((f"payload-{idx}-" * 10).encode())
    a = build_source_identity(root)
    b = build_source_identity(root)
    assert a.image_count == 300
    assert a.asset_set_sha256 == b.asset_set_sha256


def test_discovery_is_bounded(tmp_path: Path) -> None:
    for directory_idx in range(20):
        root = tmp_path / f"set_{directory_idx:02d}"
        root.mkdir()
        for image_idx in range(3):
            (root / f"{image_idx}.png").write_bytes(b"x")
    found = discover_image_directories([tmp_path], maximum_candidates=7)
    assert len(found) == 7
