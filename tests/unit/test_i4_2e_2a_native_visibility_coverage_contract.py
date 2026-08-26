from __future__ import annotations

import importlib.util
from pathlib import Path
import struct


def _worker():
    path = Path(__file__).resolve().parents[2] / "scripts/reconstruction_pycolmap_temple_ring_detail_support_worker.py"
    spec = importlib.util.spec_from_file_location("i4_2e_2a_visibility_worker_unit", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_vis(path: Path, rows: list[list[int]]) -> None:
    with path.open("wb") as handle:
        handle.write(struct.pack("<Q", len(rows)))
        for row in rows:
            handle.write(struct.pack("<I", len(row)))
            if row:
                handle.write(struct.pack(f"<{len(row)}I", *row))


def test_i42e2a_worker_accepts_valid_global_subset(tmp_path: Path) -> None:
    worker = _worker()
    path = tmp_path / "fused.ply.vis"
    _write_vis(path, [[5, 2], [3]])
    offsets, indices = worker._read_points_visibility(path, expected_points=2, expected_views=6)
    assert offsets.tolist() == [0, 2, 3]
    assert indices.tolist() == [2, 5, 3]


def test_i42e2a_worker_accepts_highest_in_range_index_without_full_coverage(tmp_path: Path) -> None:
    worker = _worker()
    path = tmp_path / "fused.ply.vis"
    _write_vis(path, [[5], [5]])
    offsets, indices = worker._read_points_visibility(path, expected_points=2, expected_views=6)
    assert offsets.tolist() == [0, 1, 2]
    assert indices.tolist() == [5, 5]


def test_i42e2a_core_source_keeps_range_gate_but_drops_global_full_set_gate() -> None:
    source = (Path(__file__).resolve().parents[2] / "core/reconstruction/temple_ring_detail_qualification.py").read_text(encoding="utf-8")
    assert "native visibility image indices must lie in [0," in source
    assert "full-set coverage is not a native visibility validity rule" in source
    assert "native visibility must cover the exact MVS image-index set" not in source

def test_i42e2a_historical_unit_contract_no_longer_requires_global_full_coverage() -> None:
    source = (Path(__file__).resolve().parents[2] / "tests/unit/test_i4_2e_temple_ring_detail_qualification.py").read_text(encoding="utf-8")
    assert "test_i42e_track_evidence_requires_all_47_images" not in source
    assert "exact MVS image-index set" not in source
    assert "test_i42e_track_evidence_accepts_valid_visibility_subset" in source
