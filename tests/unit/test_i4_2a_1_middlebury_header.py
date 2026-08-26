from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from core.reconstruction.errors import ReconstructionValidationError
from core.reconstruction.temple_ring_preflight import (
    TEMPLE_RING_EXPECTED_VIEWS,
    inspect_temple_ring_dataset,
    load_temple_ring_calibration,
)

UNIT = Path(__file__).resolve().parents[1] / "unit/test_i4_2a_temple_ring_preflight.py"
_spec = importlib.util.spec_from_file_location("i42a_helpers_for_header", UNIT)
assert _spec is not None and _spec.loader is not None
_helpers = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_helpers)
_dataset = _helpers._dataset
_calibration_line = _helpers._calibration_line


def _add_count_header(root: Path, value: str = "47") -> Path:
    path = root / "templeR_par.txt"
    original = path.read_text(encoding="utf-8")
    path.write_text(f"{value}\n{original}", encoding="utf-8")
    return path


def test_canonical_middlebury_count_header_is_accepted(tmp_path: Path) -> None:
    root = _dataset(tmp_path)
    _add_count_header(root)
    records = load_temple_ring_calibration(root / "templeR_par.txt")
    assert len(records) == TEMPLE_RING_EXPECTED_VIEWS
    assert records[0].image_name == "templeR0001.png"
    assert records[-1].image_name == "templeR0047.png"


def test_dataset_evidence_records_canonical_header_metadata(tmp_path: Path) -> None:
    root = _dataset(tmp_path)
    _add_count_header(root)
    dataset = inspect_temple_ring_dataset(root)
    assert dataset["calibration_count_header_present"] is True
    assert dataset["calibration_declared_view_count"] == TEMPLE_RING_EXPECTED_VIEWS
    assert dataset["validated_calibration_rows"] == TEMPLE_RING_EXPECTED_VIEWS


def test_headerless_exact_47_row_form_remains_backward_compatible(tmp_path: Path) -> None:
    dataset = inspect_temple_ring_dataset(_dataset(tmp_path))
    assert dataset["calibration_count_header_present"] is False
    assert dataset["calibration_declared_view_count"] is None
    assert dataset["validated_calibration_rows"] == TEMPLE_RING_EXPECTED_VIEWS


def test_wrong_declared_count_fails_closed(tmp_path: Path) -> None:
    root = _dataset(tmp_path)
    _add_count_header(root, "46")
    with pytest.raises(ReconstructionValidationError, match="count header must declare exactly 47"):
        load_temple_ring_calibration(root / "templeR_par.txt")


def test_declared_47_with_only_46_rows_fails_closed(tmp_path: Path) -> None:
    root = _dataset(tmp_path)
    path = _add_count_header(root)
    lines = path.read_text(encoding="utf-8").splitlines()
    path.write_text("\n".join(lines[:-1]) + "\n", encoding="utf-8")
    with pytest.raises(ReconstructionValidationError, match="exactly 47 calibration rows after the count header; found 46"):
        load_temple_ring_calibration(path)


def test_noncanonical_47_name_set_is_rejected_even_if_count_matches(tmp_path: Path) -> None:
    root = _dataset(tmp_path)
    path = _add_count_header(root)
    lines = path.read_text(encoding="utf-8").splitlines()
    lines[-1] = _calibration_line("templeR0048.png", 47)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    with pytest.raises(ReconstructionValidationError, match="exact canonical"):
        load_temple_ring_calibration(path)
