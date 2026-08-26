from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from core.reconstruction.errors import ReconstructionValidationError
from core.reconstruction.temple_ring_preflight import load_temple_ring_calibration

UNIT = Path(__file__).resolve().parents[1] / "unit/test_i4_2a_temple_ring_preflight.py"
_spec = importlib.util.spec_from_file_location("i42a_helpers_for_header_stress", UNIT)
assert _spec is not None and _spec.loader is not None
_helpers = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_helpers)
_dataset = _helpers._dataset


def _prepend(root: Path, header: str) -> Path:
    path = root / "templeR_par.txt"
    path.write_text(header + "\n" + path.read_text(encoding="utf-8"), encoding="utf-8")
    return path


@pytest.mark.parametrize("declared", ["0", "1", "46", "48", "999", "-47"])
def test_i42a1_rejects_wrong_integer_count_headers(tmp_path: Path, declared: str) -> None:
    path = _prepend(_dataset(tmp_path), declared)
    with pytest.raises(ReconstructionValidationError, match="count header must declare exactly 47"):
        load_temple_ring_calibration(path)


def test_i42a1_malformed_float_count_header_cannot_be_silently_accepted(tmp_path: Path) -> None:
    path = _prepend(_dataset(tmp_path), "47.0")
    with pytest.raises(ReconstructionValidationError, match="exactly 47 calibration rows"):
        load_temple_ring_calibration(path)


def test_i42a1_header_with_trailing_metadata_cannot_be_silently_accepted(tmp_path: Path) -> None:
    path = _prepend(_dataset(tmp_path), "47 views")
    with pytest.raises(ReconstructionValidationError, match="exactly 47 calibration rows"):
        load_temple_ring_calibration(path)
