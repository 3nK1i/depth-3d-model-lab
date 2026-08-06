from pathlib import Path
import json
import numpy as np
import pytest

from core.contracts.validation import ContractValidationError
from core.inspection import load_inspection_bundle
from tests.inspection_helpers import write_inspection_bundle


def test_bundle_loader_returns_validated_summary_and_dense_boolean_masks(tmp_path: Path):
    bundle = write_inspection_bundle(tmp_path / "bundle")
    summary, arrays = load_inspection_bundle(bundle)
    assert summary.objects[0].class_name == "box"
    assert summary.state_path.is_file()
    assert arrays[summary.objects[0].mask_key].dtype == np.bool_
    assert arrays["aligned_rgb"].shape == (48, 64, 3)


def test_bundle_loader_rejects_missing_required_arrays(tmp_path: Path):
    bundle = write_inspection_bundle(tmp_path / "bundle")
    archive = np.load(bundle / "inspection_arrays.npz", allow_pickle=False)
    arrays = {key: np.array(archive[key]) for key in archive.files if key != "object_000_mask"}
    archive.close()
    np.savez_compressed(bundle / "inspection_arrays.npz", **arrays)
    with pytest.raises(ContractValidationError, match="incomplete"):
        load_inspection_bundle(bundle)


def test_bundle_loader_rejects_unsupported_schema(tmp_path: Path):
    bundle = write_inspection_bundle(tmp_path / "bundle")
    path = bundle / "inspection_bundle.json"
    payload = json.loads(path.read_text())
    payload["schema_version"] = 99
    path.write_text(json.dumps(payload))
    with pytest.raises(ContractValidationError, match="schema"):
        load_inspection_bundle(bundle)
