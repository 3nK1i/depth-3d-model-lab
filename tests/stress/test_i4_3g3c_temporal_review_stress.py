from __future__ import annotations

import json

import pytest

from core.reconstruction.errors import ReconstructionValidationError
from core.reconstruction.temporal_provider_matched_view_review import (
    load_g3b_matched_view_sources,
    sha256_file,
)
from tests.i4_3g3c_helpers import g3c_fixture


def _rewrite(path, payload):
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return sha256_file(path)


def test_g3c_fails_closed_on_authority_drift(tmp_path):
    fixture = g3c_fixture(tmp_path)
    payload = json.loads(fixture["production_path"].read_text(encoding="utf-8"))
    payload["authority"]["metric_scale_claimed"] = True
    sha = _rewrite(fixture["production_path"], payload)
    with pytest.raises(ReconstructionValidationError):
        load_g3b_matched_view_sources(
            fixture["production_path"],
            expected_sha256=sha,
            scenario_id=fixture["scenario_id"],
        )


def test_g3c_fails_closed_when_provider_order_is_not_common(tmp_path):
    fixture = g3c_fixture(tmp_path)
    production = json.loads(fixture["production_path"].read_text(encoding="utf-8"))
    vda_ref = production["providers"]["video_depth_anything"]["evidence"]
    vda_path = __import__("pathlib").Path(vda_ref["path"])
    vda = json.loads(vda_path.read_text(encoding="utf-8"))
    vda["temporal_sequence"]["ordered_source_sequence_indices"] = [10, 12, 11]
    new_vda_sha = _rewrite(vda_path, vda)
    production["providers"]["video_depth_anything"]["evidence"]["sha256"] = new_vda_sha
    new_prod_sha = _rewrite(fixture["production_path"], production)
    with pytest.raises(ReconstructionValidationError):
        load_g3b_matched_view_sources(
            fixture["production_path"],
            expected_sha256=new_prod_sha,
            scenario_id=fixture["scenario_id"],
        )


def test_g3c_detects_tampered_native_depth_artifact(tmp_path):
    fixture = g3c_fixture(tmp_path)
    production = json.loads(fixture["production_path"].read_text(encoding="utf-8"))
    rd_path = __import__("pathlib").Path(production["providers"]["rolling_depth"]["evidence"]["path"])
    rd = json.loads(rd_path.read_text(encoding="utf-8"))
    depth_ref = next(item for item in rd["artifacts"] if item["path"].endswith("rolling_depth_evidence.npz"))
    depth_path = __import__("pathlib").Path(depth_ref["path"])
    depth_path.write_bytes(depth_path.read_bytes() + b"tamper")
    with pytest.raises(ReconstructionValidationError):
        load_g3b_matched_view_sources(
            fixture["production_path"],
            expected_sha256=fixture["production_sha256"],
            scenario_id=fixture["scenario_id"],
        )
