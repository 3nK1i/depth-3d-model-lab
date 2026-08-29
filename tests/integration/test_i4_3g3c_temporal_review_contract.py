from __future__ import annotations

import json

import pytest

from core.reconstruction.errors import ReconstructionValidationError
from core.reconstruction.temporal_provider_matched_view_review import (
    I43G3C_SCENARIOS,
    load_depth_stack,
    load_g3b_matched_view_sources,
    rolling_depth_display_multiplier,
    sha256_file,
)
from core.reconstruction.temporal_provider_production_qualification import read_json_object
from tests.i4_3g3c_helpers import g3c_fixture


def test_g3c_loads_frozen_terminal_g3b_evidence_without_rerunning_or_overriding_rejection(tmp_path):
    fixture = g3c_fixture(
        tmp_path,
        scenario_id="g3_temporal_motion",
        rolling_status="rejected",
        vda_status="degraded",
    )
    sources = load_g3b_matched_view_sources(
        fixture["production_path"],
        expected_sha256=fixture["production_sha256"],
        scenario_id="g3_temporal_motion",
    )
    assert sources.scenario_id == "g3_temporal_motion"
    assert sources.rolling_depth.status == "rejected"
    assert sources.video_depth_anything.status == "degraded"
    assert sources.rolling_depth.ordered_source_sequence_indices == (10, 11, 12)
    assert sources.rolling_depth.ordered_source_sequence_indices == sources.video_depth_anything.ordered_source_sequence_indices
    assert sources.sfm_frame_id == "g3c:g3_temporal_motion:pycolmap"

    rd = load_depth_stack(sources.rolling_depth.depth_npz_path, expected_view_count=3, label="RollingDepth")
    vda = load_depth_stack(sources.video_depth_anything.depth_npz_path, expected_view_count=3, label="VDA")
    assert rd.shape == vda.shape == (3, 12, 16)

    summary = read_json_object(sources.rolling_depth.summary_path, label="RollingDepth summary")
    assert rolling_depth_display_multiplier(summary) == -1.0


def test_g3c_scenario_contract_remains_exactly_the_three_temporal_scenarios():
    assert I43G3C_SCENARIOS == (
        "g3_temporal_static_sweep",
        "g3_temporal_motion",
        "g3_temporal_low_texture",
    )


def test_g3c_accepts_pre_terminal_outcome_field_clean_g3b_snapshot(tmp_path):
    fixture = g3c_fixture(
        tmp_path,
        scenario_id="g3_temporal_static_sweep",
        rolling_status="degraded",
        vda_status="pass",
    )
    production_path = fixture["production_path"]
    payload = json.loads(production_path.read_text(encoding="utf-8"))
    payload["decision"].pop("provider_terminal_outcomes_complete")
    production_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    sources = load_g3b_matched_view_sources(
        production_path,
        expected_sha256=sha256_file(production_path),
        scenario_id="g3_temporal_static_sweep",
    )
    assert sources.rolling_depth.status == "degraded"
    assert sources.video_depth_anything.status == "pass"


def test_g3c_missing_terminal_outcome_field_still_fails_closed_for_rejection_snapshot(tmp_path):
    fixture = g3c_fixture(
        tmp_path,
        scenario_id="g3_temporal_motion",
        rolling_status="rejected",
        vda_status="degraded",
    )
    production_path = fixture["production_path"]
    payload = json.loads(production_path.read_text(encoding="utf-8"))
    payload["decision"].pop("provider_terminal_outcomes_complete")
    production_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    with pytest.raises(
        ReconstructionValidationError,
        match="requires complete G3B provider terminal outcomes",
    ):
        load_g3b_matched_view_sources(
            production_path,
            expected_sha256=sha256_file(production_path),
            scenario_id="g3_temporal_motion",
        )
