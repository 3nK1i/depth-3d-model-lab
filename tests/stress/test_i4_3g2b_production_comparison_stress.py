from __future__ import annotations

from pathlib import Path
import pytest

from core.reconstruction.errors import ReconstructionValidationError
from core.reconstruction.mvs_da3_production_comparison import build_g2b_runtime_preflight_snapshot
from tests.i4_3g2b_helpers import da3_runtime, g1_snapshot, g2a_snapshot, make_source, mvs_runtime


@pytest.mark.parametrize("scenario", [
    "g2_complete_room_mvs_da3",
    "g2_temple_ring_mvs_da3",
    "g2_low_texture_mvs_da3",
    "g2_weak_overlap_mvs_da3",
])
def test_all_fresh_g2_scenarios_fail_closed_and_preflight_without_execution(tmp_path: Path, scenario: str) -> None:
    source = make_source(tmp_path / scenario, count=12)
    snapshot = build_g2b_runtime_preflight_snapshot(
        i4_3g1_snapshot=g1_snapshot(), i4_3g2a_snapshot=g2a_snapshot(source), scenario_id=scenario,
        mvs_runtime=mvs_runtime(), da3_runtime=da3_runtime(), verify_files=True,
    )
    assert snapshot["pass"] is True
    assert snapshot["scope"] == {
        "runtime_preflight_only": True,
        "source_identity_verified": True,
        "authoritative_sfm_started": False,
        "mvs_started": False,
        "da3_started": False,
        "provider_quality_results_consumed": False,
        "matched_view_diagnostic_started": False,
    }


def test_repeated_source_tampering_never_authorizes_execution(tmp_path: Path) -> None:
    source = make_source(tmp_path / "images", count=20)
    g2a = g2a_snapshot(source)
    for index in range(5):
        (source / f"frame_{index:03d}.png").write_bytes(f"tamper-{index}".encode())
        with pytest.raises(ReconstructionValidationError):
            build_g2b_runtime_preflight_snapshot(
                i4_3g1_snapshot=g1_snapshot(), i4_3g2a_snapshot=g2a,
                scenario_id="g2_complete_room_mvs_da3", mvs_runtime=mvs_runtime(), da3_runtime=da3_runtime(), verify_files=True,
            )
