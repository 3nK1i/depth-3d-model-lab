from __future__ import annotations

from pathlib import Path

import pytest

from core.reconstruction.errors import ReconstructionValidationError
from core.reconstruction.single_image_provider_matched_view_review import (
    I43G4C_PROVIDER_IMPLEMENTATIONS,
    load_g4b_matched_view_sources,
)
from tests.i4_3g4c_helpers import synthetic_g4b_snapshot


def test_load_complete_room_frozen_six_provider_sources(tmp_path: Path) -> None:
    path, digest = synthetic_g4b_snapshot(tmp_path / "complete")
    sources = load_g4b_matched_view_sources(
        path,
        expected_sha256=digest,
        scenario_id="g4_single_image_complete_room",
    )
    assert sources.common_keyframe_ids == ("kf-0", "kf-1")
    assert set(sources.providers) == set(I43G4C_PROVIDER_IMPLEMENTATIONS)
    assert all(source.canonical_geometry_path is not None for source in sources.providers.values())
    assert all(not source.native_depth_artifacts for source in sources.providers.values())


def test_low_texture_rejected_providers_remain_native_only(tmp_path: Path) -> None:
    rejected = ("depth_anything_v2", "marigold")
    path, digest = synthetic_g4b_snapshot(
        tmp_path / "low",
        scenario_id="g4_single_image_low_texture",
        rejected=rejected,
    )
    sources = load_g4b_matched_view_sources(
        path,
        expected_sha256=digest,
        scenario_id="g4_single_image_low_texture",
    )
    for implementation in rejected:
        assert sources.providers[implementation].status == "rejected"
        assert sources.providers[implementation].canonical_geometry_path is None
        assert set(sources.providers[implementation].native_depth_artifacts) == {"kf-0", "kf-1"}


def test_tampered_rejected_native_artifact_fails_closed(tmp_path: Path) -> None:
    path, digest = synthetic_g4b_snapshot(
        tmp_path / "tamper",
        scenario_id="g4_single_image_low_texture",
        rejected=("depth_anything_v2", "marigold"),
    )
    frame = next((tmp_path / "tamper" / "depth_anything_v2" / "worker").glob("frame_*.npz"))
    frame.write_bytes(frame.read_bytes() + b"tamper")
    with pytest.raises(ReconstructionValidationError, match="SHA256 mismatch"):
        load_g4b_matched_view_sources(
            path,
            expected_sha256=digest,
            scenario_id="g4_single_image_low_texture",
        )
