"""Bounded deterministic final-readiness repetition for Phase 8.7G4."""
from __future__ import annotations

from pathlib import Path

from tests.reconstruction_g3_helpers import clean_evidence, optional_degradation_evidence


def test_g4_repeated_clean_runs_after_degradation_keep_unique_ready_scenes(tmp_path: Path) -> None:
    clean_scene_ids: set[str] = set()
    for index in range(10):
        if index % 3 == 1:
            degraded = optional_degradation_evidence(index, tmp_path)
            assert degraded.pipeline_status == "degraded"
            assert degraded.canonical_geometry_authoritative is True
            continue
        clean = clean_evidence(index)
        assert clean.pipeline_status == "completed"
        assert clean.persistent_scene_state == "ready"
        assert clean.canonical_geometry_authoritative is True
        assert clean.gaussian_authority == "appearance_only"
        assert clean.nerf_authority == "appearance_only"
        assert clean.scene_id not in clean_scene_ids
        clean_scene_ids.add(clean.scene_id)
    assert len(clean_scene_ids) == 7
