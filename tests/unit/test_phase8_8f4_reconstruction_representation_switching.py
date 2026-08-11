from __future__ import annotations

from types import SimpleNamespace

import pytest

from app.reconstruction_representation_switching import (
    RepresentationSwitchTarget,
    build_representation_switch_target,
    canonical_view_text,
)


def _selected(kind: str, artifact_id: str, label: str):
    return SimpleNamespace(representation_kind=kind, artifact_id=artifact_id, label=label)


def _presentation(artifact_id: str, *, ready: bool, state: str = "ready"):
    return SimpleNamespace(artifact_id=artifact_id, ready=ready, state_text=state)


def test_f4_builds_ready_gaussian_target_from_matching_f2_presentation() -> None:
    target = build_representation_switch_target(
        _selected("gaussian_scene", "gaussian-a", "Gaussian"),
        gaussian_presentation=_presentation("gaussian-a", ready=True, state="Gaussian preview ready"),
        nerf_presentation=_presentation("nerf-a", ready=True),
    )
    assert target is not None
    assert target.representation_kind == "gaussian_scene"
    assert target.artifact_id == "gaussian-a"
    assert target.ready is True
    assert target.active_text == "Gaussian appearance • appearance only"
    assert target.canonical_geometry_authoritative is False
    assert target.certified_metrology is False


def test_f4_builds_ready_nerf_target_from_matching_f3_presentation() -> None:
    target = build_representation_switch_target(
        _selected("nerf_scene", "nerf-a", "NeRF"),
        gaussian_presentation=_presentation("gaussian-a", ready=True),
        nerf_presentation=_presentation("nerf-a", ready=True, state="NeRF preview ready"),
    )
    assert target is not None
    assert target.representation_kind == "nerf_scene"
    assert target.ready is True
    assert target.active_text == "NeRF appearance • appearance only"


def test_f4_never_falls_back_to_other_representation_when_selected_target_is_unavailable() -> None:
    target = build_representation_switch_target(
        _selected("gaussian_scene", "gaussian-a", "Gaussian"),
        gaussian_presentation=_presentation("gaussian-a", ready=False, state="Gaussian preview missing"),
        nerf_presentation=_presentation("nerf-a", ready=True, state="NeRF preview ready"),
    )
    assert target is not None
    assert target.representation_kind == "gaussian_scene"
    assert target.ready is False
    assert target.state_text == "Gaussian preview missing"
    assert target.unavailable_text == "Gaussian unavailable • canonical geometry retained"


def test_f4_requires_artifact_identity_match_before_switch_is_ready() -> None:
    target = build_representation_switch_target(
        _selected("nerf_scene", "nerf-selected", "NeRF"),
        nerf_presentation=_presentation("nerf-other", ready=True),
    )
    assert target is not None
    assert target.ready is False
    assert "unavailable" in target.state_text.lower()


def test_f4_ignores_unknown_future_representation_instead_of_special_casing_objects() -> None:
    assert build_representation_switch_target(
        _selected("future_appearance_scene", "future-a", "Future")
    ) is None


def test_f4_target_cannot_claim_geometry_or_metrology_authority() -> None:
    with pytest.raises(ValueError):
        RepresentationSwitchTarget(
            artifact_id="a",
            representation_kind="gaussian_scene",
            label="Gaussian",
            ready=True,
            state_text="ready",
            canonical_geometry_authoritative=True,
        )
    with pytest.raises(ValueError):
        RepresentationSwitchTarget(
            artifact_id="a",
            representation_kind="nerf_scene",
            label="NeRF",
            ready=True,
            state_text="ready",
            certified_metrology=True,
        )


def test_f4_canonical_view_text_is_explicit() -> None:
    assert canonical_view_text() == "Canonical geometry"
