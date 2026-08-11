from __future__ import annotations

from types import SimpleNamespace

import pytest

from app.reconstruction_appearance_resources import (
    AppearanceResourceFallbackPresentation,
    build_appearance_resource_fallback_presentation,
    empty_appearance_resource_fallback_text,
)
from app.reconstruction_appearance_presentation import AppearanceRepresentationEntry
from app.reconstruction_representation_switching import RepresentationSwitchTarget
from tests.reconstruction_helpers import make_hardware


def _selected(kind: str = "gaussian_scene", label: str = "Gaussian") -> AppearanceRepresentationEntry:
    return AppearanceRepresentationEntry(
        artifact_id=f"f5-{kind}",
        representation_kind=kind,
        label=label,
        provider_id="external-provider",
        backend_id="external-provider",
        backend_version="1.0",
        render_capable=True,
        metric_geometry_status="arbitrary",
        state_text="Ready",
    )


def _target(kind: str = "gaussian_scene", *, ready: bool = True, state: str = "Preview ready") -> RepresentationSwitchTarget:
    label = "Gaussian" if kind == "gaussian_scene" else "NeRF"
    return RepresentationSwitchTarget(
        artifact_id=f"f5-{kind}",
        representation_kind=kind,
        label=label,
        ready=ready,
        state_text=state,
    )


def _scene(*, fallback=()):
    return SimpleNamespace(
        session_id="f5-session",
        scene_id="f5-scene",
        fallback_evidence=tuple(fallback),
        canonical_geometry_authoritative=True,
        certified_metrology=False,
    )


def test_f5_cpu_does_not_block_existing_static_preview() -> None:
    presentation = build_appearance_resource_fallback_presentation(
        _scene(),
        selected=_selected(),
        switch_target=_target(ready=True),
        active_view="canonical_geometry",
        hardware=make_hardware(total_gib=None),
    )
    assert presentation.preview_ready is True
    assert presentation.fallback_active is False
    assert "CPU" in presentation.resources_text
    assert "static previews remain supported" in presentation.resources_text
    assert "static preview ready" in presentation.presentation_text


def test_f5_active_appearance_view_keeps_canonical_authority_explicit() -> None:
    presentation = build_appearance_resource_fallback_presentation(
        _scene(),
        selected=_selected("nerf_scene", "NeRF"),
        switch_target=_target("nerf_scene", ready=True),
        active_view="nerf_scene",
        hardware=make_hardware(total_gib=6.0, free_gib=4.5, name="RTX 2060"),
    )
    assert presentation.active_view == "nerf_scene"
    assert presentation.fallback_active is False
    assert "no neural runtime required" in presentation.presentation_text
    assert presentation.appearance_canonical_geometry_authoritative is False
    assert presentation.certified_metrology is False
    assert presentation.authority_text == "Appearance only • canonical geometry remains authoritative"


def test_f5_unavailable_selected_target_fails_closed_to_canonical() -> None:
    presentation = build_appearance_resource_fallback_presentation(
        _scene(),
        selected=_selected("nerf_scene", "NeRF"),
        switch_target=_target("nerf_scene", ready=False, state="NeRF backend preview render is unavailable"),
        active_view="canonical_geometry",
        hardware=make_hardware(total_gib=32.0, free_gib=28.0, name="RTX 5090", capability=(12, 0)),
    )
    assert presentation.fallback_active is True
    assert presentation.active_view == "canonical_geometry"
    assert "Fallback active" in presentation.state_text
    assert "preview render is unavailable" in presentation.state_text
    assert presentation.fallback_text == "Canonical geometry • no neural cross-fallback"
    assert presentation.cross_representation_fallback is False


def test_f5_display_failure_reason_overrides_ready_target_and_retains_canonical() -> None:
    presentation = build_appearance_resource_fallback_presentation(
        _scene(),
        selected=_selected(),
        switch_target=_target(ready=True),
        active_view="canonical_geometry",
        failure_reason="Gaussian preview image could not be displayed",
        hardware=make_hardware(total_gib=6.0),
    )
    assert presentation.preview_ready is True
    assert presentation.fallback_active is True
    assert presentation.state_text == "Fallback active • Gaussian preview image could not be displayed"
    assert "could not be presented" in presentation.state_details


def test_f5_surfaces_persisted_appearance_pipeline_degradation() -> None:
    evidence = SimpleNamespace(
        stage=SimpleNamespace(value="gaussian_appearance"),
        state=SimpleNamespace(value="skipped"),
        provider_attempts=("external-gaussian",),
        fallback_used=False,
        reason="resource_profile_unavailable",
    )
    presentation = build_appearance_resource_fallback_presentation(
        _scene(fallback=(evidence,)),
        hardware=make_hardware(total_gib=6.0),
    )
    assert presentation.fallback_active is True
    assert "Gaussian appearance" in presentation.pipeline_text
    assert "resource_profile_unavailable" in (presentation.pipeline_text + presentation.pipeline_details)
    assert presentation.active_view == "canonical_geometry"


def test_f5_gpu_snapshot_is_advisory_and_readable() -> None:
    presentation = build_appearance_resource_fallback_presentation(
        _scene(),
        selected=_selected(),
        switch_target=_target(ready=True),
        hardware=make_hardware(total_gib=32.0, free_gib=25.5, name="NVIDIA GeForce RTX 5090", capability=(12, 0)),
    )
    assert "RTX 5090" in presentation.device_text
    assert "25.5/32.0 GB startup free/total" in presentation.resources_text
    assert presentation.resource_snapshot_available is True


def test_f5_contract_rejects_cross_representation_or_authority_promotion() -> None:
    common = dict(
        device_text="CPU",
        resources_text="CPU",
        resource_details="",
        presentation_text="Canonical",
        fallback_text="Canonical geometry",
        pipeline_text="None",
        pipeline_details="",
        state_text="Ready",
        state_details="",
        selected_kind="gaussian_scene",
        active_view="canonical_geometry",
        preview_ready=False,
        fallback_active=True,
        resource_snapshot_available=True,
    )
    with pytest.raises(ValueError):
        AppearanceResourceFallbackPresentation(**common, cross_representation_fallback=True)
    with pytest.raises(ValueError):
        AppearanceResourceFallbackPresentation(**common, appearance_canonical_geometry_authoritative=True)
    with pytest.raises(ValueError):
        AppearanceResourceFallbackPresentation(**common, certified_metrology=True)


def test_f5_empty_text_is_explicit_and_safe() -> None:
    values = empty_appearance_resource_fallback_text()
    assert values["fallback"] == "Canonical geometry • no neural cross-fallback"
    assert "Waiting" in values["state"]
