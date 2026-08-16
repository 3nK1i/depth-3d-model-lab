from __future__ import annotations

import pytest

from app.reconstruction_digital_twin_resources import (
    DigitalTwinResourceOwner,
    build_digital_twin_resource_coordination,
)
from tests.reconstruction_helpers import make_hardware


def test_96g_idle_and_static_previews_do_not_reserve_heavy_resources() -> None:
    state = build_digital_twin_resource_coordination(
        processing_active=False,
        reconstruction_active=False,
        interactive_nerf_active=False,
        interactive_gaussian_resident=False,
        static_preview_active=True,
        hardware=make_hardware(total_gib=6.0, free_gib=4.5, name="NVIDIA GeForce RTX 2060"),
    )
    assert state.owner is DigitalTwinResourceOwner.IDLE
    assert state.page_resource_active is False
    assert state.heavy_runtime_active is False
    assert state.static_preview_active is True
    assert "no dynamic Digital Twin owner" in state.owner_text
    assert "startup 4.5/6.0 GB" in state.startup_resources_text
    assert state.can_start_interactive_nerf is True
    assert state.can_start_interactive_gaussian is True


def test_96g_build_and_interactive_nerf_are_application_visible_resource_owners() -> None:
    build = build_digital_twin_resource_coordination(
        processing_active=True,
        reconstruction_active=False,
        interactive_nerf_active=False,
        interactive_gaussian_resident=False,
        build_label="Build Full Scene",
        hardware=make_hardware(total_gib=6.0),
    )
    assert build.owner is DigitalTwinResourceOwner.DIGITAL_TWIN_BUILD
    assert build.page_resource_active is True
    assert build.can_start_interactive_nerf is False
    assert "Cancel the active Digital Twin build" in build.interactive_admission_failure("interactive NeRF")

    nerf = build_digital_twin_resource_coordination(
        processing_active=False,
        reconstruction_active=False,
        interactive_nerf_active=True,
        interactive_gaussian_resident=False,
        hardware=make_hardware(total_gib=6.0),
    )
    assert nerf.owner is DigitalTwinResourceOwner.INTERACTIVE_NERF
    assert nerf.page_resource_active is True
    assert "external GPU runtime" in nerf.owner_text


def test_96g_gaussian_is_bounded_page_residency_not_external_heavy_owner() -> None:
    state = build_digital_twin_resource_coordination(
        processing_active=False,
        reconstruction_active=False,
        interactive_nerf_active=False,
        interactive_gaussian_resident=True,
        hardware=make_hardware(total_gib=6.0),
    )
    assert state.owner is DigitalTwinResourceOwner.IDLE
    assert state.heavy_runtime_active is False
    assert state.page_resource_active is True
    assert "Gaussian display resident" in state.owner_text


def test_96g_multiple_heavy_owners_fail_closed_without_authority_promotion() -> None:
    state = build_digital_twin_resource_coordination(
        processing_active=True,
        reconstruction_active=False,
        interactive_nerf_active=True,
        interactive_gaussian_resident=False,
        hardware=make_hardware(total_gib=6.0),
    )
    assert state.conflict is True
    assert "Concurrent heavyweight owners" in state.conflict_reason
    assert state.can_start_interactive_nerf is False
    assert state.canonical_geometry_authoritative is False
    assert state.metric_scale_authoritative is False
    assert state.measurement_authoritative is False
    assert state.qc_authoritative is False
    assert state.certified_metrology is False
    with pytest.raises(ValueError, match="cannot claim"):
        type(state)(
            owner=state.owner,
            device_text=state.device_text,
            startup_resources_text=state.startup_resources_text,
            processing_active=True,
            reconstruction_active=False,
            interactive_nerf_active=False,
            interactive_gaussian_resident=False,
            static_preview_active=False,
            canonical_geometry_authoritative=True,
        )
