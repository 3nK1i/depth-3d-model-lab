from __future__ import annotations

import pytest

from app.reconstruction_nerf_controls import (
    NERF_RENDER_PROFILES,
    NerfDisplaySettings,
)
from app.reconstruction_nerf_interactive import turntable_state_to_nerf_render_request


def test_95f_defaults_preserve_95e_render_and_authority_boundaries() -> None:
    settings = NerfDisplaySettings()
    assert settings.render_profile == "detail"
    assert (settings.render_width, settings.render_height) == (640, 360)
    assert settings.opacity_percent == 100
    assert settings.auto_refresh is True
    assert settings.visibility_only is True
    assert settings.canonical_geometry_authoritative is False
    assert settings.metric_scale_authoritative is False
    assert settings.measurement_authoritative is False
    assert settings.qc_authoritative is False
    assert settings.certified_metrology is False


def test_95f_render_profiles_are_bounded_16_by_9_presentation_choices() -> None:
    assert [(key, width, height) for key, _label, width, height in NERF_RENDER_PROFILES] == [
        ("fast", 320, 180),
        ("balanced", 480, 270),
        ("detail", 640, 360),
    ]
    for key, _label, width, height in NERF_RENDER_PROFILES:
        settings = NerfDisplaySettings(render_profile=key)
        assert (settings.render_width, settings.render_height) == (width, height)
        assert width / height == pytest.approx(16 / 9)


def test_95f_selected_render_detail_drives_only_request_resolution() -> None:
    settings = NerfDisplaySettings(render_profile="fast", opacity_percent=55, auto_refresh=False)
    request = turntable_state_to_nerf_render_request(
        request_id=5,
        center_viewer=(0.0, 0.0, 0.0),
        distance=2.0,
        azimuth_degrees=0.0,
        elevation_degrees=0.0,
        fov_degrees=45.0,
        scene_id="scene",
        artifact_id="artifact",
        width=settings.render_width,
        height=settings.render_height,
    )
    assert (request.width, request.height) == (320, 180)
    payload = request.as_protocol_payload()
    assert payload["scene_role"] == "appearance_only"
    assert payload["canonical_geometry_authoritative"] is False
    assert payload["measurement_authoritative"] is False
    assert settings.opacity_percent == 55
    assert settings.auto_refresh is False


def test_95f_invalid_settings_fail_closed() -> None:
    with pytest.raises(ValueError, match="profile"):
        NerfDisplaySettings(render_profile="ultra")
    with pytest.raises(ValueError, match="opacity"):
        NerfDisplaySettings(opacity_percent=20)
    with pytest.raises(ValueError, match="opacity"):
        NerfDisplaySettings(opacity_percent=120)


def test_95f_status_text_is_explicitly_presentation_oriented() -> None:
    settings = NerfDisplaySettings(render_profile="balanced", opacity_percent=70, auto_refresh=False)
    assert "Balanced" in settings.status_text
    assert "70%" in settings.status_text
    assert "manual refresh" in settings.status_text
