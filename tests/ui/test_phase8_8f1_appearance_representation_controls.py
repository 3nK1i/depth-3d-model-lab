from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from app.main_window import MainWindow

SESSION = "12121212-1212-4121-8121-121212121212"


@dataclass
class _Artifact:
    artifact_id: str
    kind_name: str
    provider_id: str
    backend_id: str
    backend_version: str
    state_name: str

    @property
    def kind(self):
        return SimpleNamespace(value=self.kind_name)

    @property
    def authority(self):
        return SimpleNamespace(value="appearance_only")

    @property
    def canonical_geometry_authoritative(self):
        return False

    def as_dict(self):
        state_key = "optimization_state" if self.kind_name == "gaussian_scene" else "training_state"
        return {
            "artifact_id": self.artifact_id,
            "kind": self.kind_name,
            "authority": "appearance_only",
            "provider_id": self.provider_id,
            "canonical_geometry_authoritative": False,
            "certified_metrology": False,
            "metadata": {
                "backend_id": self.backend_id,
                "backend_version": self.backend_version,
                "render_capable": True,
                "metric_geometry_status": "arbitrary",
            },
            "payload_snapshot": {
                "backend_id": self.backend_id,
                "backend_version": self.backend_version,
                state_key: self.state_name,
                "render_capable": True,
                "metric_geometry_status": "arbitrary",
            },
        }


def _scene():
    return SimpleNamespace(
        scene_id="prs-f1-ui",
        session_id=SESSION,
        state=SimpleNamespace(value="ready"),
        artifacts=(
            _Artifact("f1-ui-gaussian", "gaussian_scene", "gaussian-provider", "eia-gsplat-external", "1.5.3", "optimized"),
            _Artifact("f1-ui-nerf", "nerf_scene", "nerf-provider", "nerfstudio-nerfacto", "1.1.5", "trained"),
        ),
        canonical_geometry_authoritative=True,
        certified_metrology=False,
    )


@pytest.fixture
def window(qtbot, monkeypatch):
    monkeypatch.setattr("app.application_controller.CameraEnumerator.start", lambda self: None)
    win = MainWindow()
    qtbot.addWidget(win)
    yield win
    win.close()


def test_f1_group_is_idle_and_disabled_without_persistent_scene(window) -> None:
    page = window._reconstruction_page
    assert page._appearance_group.isHidden() is False
    assert page._appearance_representation_combo.isEnabled() is False
    assert page._appearance_available_value.text() == "No persistent appearance scene"
    assert page.appearance_representation_presentation is None


def test_f1_persistent_scene_exposes_gaussian_and_nerf_metadata_controls(window) -> None:
    page = window._reconstruction_page
    before_viewport = page._viewport_state.text()
    page.set_persistent_scene(_scene(), "persistent_reconstruction_scene.json")
    presentation = page.appearance_representation_presentation
    assert presentation is not None
    assert page._appearance_representation_combo.isEnabled() is True
    assert page._appearance_representation_combo.count() == 2
    assert page._appearance_available_value.text() == "Gaussian • NeRF"
    assert page.selected_appearance_representation is not None
    assert page.selected_appearance_representation.label == "Gaussian"
    assert "eia-gsplat-external" in page._appearance_provider_value.text()
    assert page._appearance_render_value.text() == "Declared render-capable"
    assert page._viewport_state.text() == before_viewport

    page._appearance_representation_combo.setCurrentIndex(1)
    assert page.selected_appearance_representation is not None
    assert page.selected_appearance_representation.label == "NeRF"
    assert "nerfstudio-nerfacto" in page._appearance_provider_value.text()
    assert page._viewport_state.text() == before_viewport
    assert page._appearance_authority_value.text() == "Appearance only • canonical geometry remains authoritative"
    assert page._appearance_authority_value.minimumHeight() >= (
        page._appearance_authority_value.fontMetrics().lineSpacing() * 3
    )


def test_f1_clear_persistent_scene_clears_appearance_controls(window) -> None:
    page = window._reconstruction_page
    page.set_persistent_scene(_scene(), "persistent_reconstruction_scene.json")
    assert page.appearance_representation_presentation is not None
    page.set_persistent_scene(None, "")
    assert page.appearance_representation_presentation is None
    assert page.selected_appearance_representation is None
    assert page._appearance_representation_combo.isEnabled() is False
    assert page._appearance_available_value.text() == "No persistent appearance scene"
