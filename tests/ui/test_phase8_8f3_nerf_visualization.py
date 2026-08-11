from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from types import SimpleNamespace

from PIL import Image, ImageDraw
import pytest

from app.main_window import MainWindow

SESSION = "78787878-7878-4787-8787-787878787878"


@dataclass
class _NerfArtifact:
    artifact_id: str
    artifact_dir: str

    @property
    def kind(self):
        return SimpleNamespace(value="nerf_scene")

    @property
    def authority(self):
        return SimpleNamespace(value="appearance_only")

    @property
    def canonical_geometry_authoritative(self):
        return False

    def as_dict(self):
        return {
            "artifact_id": self.artifact_id,
            "kind": "nerf_scene",
            "authority": "appearance_only",
            "provider_id": "nerf-provider",
            "canonical_geometry_authoritative": False,
            "certified_metrology": False,
            "metadata": {
                "backend_id": "external-nerf-provider",
                "backend_version": "1.1.5",
                "artifact_dir": self.artifact_dir,
                "render_capable": True,
                "metric_geometry_status": "arbitrary",
            },
            "payload_snapshot": {
                "backend_id": "external-nerf-provider",
                "backend_version": "1.1.5",
                "artifact_dir": self.artifact_dir,
                "training_state": "trained",
                "input_frame_count": 24,
                "training_steps": 1000,
                "render_capable": True,
                "metric_geometry_status": "arbitrary",
            },
        }


@dataclass
class _GaussianArtifact:
    artifact_id: str
    artifact_dir: str

    @property
    def kind(self):
        return SimpleNamespace(value="gaussian_scene")

    @property
    def authority(self):
        return SimpleNamespace(value="appearance_only")

    @property
    def canonical_geometry_authoritative(self):
        return False

    def as_dict(self):
        return {
            "artifact_id": self.artifact_id,
            "kind": "gaussian_scene",
            "authority": "appearance_only",
            "provider_id": "gaussian-provider",
            "canonical_geometry_authoritative": False,
            "certified_metrology": False,
            "metadata": {
                "backend_id": "external-gaussian-provider",
                "backend_version": "1.5.3",
                "artifact_dir": self.artifact_dir,
                "gaussian_count": 1024,
                "render_capable": True,
                "metric_geometry_status": "arbitrary",
            },
            "payload_snapshot": {
                "backend_id": "external-gaussian-provider",
                "backend_version": "1.5.3",
                "artifact_dir": self.artifact_dir,
                "gaussian_count": 1024,
                "optimization_state": "optimized",
                "render_capable": True,
                "metric_geometry_status": "arbitrary",
            },
        }


def _preview(root: Path, *, label: str) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", (640, 360), (8, 12, 18))
    draw = ImageDraw.Draw(image)
    draw.rectangle((145, 90, 495, 285), outline=(90, 155, 220), width=5)
    draw.ellipse((250, 115, 390, 255), fill=(70, 120, 185))
    draw.text((18, 18), label, fill=(205, 215, 225))
    path = root / "preview.png"
    image.save(path)
    return path


def _nerf_scene(root: Path):
    artifact = _NerfArtifact("f3-ui-nerf", str(root.resolve()))
    return SimpleNamespace(
        scene_id="prs-f3-ui",
        session_id=SESSION,
        state=SimpleNamespace(value="ready"),
        artifacts=(artifact,),
        nerf_artifact=artifact,
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


def test_f3_group_is_safe_without_nerf_scene(window) -> None:
    page = window._reconstruction_page
    assert page._nerf_visualization_group.isHidden() is False
    assert page.nerf_visualization_presentation is None
    assert page._nerf_show_button.isEnabled() is False
    assert page.nerf_preview_visible is False


def test_f3_shows_and_hides_backend_authored_nerf_preview(window, tmp_path: Path, qtbot) -> None:
    root = (tmp_path / "nerf").resolve()
    preview = _preview(root, label="F3 NeRF preview")
    (root / "e2_runtime_evidence.json").write_text(
        json.dumps({"preview_path": str(preview.resolve()), "training_completed_without_oom": True}),
        encoding="utf-8",
    )
    page = window._reconstruction_page
    page.set_persistent_scene(_nerf_scene(root), "persistent_reconstruction_scene.json")
    presentation = page.nerf_visualization_presentation
    assert presentation is not None and presentation.ready
    assert page.selected_appearance_representation is not None
    assert page.selected_appearance_representation.label == "NeRF"
    assert page._nerf_show_button.isEnabled() is True

    page._nerf_show_button.click()
    qtbot.wait(10)
    assert page.nerf_preview_visible is True
    assert page._appearance_preview.has_image is True
    assert page._appearance_preview.source_path == str(preview.resolve())
    assert page._viewport_state.text() == "NeRF preview • appearance only"
    assert page._nerf_authority_value.text() == "Appearance only • canonical geometry remains authoritative"

    page._nerf_hide_button.click()
    qtbot.wait(10)
    assert page.nerf_preview_visible is False
    assert page._nerf_state_value.text() == presentation.state_text
    assert page._nerf_show_button.isEnabled() is True
    assert page._nerf_hide_button.isEnabled() is False


def test_f3_selection_change_hides_stale_nerf_preview(window, tmp_path: Path, qtbot) -> None:
    gaussian_root = (tmp_path / "gaussian").resolve()
    nerf_root = (tmp_path / "nerf").resolve()
    _preview(gaussian_root, label="Gaussian")
    nerf_preview = _preview(nerf_root, label="NeRF")
    (nerf_root / "e2_runtime_evidence.json").write_text(
        json.dumps({"preview_path": str(nerf_preview.resolve())}), encoding="utf-8"
    )
    gaussian = _GaussianArtifact("f3-ui-gaussian", str(gaussian_root))
    nerf = _NerfArtifact("f3-ui-nerf", str(nerf_root))
    scene = SimpleNamespace(
        scene_id="prs-f3-ui-switch",
        session_id=SESSION,
        state=SimpleNamespace(value="ready"),
        artifacts=(gaussian, nerf),
        gaussian_artifact=gaussian,
        nerf_artifact=nerf,
        canonical_geometry_authoritative=True,
        certified_metrology=False,
    )
    page = window._reconstruction_page
    page.set_persistent_scene(scene, "persistent_reconstruction_scene.json")
    nerf_index = page._appearance_representation_combo.findText("NeRF")
    gaussian_index = page._appearance_representation_combo.findText("Gaussian")
    assert nerf_index >= 0 and gaussian_index >= 0
    page._appearance_representation_combo.setCurrentIndex(nerf_index)
    qtbot.wait(10)
    page._nerf_show_button.click()
    qtbot.wait(10)
    assert page.nerf_preview_visible is True

    page._appearance_representation_combo.setCurrentIndex(gaussian_index)
    qtbot.wait(10)
    assert page.selected_appearance_representation is not None
    assert page.selected_appearance_representation.label == "Gaussian"
    assert page.nerf_preview_visible is False
    assert page._nerf_show_button.isEnabled() is False
    assert page._gaussian_show_button.isEnabled() is True


def test_f3_shared_overlay_refresh_preserves_active_representation(window, tmp_path: Path, qtbot) -> None:
    gaussian_root = (tmp_path / "gaussian").resolve()
    nerf_root = (tmp_path / "nerf").resolve()
    _preview(gaussian_root, label="Gaussian")
    nerf_preview = _preview(nerf_root, label="NeRF")
    (nerf_root / "e2_runtime_evidence.json").write_text(
        json.dumps({"preview_path": str(nerf_preview.resolve())}), encoding="utf-8"
    )
    gaussian = _GaussianArtifact("f3-overlay-gaussian", str(gaussian_root))
    nerf = _NerfArtifact("f3-overlay-nerf", str(nerf_root))
    scene = SimpleNamespace(
        scene_id="prs-f3-overlay",
        session_id=SESSION,
        state=SimpleNamespace(value="ready"),
        artifacts=(gaussian, nerf),
        gaussian_artifact=gaussian,
        nerf_artifact=nerf,
        canonical_geometry_authoritative=True,
        certified_metrology=False,
    )
    page = window._reconstruction_page
    page.set_persistent_scene(scene, "persistent_reconstruction_scene.json")

    nerf_index = page._appearance_representation_combo.findText("NeRF")
    gaussian_index = page._appearance_representation_combo.findText("Gaussian")
    page._appearance_representation_combo.setCurrentIndex(nerf_index)
    page._nerf_show_button.click()
    qtbot.wait(10)
    assert page.nerf_preview_visible is True
    page._refresh_gaussian_visualization_controls()
    qtbot.wait(10)
    assert page.nerf_preview_visible is True
    assert page._viewport_state.text() == "NeRF preview • appearance only"

    page._appearance_representation_combo.setCurrentIndex(gaussian_index)
    page._gaussian_show_button.click()
    qtbot.wait(10)
    assert page.gaussian_preview_visible is True
    page._refresh_nerf_visualization_controls()
    qtbot.wait(10)
    assert page.gaussian_preview_visible is True
    assert page._viewport_state.text() == "Gaussian Splatting preview • appearance only"
