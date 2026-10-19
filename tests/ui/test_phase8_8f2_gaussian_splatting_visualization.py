from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from types import SimpleNamespace

from PIL import Image, ImageDraw
import pytest

from app.main_window import MainWindow

SESSION = "56565656-5656-4565-8565-565656565656"


@dataclass
class _Artifact:
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
                "backend_id": "depth3d-gsplat-external",
                "backend_version": "1.5.3",
                "artifact_dir": self.artifact_dir,
                "gaussian_count": 1024,
                "render_capable": True,
                "metric_geometry_status": "arbitrary",
            },
            "payload_snapshot": {
                "backend_id": "depth3d-gsplat-external",
                "backend_version": "1.5.3",
                "artifact_dir": self.artifact_dir,
                "scene_path": str(Path(self.artifact_dir) / "scene.pt"),
                "gaussian_count": 1024,
                "optimization_state": "optimized",
                "render_capable": True,
                "metric_geometry_status": "arbitrary",
            },
        }


def _preview(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", (640, 360), (8, 12, 18))
    draw = ImageDraw.Draw(image)
    for x, y, r in ((210, 180, 58), (320, 145, 46), (405, 210, 68), (290, 240, 34)):
        draw.ellipse((x-r, y-r, x+r, y+r), fill=(90 + r, 90, 150 + r // 2))
    image.save(root / "preview.png")
    (root / "qualification_metrics.json").write_text(
        json.dumps({"render_written": True, "preview_path": str((root / "preview.png").resolve())}),
        encoding="utf-8",
    )


def _scene(root: Path):
    artifact = _Artifact("f2-ui-gaussian", str(root.resolve()))
    return SimpleNamespace(
        scene_id="prs-f2-ui",
        session_id=SESSION,
        state=SimpleNamespace(value="ready"),
        artifacts=(artifact,),
        gaussian_artifact=artifact,
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


def test_f2_group_is_safe_without_gaussian_scene(window) -> None:
    page = window._reconstruction_page
    assert page._gaussian_visualization_group.isHidden() is False
    assert page.gaussian_visualization_presentation is None
    assert page._gaussian_show_button.isEnabled() is False
    assert page.gaussian_preview_visible is False


def test_f2_shows_and_hides_backend_authored_gaussian_preview(window, tmp_path: Path, qtbot) -> None:
    root = (tmp_path / "gaussian").resolve()
    _preview(root)
    page = window._reconstruction_page
    page.set_persistent_scene(_scene(root), "persistent_reconstruction_scene.json")
    presentation = page.gaussian_visualization_presentation
    assert presentation is not None and presentation.ready
    assert page.selected_appearance_representation is not None
    assert page.selected_appearance_representation.label == "Gaussian"
    assert page._gaussian_show_button.isEnabled() is True

    page._gaussian_show_button.click()
    qtbot.wait(10)
    assert page.gaussian_preview_visible is True
    assert page._appearance_preview.has_image is True
    assert page._appearance_preview.source_path.endswith("preview.png")
    assert page._viewport_state.text() == "Gaussian Splatting preview • appearance only"
    assert page._gaussian_authority_value.text() == "Appearance only • canonical geometry remains authoritative"

    page._gaussian_hide_button.click()
    qtbot.wait(10)
    assert page.gaussian_preview_visible is False
    assert page._gaussian_state_value.text() == presentation.state_text
    assert page._gaussian_show_button.isEnabled() is True
    assert page._gaussian_hide_button.isEnabled() is False


def test_f2_selection_change_hides_stale_gaussian_preview(window, tmp_path: Path, qtbot) -> None:
    root = (tmp_path / "gaussian").resolve()
    _preview(root)
    gaussian = _Artifact("f2-ui-gaussian", str(root))

    @dataclass
    class _Nerf:
        artifact_id: str = "f2-ui-nerf"
        @property
        def kind(self): return SimpleNamespace(value="nerf_scene")
        @property
        def authority(self): return SimpleNamespace(value="appearance_only")
        @property
        def canonical_geometry_authoritative(self): return False
        def as_dict(self):
            return {
                "artifact_id": self.artifact_id, "kind": "nerf_scene", "authority": "appearance_only",
                "provider_id": "nerf-provider", "canonical_geometry_authoritative": False,
                "certified_metrology": False,
                "metadata": {"backend_id": "nerfstudio-nerfacto", "backend_version": "1.1.5", "render_capable": True, "metric_geometry_status": "arbitrary"},
                "payload_snapshot": {"backend_id": "nerfstudio-nerfacto", "backend_version": "1.1.5", "training_state": "trained", "render_capable": True, "metric_geometry_status": "arbitrary"},
            }

    scene = SimpleNamespace(
        scene_id="prs-f2-ui-switch", session_id=SESSION, state=SimpleNamespace(value="ready"),
        artifacts=(gaussian, _Nerf()), gaussian_artifact=gaussian,
        canonical_geometry_authoritative=True, certified_metrology=False,
    )
    page = window._reconstruction_page
    page.set_persistent_scene(scene, "persistent_reconstruction_scene.json")
    page._gaussian_show_button.click(); qtbot.wait(10)
    assert page.gaussian_preview_visible
    page._appearance_representation_combo.setCurrentIndex(1); qtbot.wait(10)
    assert page.selected_appearance_representation is not None
    assert page.selected_appearance_representation.label == "NeRF"
    assert page.gaussian_preview_visible is False
    assert page._gaussian_show_button.isEnabled() is False
