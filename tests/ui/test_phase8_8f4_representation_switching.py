from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from types import SimpleNamespace

from PIL import Image, ImageDraw
import pytest

from app.main_window import MainWindow

SESSION = "89898989-8989-4989-8989-898989898989"


@dataclass
class _Artifact:
    artifact_id: str
    artifact_dir: str
    representation_kind: str

    @property
    def kind(self):
        return SimpleNamespace(value=self.representation_kind)

    @property
    def authority(self):
        return SimpleNamespace(value="appearance_only")

    @property
    def canonical_geometry_authoritative(self):
        return False

    def as_dict(self):
        gaussian = self.representation_kind == "gaussian_scene"
        return {
            "artifact_id": self.artifact_id,
            "kind": self.representation_kind,
            "authority": "appearance_only",
            "provider_id": "appearance-provider",
            "canonical_geometry_authoritative": False,
            "certified_metrology": False,
            "metadata": {
                "backend_id": "external-appearance-provider",
                "backend_version": "1.0",
                "artifact_dir": self.artifact_dir,
                "gaussian_count": 1200 if gaussian else 0,
                "render_capable": True,
                "metric_geometry_status": "arbitrary",
            },
            "payload_snapshot": {
                "backend_id": "external-appearance-provider",
                "backend_version": "1.0",
                "artifact_dir": self.artifact_dir,
                "gaussian_count": 1200 if gaussian else 0,
                "optimization_state": "optimized" if gaussian else "",
                "training_state": "" if gaussian else "trained",
                "input_frame_count": 0 if gaussian else 24,
                "training_steps": 0 if gaussian else 1000,
                "render_capable": True,
                "metric_geometry_status": "arbitrary",
            },
        }


def _preview(root: Path, *, label: str) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", (640, 360), (8, 12, 18))
    draw = ImageDraw.Draw(image)
    draw.rectangle((130, 75, 510, 295), outline=(90, 155, 220), width=5)
    draw.ellipse((245, 105, 395, 255), fill=(65, 120, 185))
    draw.text((18, 18), label, fill=(205, 215, 225))
    path = root / "preview.png"
    image.save(path)
    return path.resolve()


def _scene(tmp_path: Path, *, nerf_preview: bool = True):
    gaussian_root = (tmp_path / "gaussian").resolve()
    nerf_root = (tmp_path / "nerf").resolve()
    gaussian_preview = _preview(gaussian_root, label="F4 Gaussian")
    (gaussian_root / "qualification_metrics.json").write_text(
        json.dumps({"render_written": True, "preview_path": str(gaussian_preview)}),
        encoding="utf-8",
    )
    nerf_path = None
    if nerf_preview:
        nerf_path = _preview(nerf_root, label="F4 NeRF")
        (nerf_root / "e2_runtime_evidence.json").write_text(
            json.dumps({"preview_path": str(nerf_path)}), encoding="utf-8"
        )
    else:
        nerf_root.mkdir(parents=True, exist_ok=True)

    gaussian = _Artifact("f4-gaussian", str(gaussian_root), "gaussian_scene")
    nerf = _Artifact("f4-nerf", str(nerf_root), "nerf_scene")
    scene = SimpleNamespace(
        scene_id="prs-f4-ui",
        session_id=SESSION,
        state=SimpleNamespace(value="ready"),
        artifacts=(gaussian, nerf),
        gaussian_artifact=gaussian,
        nerf_artifact=nerf,
        canonical_geometry_authoritative=True,
        certified_metrology=False,
    )
    return scene, gaussian_preview, nerf_path


@pytest.fixture
def window(qtbot, monkeypatch):
    monkeypatch.setattr("app.application_controller.CameraEnumerator.start", lambda self: None)
    win = MainWindow()
    qtbot.addWidget(win)
    yield win
    win.close()


def _activate(page, index: int, qtbot) -> None:
    page._appearance_representation_combo.setCurrentIndex(index)
    page._appearance_representation_combo.activated.emit(index)
    qtbot.wait(10)


def test_f4_user_activation_switches_gaussian_nerf_gaussian_on_shared_viewport(
    window, tmp_path: Path, qtbot
) -> None:
    scene, gaussian_preview, nerf_preview = _scene(tmp_path)
    assert nerf_preview is not None
    page = window._reconstruction_page
    page.set_persistent_scene(scene, "persistent_reconstruction_scene.json")
    gaussian_index = page._appearance_representation_combo.findText("Gaussian")
    nerf_index = page._appearance_representation_combo.findText("NeRF")
    assert gaussian_index >= 0 and nerf_index >= 0
    assert page.active_view_representation == "canonical_geometry"

    _activate(page, gaussian_index, qtbot)
    assert page.gaussian_preview_visible is True
    assert page.nerf_preview_visible is False
    assert page._appearance_preview.source_path == str(gaussian_preview)
    assert page.active_view_representation == "gaussian_scene"
    assert page._appearance_viewport_value.text() == "Gaussian appearance • appearance only"

    _activate(page, nerf_index, qtbot)
    assert page.gaussian_preview_visible is False
    assert page.nerf_preview_visible is True
    assert page._appearance_preview.source_path == str(nerf_preview)
    assert page.active_view_representation == "nerf_scene"
    assert page._appearance_viewport_value.text() == "NeRF appearance • appearance only"

    _activate(page, gaussian_index, qtbot)
    assert page.gaussian_preview_visible is True
    assert page.nerf_preview_visible is False
    assert page._appearance_preview.source_path == str(gaussian_preview)


def test_f4_reactivating_current_selection_enters_appearance_view(window, tmp_path: Path, qtbot) -> None:
    scene, gaussian_preview, _ = _scene(tmp_path)
    page = window._reconstruction_page
    page.set_persistent_scene(scene, "persistent_reconstruction_scene.json")
    index = page._appearance_representation_combo.currentIndex()
    assert page._appearance_representation_combo.currentText() == "Gaussian"
    assert page.active_view_representation == "canonical_geometry"

    page._appearance_representation_combo.activated.emit(index)
    qtbot.wait(10)
    assert page.gaussian_preview_visible is True
    assert page._appearance_preview.source_path == str(gaussian_preview)


def test_f4_programmatic_metadata_selection_still_does_not_implicitly_switch(window, tmp_path: Path, qtbot) -> None:
    scene, _, _ = _scene(tmp_path)
    page = window._reconstruction_page
    page.set_persistent_scene(scene, "persistent_reconstruction_scene.json")
    nerf_index = page._appearance_representation_combo.findText("NeRF")
    page._appearance_representation_combo.setCurrentIndex(nerf_index)
    qtbot.wait(10)
    assert page.selected_appearance_representation is not None
    assert page.selected_appearance_representation.label == "NeRF"
    assert page.active_view_representation == "canonical_geometry"
    assert page.gaussian_preview_visible is False
    assert page.nerf_preview_visible is False


def test_f4_unavailable_selected_target_retains_canonical_without_cross_representation_fallback(
    window, tmp_path: Path, qtbot
) -> None:
    scene, _, _ = _scene(tmp_path, nerf_preview=False)
    page = window._reconstruction_page
    page.set_persistent_scene(scene, "persistent_reconstruction_scene.json")
    gaussian_index = page._appearance_representation_combo.findText("Gaussian")
    nerf_index = page._appearance_representation_combo.findText("NeRF")

    _activate(page, gaussian_index, qtbot)
    assert page.gaussian_preview_visible is True
    _activate(page, nerf_index, qtbot)
    assert page.gaussian_preview_visible is False
    assert page.nerf_preview_visible is False
    assert page.active_view_representation == "canonical_geometry"
    assert page.selected_appearance_representation is not None
    assert page.selected_appearance_representation.label == "NeRF"
    assert page._appearance_viewport_value.text() == "NeRF unavailable • canonical geometry retained"


def test_f4_hide_returns_to_canonical_without_changing_selected_metadata(window, tmp_path: Path, qtbot) -> None:
    scene, _, _ = _scene(tmp_path)
    page = window._reconstruction_page
    page.set_persistent_scene(scene, "persistent_reconstruction_scene.json")
    nerf_index = page._appearance_representation_combo.findText("NeRF")
    _activate(page, nerf_index, qtbot)
    assert page.nerf_preview_visible is True

    page._nerf_hide_button.click()
    qtbot.wait(10)
    assert page.nerf_preview_visible is False
    assert page.active_view_representation == "canonical_geometry"
    assert page.selected_appearance_representation is not None
    assert page.selected_appearance_representation.label == "NeRF"
    assert page._appearance_viewport_value.text() == "Canonical geometry"
    assert page._appearance_authority_value.text() == "Appearance only • canonical geometry remains authoritative"
