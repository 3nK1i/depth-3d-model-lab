from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from types import SimpleNamespace

from PIL import Image, ImageDraw
import pytest

from app.main_window import MainWindow
from tests.reconstruction_helpers import make_hardware

SESSION = "95959595-9595-4959-9595-959595959595"


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


def _preview(root: Path, *, label: str, valid: bool = True) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    path = root / "preview.png"
    if not valid:
        path.write_bytes(b"not-a-decodable-image")
        return path.resolve()
    image = Image.new("RGB", (640, 360), (8, 12, 18))
    draw = ImageDraw.Draw(image)
    draw.rectangle((130, 75, 510, 295), outline=(90, 155, 220), width=5)
    draw.ellipse((245, 105, 395, 255), fill=(65, 120, 185))
    draw.text((18, 18), label, fill=(205, 215, 225))
    image.save(path)
    return path.resolve()


def _scene(
    tmp_path: Path,
    *,
    gaussian_preview: bool = True,
    gaussian_valid: bool = True,
    nerf_preview: bool = True,
    fallback_evidence=(),
):
    gaussian_root = (tmp_path / "gaussian").resolve()
    nerf_root = (tmp_path / "nerf").resolve()
    gaussian_path = None
    if gaussian_preview:
        gaussian_path = _preview(gaussian_root, label="F5 Gaussian", valid=gaussian_valid)
        (gaussian_root / "qualification_metrics.json").write_text(
            json.dumps({"render_written": True, "preview_path": str(gaussian_path)}),
            encoding="utf-8",
        )
    else:
        gaussian_root.mkdir(parents=True, exist_ok=True)
    nerf_path = None
    if nerf_preview:
        nerf_path = _preview(nerf_root, label="F5 NeRF")
        (nerf_root / "e2_runtime_evidence.json").write_text(
            json.dumps({"preview_path": str(nerf_path)}), encoding="utf-8"
        )
    else:
        nerf_root.mkdir(parents=True, exist_ok=True)

    gaussian = _Artifact("f5-gaussian", str(gaussian_root), "gaussian_scene")
    nerf = _Artifact("f5-nerf", str(nerf_root), "nerf_scene")
    return SimpleNamespace(
        scene_id="prs-f5-ui",
        session_id=SESSION,
        state=SimpleNamespace(value="ready"),
        artifacts=(gaussian, nerf),
        gaussian_artifact=gaussian,
        nerf_artifact=nerf,
        fallback_evidence=tuple(fallback_evidence),
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


def _activate(page, label: str, qtbot) -> None:
    index = page._appearance_representation_combo.findText(label)
    assert index >= 0
    page._appearance_representation_combo.setCurrentIndex(index)
    page._appearance_representation_combo.activated.emit(index)
    qtbot.wait(10)


def test_f5_resource_group_is_compact_and_cpu_does_not_disable_static_preview(
    window, tmp_path: Path, qtbot, monkeypatch
) -> None:
    monkeypatch.setattr(
        "app.reconstruction_appearance_resources.detect_hardware",
        lambda: make_hardware(total_gib=None),
    )
    page = window._reconstruction_page
    page.set_persistent_scene(_scene(tmp_path), "persistent_reconstruction_scene.json")
    qtbot.wait(10)

    assert page._appearance_resource_group is not None
    assert page._appearance_resource_device_value.text() == "CPU"
    assert "static previews remain supported" in page._appearance_resource_resources_value.text()
    assert "Gaussian static preview ready" in page._appearance_resource_presentation_value.text()
    assert page._gaussian_show_button.isEnabled() is True
    assert page._appearance_resource_fallback_value.text() == "Canonical geometry • no neural cross-fallback"


def test_f5_active_preview_reports_backend_free_presentation_and_authority(
    window, tmp_path: Path, qtbot, monkeypatch
) -> None:
    monkeypatch.setattr(
        "app.reconstruction_appearance_resources.detect_hardware",
        lambda: make_hardware(total_gib=6.0, free_gib=4.0, name="RTX 2060"),
    )
    page = window._reconstruction_page
    page.set_persistent_scene(_scene(tmp_path), "persistent_reconstruction_scene.json")
    _activate(page, "Gaussian", qtbot)

    assert page.gaussian_preview_visible is True
    presentation = page.appearance_resource_fallback_presentation
    assert presentation is not None and presentation.fallback_active is False
    assert "no neural runtime required" in page._appearance_resource_presentation_value.text()
    assert page._appearance_resource_state_value.text() == "Appearance view active • canonical authority retained"
    assert page._appearance_authority_value.text() == "Appearance only • canonical geometry remains authoritative"


def test_f5_unavailable_selected_preview_falls_back_to_canonical_without_cross_fallback(
    window, tmp_path: Path, qtbot
) -> None:
    page = window._reconstruction_page
    page.set_persistent_scene(
        _scene(tmp_path, gaussian_preview=True, nerf_preview=False),
        "persistent_reconstruction_scene.json",
    )
    _activate(page, "Gaussian", qtbot)
    assert page.gaussian_preview_visible is True

    _activate(page, "NeRF", qtbot)
    assert page.gaussian_preview_visible is False
    assert page.nerf_preview_visible is False
    assert page.active_view_representation == "canonical_geometry"
    presentation = page.appearance_resource_fallback_presentation
    assert presentation is not None and presentation.fallback_active is True
    assert presentation.cross_representation_fallback is False
    assert page._appearance_resource_state_value.text().startswith("Fallback active •")
    assert page._appearance_resource_fallback_value.text() == "Canonical geometry • no neural cross-fallback"


def test_f5_decode_failure_is_explicit_and_retains_canonical(window, tmp_path: Path, qtbot) -> None:
    page = window._reconstruction_page
    page.set_persistent_scene(
        _scene(tmp_path, gaussian_preview=True, gaussian_valid=False),
        "persistent_reconstruction_scene.json",
    )
    _activate(page, "Gaussian", qtbot)

    assert page.gaussian_preview_visible is False
    assert page.active_view_representation == "canonical_geometry"
    assert page._appearance_resource_state_value.text() == (
        "Fallback active • Gaussian preview image could not be displayed"
    )
    assert page._appearance_resource_presentation_value.text() == (
        "Canonical viewport • appearance request failed safely"
    )


def test_f5_new_selection_clears_stale_failure_state(window, tmp_path: Path, qtbot) -> None:
    page = window._reconstruction_page
    scene = _scene(tmp_path, gaussian_preview=True, gaussian_valid=False, nerf_preview=True)
    page.set_persistent_scene(scene, "persistent_reconstruction_scene.json")
    _activate(page, "Gaussian", qtbot)
    assert page.appearance_resource_fallback_presentation is not None
    assert page.appearance_resource_fallback_presentation.fallback_active is True

    _activate(page, "NeRF", qtbot)
    assert page.nerf_preview_visible is True
    assert page.appearance_resource_fallback_presentation is not None
    assert page.appearance_resource_fallback_presentation.fallback_active is False
    assert page._appearance_resource_state_value.text() == "Appearance view active • canonical authority retained"


def test_f5_surfaces_persisted_appearance_pipeline_fallback_evidence(window, tmp_path: Path, qtbot) -> None:
    evidence = SimpleNamespace(
        stage=SimpleNamespace(value="gaussian_appearance"),
        state=SimpleNamespace(value="skipped"),
        provider_attempts=("external-provider",),
        fallback_used=False,
        reason="resource_profile_unavailable",
    )
    page = window._reconstruction_page
    page.set_persistent_scene(
        _scene(tmp_path, fallback_evidence=(evidence,)),
        "persistent_reconstruction_scene.json",
    )
    qtbot.wait(10)
    assert "Gaussian appearance" in page._appearance_resource_pipeline_value.text()
    pipeline_detail = (
        page._appearance_resource_pipeline_value.text()
        + page._appearance_resource_pipeline_value.toolTip()
    )
    assert "resource_profile_unavailable" in pipeline_detail
