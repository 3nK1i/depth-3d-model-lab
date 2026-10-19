from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from app.reconstruction_gaussian_visualization import (
    build_gaussian_visualization_presentation,
    resolve_gaussian_preview_path,
)

SESSION = "34343434-3434-4343-8343-343434343434"


@dataclass
class _Artifact:
    artifact_id: str
    artifact_dir: str
    authority_name: str = "appearance_only"
    canonical: bool = False
    certified: bool = False
    render_capable: bool = True

    @property
    def kind(self):
        return SimpleNamespace(value="gaussian_scene")

    @property
    def authority(self):
        return SimpleNamespace(value=self.authority_name)

    @property
    def canonical_geometry_authoritative(self):
        return self.canonical

    def as_dict(self):
        return {
            "artifact_id": self.artifact_id,
            "kind": "gaussian_scene",
            "authority": self.authority_name,
            "provider_id": "gaussian-provider",
            "canonical_geometry_authoritative": self.canonical,
            "certified_metrology": self.certified,
            "metadata": {
                "backend_id": "depth3d-gsplat-external",
                "backend_version": "1.5.3",
                "render_capable": self.render_capable,
                "metric_geometry_status": "arbitrary",
                "gaussian_count": 2048,
                "artifact_dir": self.artifact_dir,
            },
            "payload_snapshot": {
                "backend_id": "depth3d-gsplat-external",
                "backend_version": "1.5.3",
                "render_capable": self.render_capable,
                "metric_geometry_status": "arbitrary",
                "gaussian_count": 2048,
                "artifact_dir": self.artifact_dir,
                "scene_path": str(Path(self.artifact_dir) / "scene.pt"),
            },
        }


def _scene(artifact: _Artifact | None):
    artifacts = () if artifact is None else (artifact,)
    return SimpleNamespace(
        scene_id="prs-f2-unit",
        session_id=SESSION,
        artifacts=artifacts,
        gaussian_artifact=artifact,
        canonical_geometry_authoritative=True,
        certified_metrology=False,
    )


def _write_preview(root: Path, *, metrics: bool = True) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    preview = root / "preview.png"
    preview.write_bytes(b"not-decoded-by-pure-contract")
    if metrics:
        (root / "qualification_metrics.json").write_text(
            json.dumps(
                {
                    "render_written": True,
                    "preview_path": str(preview.resolve()),
                    "backend_kind": "gsplat_external",
                }
            ),
            encoding="utf-8",
        )
    return preview


def test_f2_resolves_qualified_artifact_local_preview(tmp_path: Path) -> None:
    root = (tmp_path / "gaussian").resolve()
    preview = _write_preview(root)
    path, source, error = resolve_gaussian_preview_path(root)
    assert path == preview.resolve()
    assert source == "qualification metrics"
    assert error == ""


def test_f2_uses_worker_convention_when_metrics_are_absent(tmp_path: Path) -> None:
    root = (tmp_path / "gaussian").resolve()
    preview = _write_preview(root, metrics=False)
    path, source, error = resolve_gaussian_preview_path(root)
    assert path == preview.resolve()
    assert source == "artifact-local preview"
    assert error == ""


def test_f2_rejects_qualification_preview_escape(tmp_path: Path) -> None:
    root = (tmp_path / "gaussian").resolve()
    root.mkdir()
    outside = (tmp_path / "outside.png").resolve()
    outside.write_bytes(b"outside")
    (root / "qualification_metrics.json").write_text(
        json.dumps({"render_written": True, "preview_path": str(outside)}),
        encoding="utf-8",
    )
    path, source, error = resolve_gaussian_preview_path(root)
    assert path is None
    assert source == ""
    assert "escapes" in error


def test_f2_builds_ready_presentation_without_parsing_gaussian_scene(tmp_path: Path) -> None:
    root = (tmp_path / "gaussian").resolve()
    preview = _write_preview(root)
    artifact = _Artifact("f2-gaussian", str(root))
    presentation = build_gaussian_visualization_presentation(_scene(artifact))
    assert presentation is not None
    assert presentation.ready is True
    assert presentation.preview_path == str(preview.resolve())
    assert presentation.provider_text == "depth3d-gsplat-external • v1.5.3"
    assert presentation.scene_text == "2,048 Gaussians • appearance only"
    assert presentation.scale_text == "Arbitrary scene gauge • appearance only"
    assert presentation.canonical_geometry_authoritative is False
    assert presentation.certified_metrology is False
    assert presentation.authority_text == "Appearance only • canonical geometry remains authoritative"


def test_f2_missing_preview_is_explicit_and_not_promoted(tmp_path: Path) -> None:
    root = (tmp_path / "gaussian").resolve()
    root.mkdir()
    presentation = build_gaussian_visualization_presentation(_scene(_Artifact("f2-missing", str(root))))
    assert presentation is not None
    assert presentation.ready is False
    assert presentation.preview_path == ""
    assert "unavailable" in presentation.state_text.lower()
    assert presentation.canonical_geometry_authoritative is False


def test_f2_render_capability_false_never_enables_preview(tmp_path: Path) -> None:
    root = (tmp_path / "gaussian").resolve()
    _write_preview(root)
    presentation = build_gaussian_visualization_presentation(
        _scene(_Artifact("f2-seed", str(root), render_capable=False))
    )
    assert presentation is not None
    assert presentation.ready is False
    assert "not declared render-capable" in presentation.state_text


def test_f2_no_gaussian_artifact_returns_none() -> None:
    assert build_gaussian_visualization_presentation(_scene(None)) is None


@pytest.mark.parametrize(
    ("authority", "canonical", "certified"),
    [
        ("canonical_geometry", False, False),
        ("appearance_only", True, False),
        ("appearance_only", False, True),
    ],
)
def test_f2_rejects_forbidden_authority_mutations(
    tmp_path: Path, authority: str, canonical: bool, certified: bool
) -> None:
    root = (tmp_path / "gaussian").resolve()
    _write_preview(root)
    artifact = _Artifact(
        "f2-bad-authority",
        str(root),
        authority_name=authority,
        canonical=canonical,
        certified=certified,
    )
    with pytest.raises(ValueError):
        build_gaussian_visualization_presentation(_scene(artifact))
