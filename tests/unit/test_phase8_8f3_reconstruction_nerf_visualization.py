from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from app.reconstruction_nerf_visualization import (
    build_nerf_visualization_presentation,
    resolve_nerf_preview_path,
)

SESSION = "67676767-6767-4676-8676-676767676767"


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
        return SimpleNamespace(value="nerf_scene")

    @property
    def authority(self):
        return SimpleNamespace(value=self.authority_name)

    @property
    def canonical_geometry_authoritative(self):
        return self.canonical

    def as_dict(self):
        return {
            "artifact_id": self.artifact_id,
            "kind": "nerf_scene",
            "authority": self.authority_name,
            "provider_id": "nerf-provider",
            "canonical_geometry_authoritative": self.canonical,
            "certified_metrology": self.certified,
            "metadata": {
                "backend_id": "external-nerf-provider",
                "backend_version": "1.1.5",
                "render_capable": self.render_capable,
                "metric_geometry_status": "arbitrary",
                "artifact_dir": self.artifact_dir,
            },
            "payload_snapshot": {
                "backend_id": "external-nerf-provider",
                "backend_version": "1.1.5",
                "training_state": "trained",
                "render_capable": self.render_capable,
                "metric_geometry_status": "arbitrary",
                "artifact_dir": self.artifact_dir,
                "config_path": str(Path(self.artifact_dir) / "config.yml"),
                "checkpoint_path": str(Path(self.artifact_dir) / "model.ckpt"),
                "input_frame_count": 18,
                "training_steps": 1000,
            },
        }


def _scene(artifact: _Artifact | None):
    artifacts = () if artifact is None else (artifact,)
    return SimpleNamespace(
        scene_id="prs-f3-unit",
        session_id=SESSION,
        artifacts=artifacts,
        nerf_artifact=artifact,
        canonical_geometry_authoritative=True,
        certified_metrology=False,
    )


def _preview(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    path = root / "preview.png"
    path.write_bytes(b"not-decoded-by-pure-contract")
    return path


def test_f3_resolves_runtime_evidence_preview(tmp_path: Path) -> None:
    root = (tmp_path / "nerf").resolve()
    preview = _preview(root)
    (root / "e2_runtime_evidence.json").write_text(
        json.dumps({"preview_path": str(preview.resolve()), "training_completed_without_oom": True}),
        encoding="utf-8",
    )
    path, source, error = resolve_nerf_preview_path(root)
    assert path == preview.resolve()
    assert source == "runtime evidence"
    assert error == ""


def test_f3_resolves_heldout_manifest_render_when_preview_is_absent(tmp_path: Path) -> None:
    root = (tmp_path / "nerf").resolve()
    root.mkdir()
    rendered = root / "rendered" / "test" / "frame-000.png"
    rendered.parent.mkdir(parents=True)
    rendered.write_bytes(b"render")
    (root / "heldout_render_manifest.json").write_text(
        json.dumps({"renders": [{"rendered_image_path": str(rendered.resolve())}]}),
        encoding="utf-8",
    )
    path, source, error = resolve_nerf_preview_path(root)
    assert path == rendered.resolve()
    assert source == "held-out render manifest"
    assert error == ""


def test_f3_uses_artifact_local_preview_as_compatibility_fallback(tmp_path: Path) -> None:
    root = (tmp_path / "nerf").resolve()
    preview = _preview(root)
    path, source, error = resolve_nerf_preview_path(root)
    assert path == preview.resolve()
    assert source == "artifact-local preview"
    assert error == ""


def test_f3_rejects_runtime_evidence_preview_escape(tmp_path: Path) -> None:
    root = (tmp_path / "nerf").resolve()
    root.mkdir()
    outside = (tmp_path / "outside.png").resolve()
    outside.write_bytes(b"outside")
    (root / "e2_runtime_evidence.json").write_text(
        json.dumps({"preview_path": str(outside)}), encoding="utf-8"
    )
    path, source, error = resolve_nerf_preview_path(root)
    assert path is None
    assert source == ""
    assert "escapes" in error


def test_f3_builds_ready_presentation_without_parsing_nerf_checkpoint(tmp_path: Path) -> None:
    root = (tmp_path / "nerf").resolve()
    preview = _preview(root)
    artifact = _Artifact("f3-nerf", str(root))
    presentation = build_nerf_visualization_presentation(_scene(artifact))
    assert presentation is not None
    assert presentation.ready is True
    assert presentation.preview_path == str(preview.resolve())
    assert presentation.provider_text == "external-nerf-provider • v1.1.5"
    assert presentation.scene_text == "18 frames • 1,000 steps • appearance only"
    assert presentation.scale_text == "Arbitrary scene gauge • appearance only"
    assert presentation.canonical_geometry_authoritative is False
    assert presentation.certified_metrology is False
    assert presentation.authority_text == "Appearance only • canonical geometry remains authoritative"


def test_f3_missing_preview_is_explicit_and_not_promoted(tmp_path: Path) -> None:
    root = (tmp_path / "nerf").resolve()
    root.mkdir()
    presentation = build_nerf_visualization_presentation(_scene(_Artifact("f3-missing", str(root))))
    assert presentation is not None
    assert presentation.ready is False
    assert presentation.preview_path == ""
    assert "unavailable" in presentation.state_text.lower()
    assert presentation.canonical_geometry_authoritative is False


def test_f3_render_capability_false_never_enables_preview(tmp_path: Path) -> None:
    root = (tmp_path / "nerf").resolve()
    _preview(root)
    presentation = build_nerf_visualization_presentation(
        _scene(_Artifact("f3-seed", str(root), render_capable=False))
    )
    assert presentation is not None
    assert presentation.ready is False
    assert "not declared render-capable" in presentation.state_text


def test_f3_no_nerf_artifact_returns_none() -> None:
    assert build_nerf_visualization_presentation(_scene(None)) is None


@pytest.mark.parametrize(
    ("authority", "canonical", "certified"),
    [
        ("canonical_geometry", False, False),
        ("appearance_only", True, False),
        ("appearance_only", False, True),
    ],
)
def test_f3_rejects_forbidden_authority_mutations(
    tmp_path: Path, authority: str, canonical: bool, certified: bool
) -> None:
    root = (tmp_path / "nerf").resolve()
    _preview(root)
    artifact = _Artifact(
        "f3-bad-authority",
        str(root),
        authority_name=authority,
        canonical=canonical,
        certified=certified,
    )
    with pytest.raises(ValueError):
        build_nerf_visualization_presentation(_scene(artifact))
