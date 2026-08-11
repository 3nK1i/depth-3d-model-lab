from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from app.reconstruction_appearance_presentation import (
    AppearanceRepresentationEntry,
    build_reconstruction_appearance_presentation,
    empty_appearance_representation_text,
)


@dataclass
class _Artifact:
    artifact_id: str
    kind_name: str
    provider_id: str
    metadata: dict[str, object]
    payload_snapshot: dict[str, object]
    authority_name: str = "appearance_only"
    canonical_geometry_authoritative: bool = False
    certified_metrology: bool = False

    @property
    def kind(self):
        return SimpleNamespace(value=self.kind_name)

    @property
    def authority(self):
        return SimpleNamespace(value=self.authority_name)

    def as_dict(self) -> dict[str, object]:
        return {
            "artifact_id": self.artifact_id,
            "kind": self.kind_name,
            "authority": self.authority_name,
            "provider_id": self.provider_id,
            "metadata": self.metadata,
            "payload_snapshot": self.payload_snapshot,
            "canonical_geometry_authoritative": self.canonical_geometry_authoritative,
            "certified_metrology": self.certified_metrology,
        }


def _scene(*artifacts: _Artifact):
    return SimpleNamespace(
        scene_id="prs-f1-test",
        session_id="11111111-1111-4111-8111-111111111111",
        state=SimpleNamespace(value="ready"),
        artifacts=tuple(artifacts),
        fallback_evidence=(),
        canonical_geometry_authoritative=True,
        certified_metrology=False,
    )


def _gaussian() -> _Artifact:
    return _Artifact(
        artifact_id="scene:gaussian",
        kind_name="gaussian_scene",
        provider_id="qualified-gaussian-appearance",
        metadata={"backend_id": "eia-gsplat-external", "backend_version": "1.5.3", "render_capable": True},
        payload_snapshot={
            "backend_id": "eia-gsplat-external",
            "backend_version": "1.5.3",
            "optimization_state": "optimized",
            "render_capable": True,
            "metric_geometry_status": "arbitrary",
        },
    )


def _nerf() -> _Artifact:
    return _Artifact(
        artifact_id="scene:nerf",
        kind_name="nerf_scene",
        provider_id="qualified-nerf-appearance",
        metadata={"backend_id": "nerfstudio-nerfacto", "backend_version": "1.1.5", "render_capable": True},
        payload_snapshot={
            "backend_id": "nerfstudio-nerfacto",
            "backend_version": "1.1.5",
            "training_state": "trained",
            "render_capable": True,
            "metric_geometry_status": "arbitrary",
        },
    )


def test_f1_builds_generic_gaussian_and_nerf_entries_from_persisted_scene() -> None:
    presentation = build_reconstruction_appearance_presentation(_scene(_nerf(), _gaussian()))
    assert presentation is not None
    assert [item.label for item in presentation.entries] == ["Gaussian", "NeRF"]
    assert presentation.available_text == "Gaussian • NeRF"
    assert presentation.missing_kinds == ()
    assert presentation.canonical_geometry_authoritative is True
    assert presentation.certified_metrology is False
    assert "visualization deferred to F2/F3" in presentation.state_text
    assert presentation.authority_text == "Appearance only • canonical geometry remains authoritative"


def test_f1_entry_uses_persisted_backend_and_render_metadata_without_probing() -> None:
    presentation = build_reconstruction_appearance_presentation(_scene(_gaussian()))
    assert presentation is not None
    entry = presentation.entries[0]
    assert entry.provider_text == "eia-gsplat-external • v1.5.3"
    assert entry.render_text == "Declared render-capable"
    assert entry.scale_text == "Arbitrary scene gauge • appearance only"
    assert "Optimized" in entry.state_text
    assert presentation.missing_kinds == ("nerf_scene",)
    assert "NeRF missing" in presentation.state_text


def test_f1_no_persistent_scene_is_explicitly_empty() -> None:
    assert build_reconstruction_appearance_presentation(None) is None
    empty = empty_appearance_representation_text()
    assert empty["available"] == "No persistent appearance scene"
    assert "canonical geometry remains authoritative" in empty["authority"]


def test_f1_rejects_neural_artifact_authority_drift() -> None:
    bad = _gaussian()
    bad.authority_name = "canonical_geometry"
    with pytest.raises(ValueError, match="appearance-only authority"):
        build_reconstruction_appearance_presentation(_scene(bad))


def test_f1_rejects_certified_metrology_claims() -> None:
    bad = _nerf()
    bad.certified_metrology = True
    with pytest.raises(ValueError, match="forbidden geometry/metrology authority"):
        build_reconstruction_appearance_presentation(_scene(bad))


def test_f1_entry_contract_rejects_canonical_geometry_claim() -> None:
    with pytest.raises(ValueError, match="canonical geometry authority"):
        AppearanceRepresentationEntry(
            artifact_id="x",
            representation_kind="future_appearance",
            label="Future Appearance",
            provider_id="provider",
            backend_id="backend",
            backend_version="1",
            render_capable=True,
            metric_geometry_status="arbitrary",
            state_text="Ready",
            canonical_geometry_authoritative=True,
        )


def test_f1_ignores_nonappearance_artifacts_and_stays_object_agnostic() -> None:
    geometry = _Artifact(
        artifact_id="scene:mesh",
        kind_name="surface_mesh",
        provider_id="mesh",
        metadata={},
        payload_snapshot={},
        authority_name="canonical_geometry",
        canonical_geometry_authoritative=True,
    )
    presentation = build_reconstruction_appearance_presentation(_scene(geometry, _gaussian()))
    assert presentation is not None
    assert len(presentation.entries) == 1
    assert presentation.entries[0].representation_kind == "gaussian_scene"
