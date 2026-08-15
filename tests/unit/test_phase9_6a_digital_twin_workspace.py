from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest

from app.reconstruction_digital_twin_workspace import (
    DigitalTwinWorkspaceState,
    build_digital_twin_workspace_presentation,
)

ROOT = Path(__file__).resolve().parents[2]


def _canonical(*, scene_state: str = "ready", scale_state: str = "not_qualified"):
    return SimpleNamespace(
        available=True,
        scene_id="scene-96a",
        scene_state=scene_state,
        canonical_geometry_authoritative=True,
        calibration_available=False,
        scale_state=SimpleNamespace(value=scale_state),
        scale_text="Not qualified" if scale_state == "not_qualified" else "Qualified metric • ±2.00% scale",
        certified_metrology=False,
    )


def _appearance(*kinds: str):
    return SimpleNamespace(
        entries=tuple(
            SimpleNamespace(
                representation_kind=kind,
                canonical_geometry_authoritative=False,
                certified_metrology=False,
            )
            for kind in kinds
        )
    )


def test_96a_no_scene_fails_closed_without_invented_authority() -> None:
    presentation = build_digital_twin_workspace_presentation()
    assert presentation.state is DigitalTwinWorkspaceState.NO_SCENE
    assert presentation.active_view_representation == "canonical_geometry"
    assert presentation.canonical_geometry_authoritative is False
    assert presentation.gaussian_available is False
    assert presentation.nerf_available is False
    assert presentation.certified_metrology is False
    assert presentation.resource_text == "No neural appearance resource required"


def test_96a_composes_semantics_and_optional_appearance_around_canonical_authority() -> None:
    presentation = build_digital_twin_workspace_presentation(
        _canonical(),
        canonical_surface_loaded=True,
        semantic_context_available=True,
        appearance_presentation=_appearance("gaussian_scene", "nerf_scene"),
        active_view_representation="nerf_scene",
        nerf_runtime_active=True,
    )
    assert presentation.state is DigitalTwinWorkspaceState.COMPOSED
    assert presentation.canonical_geometry_authoritative is True
    assert presentation.composition_text == "Canonical • Semantics • Gaussian • NeRF"
    assert presentation.active_view_text == "NeRF appearance • canonical spatial authority"
    assert presentation.authority_text == "Canonical geometry • picking + measurement"
    assert "QC/reference separate" in presentation.safety_text
    assert presentation.resource_text == "NeRF isolated worker active • Gaussian not active"
    payload = presentation.as_dict()
    assert payload["measurement_authority"] == "canonical_geometry"
    assert payload["appearance_authority"] == "appearance_only"
    assert payload["qc_reference_authority"] == "separate_evidence_domain"


def test_96a_preserves_arbitrary_scale_and_never_promotes_calibration_or_metrology() -> None:
    canonical = _canonical(scale_state="not_qualified")
    canonical.calibration_available = True
    presentation = build_digital_twin_workspace_presentation(canonical)
    assert presentation.scale_state == "not_qualified"
    assert presentation.calibration_available is True
    assert presentation.certified_metrology is False
    assert presentation.scale_text == "Not qualified"


def test_96a_degraded_scene_retains_canonical_authority() -> None:
    presentation = build_digital_twin_workspace_presentation(
        _canonical(scene_state="degraded"),
        appearance_presentation=_appearance("gaussian_scene"),
    )
    assert presentation.state is DigitalTwinWorkspaceState.DEGRADED
    assert presentation.canonical_geometry_authoritative is True
    assert "canonical authority retained" in presentation.workspace_text.lower()


def test_96a_rejects_cross_representation_or_authority_conflicts() -> None:
    with pytest.raises(ValueError, match="Gaussian view"):
        build_digital_twin_workspace_presentation(
            _canonical(),
            active_view_representation="gaussian_scene",
        )
    with pytest.raises(ValueError, match="canonical geometry authority"):
        build_digital_twin_workspace_presentation(
            SimpleNamespace(
                available=True,
                scene_id="bad",
                scene_state="ready",
                canonical_geometry_authoritative=False,
                certified_metrology=False,
            )
        )
    with pytest.raises(ValueError, match="certified metrology"):
        build_digital_twin_workspace_presentation(
            _canonical(),
            appearance_presentation=SimpleNamespace(
                entries=(SimpleNamespace(
                    representation_kind="nerf_scene",
                    canonical_geometry_authoritative=False,
                    certified_metrology=True,
                ),)
            ),
        )


def test_96a_workspace_module_has_no_qt_canvas_or_heavy_backend_imports() -> None:
    path = ROOT / "app/reconstruction_digital_twin_workspace.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    imports: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(alias.name.split(".", 1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.add(node.module.split(".", 1)[0])
    for forbidden in (
        "PyQt6", "vispy", "torch", "pycolmap", "gsplat", "nerfstudio",
        "tinycudann", "depth_anything_3",
    ):
        assert forbidden not in imports
