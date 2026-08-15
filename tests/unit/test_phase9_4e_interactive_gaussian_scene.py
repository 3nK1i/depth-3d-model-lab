from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest

from app.reconstruction_gaussian_interactive import load_interactive_gaussian_scene
from core.reconstruction.errors import ReconstructionValidationError
from tests.phase9_4e_helpers import make_phase9_4e_persistent_scene


def test_94e_loads_integrity_bound_persisted_gaussian_without_authority_promotion(tmp_path: Path) -> None:
    scene, _manifest, artifact = make_phase9_4e_persistent_scene(tmp_path)
    presentation = load_interactive_gaussian_scene(scene)
    assert presentation.scene_id == scene.scene_id
    assert presentation.artifact_id == scene.gaussian_artifact.artifact_id
    assert presentation.scene_path == artifact.scene_path
    assert presentation.total_gaussian_count == 64
    assert presentation.display_gaussian_count == 64
    assert presentation.positions_reconstruction.shape == (64, 3)
    assert presentation.colors_rgba.shape == (64, 4)
    assert presentation.sizes_px.shape == (64,)
    assert np.isfinite(presentation.positions_reconstruction).all()
    assert np.all((presentation.colors_rgba >= 0.0) & (presentation.colors_rgba <= 1.0))
    assert presentation.canonical_geometry_authoritative is False
    assert presentation.metric_scale_authoritative is False
    assert presentation.measurement_authoritative is False
    assert presentation.qc_authoritative is False
    assert presentation.certified_metrology is False


def test_94e_bounded_display_sampling_is_deterministic(tmp_path: Path) -> None:
    scene, _manifest, _artifact = make_phase9_4e_persistent_scene(tmp_path)
    first = load_interactive_gaussian_scene(scene, max_gaussians=16)
    second = load_interactive_gaussian_scene(scene, max_gaussians=16)
    assert first.total_gaussian_count == 64
    assert first.display_gaussian_count == 16
    assert np.array_equal(first.positions_reconstruction, second.positions_reconstruction)
    assert np.array_equal(first.colors_rgba, second.colors_rgba)


def test_94e_persistent_scene_tampering_is_rejected_before_tensor_load(tmp_path: Path) -> None:
    scene, _manifest, artifact = make_phase9_4e_persistent_scene(tmp_path)
    Path(artifact.scene_path).write_bytes(b"tampered")
    with pytest.raises(ReconstructionValidationError, match="size changed|checksum"):
        load_interactive_gaussian_scene(scene)


def test_94e_does_not_import_heavy_gaussian_runtime_modules(tmp_path: Path) -> None:
    scene, _manifest, _artifact = make_phase9_4e_persistent_scene(tmp_path)
    before = {name for name in sys.modules if name.startswith(("gsplat", "diff_gaussian_rasterization"))}
    load_interactive_gaussian_scene(scene)
    after = {name for name in sys.modules if name.startswith(("gsplat", "diff_gaussian_rasterization"))}
    assert after == before
