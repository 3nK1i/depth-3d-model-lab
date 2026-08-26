from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from core.reconstruction.errors import ReconstructionValidationError
from scripts.run_b2_sfm_dense_real_alignment import (
    _adapt_da3_depth_to_sfm_pixel_space,
    _select_registered_keyframe,
)


def _fake_sfm(*, requested_exists: bool = True):
    camera_a = SimpleNamespace(keyframe_id="kf-a")
    camera_b = SimpleNamespace(keyframe_id="kf-b")
    workspace_a = SimpleNamespace(keyframe_id="kf-a")
    workspace_b = SimpleNamespace(keyframe_id="kf-b")
    observations = []
    for index in range(3):
        observations.append(
            SimpleNamespace(
                observations=(SimpleNamespace(keyframe_id="kf-a"), SimpleNamespace(keyframe_id="kf-b")),
            )
        )
    observations.append(
        SimpleNamespace(observations=(SimpleNamespace(keyframe_id="kf-b"),))
    )
    cameras = (camera_a, camera_b) if requested_exists else (camera_a,)
    images = (workspace_a, workspace_b)
    return SimpleNamespace(
        cameras=cameras,
        landmarks=tuple(observations),
        workspace=SimpleNamespace(images=images),
    )


def test_select_registered_keyframe_prefers_strongest_sparse_support() -> None:
    sfm = _fake_sfm()
    camera, workspace, count = _select_registered_keyframe(sfm)
    assert camera.keyframe_id == "kf-b"
    assert workspace.keyframe_id == "kf-b"
    assert count == 4


def test_select_registered_keyframe_honours_explicit_registered_request() -> None:
    sfm = _fake_sfm()
    camera, workspace, count = _select_registered_keyframe(sfm, requested_keyframe_id="kf-a")
    assert camera.keyframe_id == "kf-a"
    assert workspace.keyframe_id == "kf-a"
    assert count == 3


def test_select_registered_keyframe_rejects_unregistered_request() -> None:
    sfm = _fake_sfm(requested_exists=False)
    with pytest.raises(ReconstructionValidationError, match="not registered"):
        _select_registered_keyframe(sfm, requested_keyframe_id="kf-b")


def test_pixel_adapter_preserves_native_shared_grid() -> None:
    depth = np.full((4, 6), 2.5, dtype=np.float32)
    adapted, report = _adapt_da3_depth_to_sfm_pixel_space(
        depth,
        sfm_width=6,
        sfm_height=4,
    )
    assert adapted.shape == (4, 6)
    assert np.allclose(adapted, depth)
    assert report["mode"] == "identity"
    assert report["relative_scale_anisotropy"] == 0.0
    assert adapted.flags.writeable is False


def test_pixel_adapter_resizes_near_uniform_da3_grid_to_sfm_grid() -> None:
    depth = np.arange(1, 1 + 20 * 30, dtype=np.float32).reshape(20, 30)
    adapted, report = _adapt_da3_depth_to_sfm_pixel_space(
        depth,
        sfm_width=60,
        sfm_height=40,
    )
    assert adapted.shape == (40, 60)
    assert np.isfinite(adapted).all()
    assert np.all(adapted > 0.0)
    assert report["mode"] == "aspect_preserving_resize_to_sfm_grid"
    assert report["scale_x"] == pytest.approx(0.5)
    assert report["scale_y"] == pytest.approx(0.5)
    assert report["validation_only"] is True


def test_pixel_adapter_rejects_large_anisotropic_resize_assumption() -> None:
    depth = np.ones((20, 20), dtype=np.float32)
    with pytest.raises(ReconstructionValidationError, match="not sufficiently uniform"):
        _adapt_da3_depth_to_sfm_pixel_space(
            depth,
            sfm_width=100,
            sfm_height=40,
        )


def test_pixel_adapter_rejects_invalid_depth() -> None:
    depth = np.ones((20, 30), dtype=np.float32)
    depth[0, 0] = np.nan
    with pytest.raises(ReconstructionValidationError, match="finite and strictly positive"):
        _adapt_da3_depth_to_sfm_pixel_space(
            depth,
            sfm_width=60,
            sfm_height=40,
        )
