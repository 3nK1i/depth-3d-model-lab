"""The real VisPy 3D canvas (Phase 4), replacing Phase 1's painted
placeholder.

Created exactly once and never rebuilt per frame (spec §10): the
SceneCanvas, the TurntableCamera, the Markers point-cloud visual and
the GridLines floor are all built in ``__init__``;
every subsequent frame only calls :meth:`update_points`, which updates
the existing Markers visual's buffers via ``set_data`` — verified to
update in place rather than recreate the visual.

The camera is preserved across live updates: :meth:`fit_scene` (which
actually moves the camera) is only ever called by the owner when it
decides to — this class does not fit automatically on every
:meth:`update_points` call, so orbit/pan/zoom state survives live
geometry updates exactly as spec §10 requires.
"""

from __future__ import annotations

import math

import numpy as np
from PyQt6.QtCore import QObject, pyqtSignal
from vispy import scene

DEFAULT_POINT_SIZE = 4.0
DEFAULT_CAMERA_DISTANCE = 5.0
DEFAULT_FOV = 45.0


class PointCloudCanvas(QObject):
    """Owns the VisPy SceneCanvas, camera, and point-cloud visual."""

    interaction_started = pyqtSignal()
    interaction_ended = pyqtSignal()

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)

        self._canvas = scene.SceneCanvas(keys="interactive", bgcolor="#090b0e", show=False)
        self._view = self._canvas.central_widget.add_view()
        self._view.camera = scene.cameras.TurntableCamera(
            fov=DEFAULT_FOV, distance=DEFAULT_CAMERA_DISTANCE
        )

        self._grid = scene.visuals.GridLines(
            color=(1, 1, 1, 0.08),
            parent=self._view.scene,
        )

        self._markers = scene.visuals.Markers(parent=self._view.scene)
        self._point_size = DEFAULT_POINT_SIZE
        self._has_points = False
        self._points_bounds: tuple[tuple[float, float], ...] | None = None

        self._is_interacting = False
        self._canvas.events.mouse_press.connect(self._on_mouse_press)
        self._canvas.events.mouse_release.connect(self._on_mouse_release)

    @property
    def native(self):
        """The Qt-native widget — passed to ``ViewportPanel.attach_canvas``."""
        return self._canvas.native

    # -- Data update (called every frame; never recreates anything) --------

    def update_points(self, points: np.ndarray, colors: np.ndarray) -> None:
        """Update the point cloud in place. ``points`` is (N, 3) float in
        the standard camera-space convention (X-right, Y-down, Z-forward/
        depth) that :mod:`core.geometry.projection` produces. ``colors``
        is (N, 3) uint8 or float in [0, 1].

        Remapped here (not upstream in core.geometry) to VisPy's Z-up
        world convention — TurntableCamera and GridLines both assume the
        floor is the XY plane with Z as "up". Without this remap, the
        cloud's real depth structure gets viewed from an unrelated
        angle and can collapse to a near-invisible sliver hugging one
        edge of the viewport, which is exactly what was observed.
        Keeping the remap local to this VisPy wrapper means every other
        consumer of core.geometry.projection's output (PLY export, unit
        tests, a future different renderer) still sees plain camera-
        space coordinates.
        """
        if points.shape[0] == 0:
            return
        world_points = np.empty_like(points)
        world_points[:, 0] = points[:, 0]  # right stays right
        world_points[:, 1] = points[:, 2]  # depth/forward -> ground-plane "away"
        world_points[:, 2] = -points[:, 1]  # image-down, negated -> world "up"

        color_float = colors.astype(np.float32) / 255.0 if colors.dtype == np.uint8 else colors
        self._markers.set_data(
            pos=world_points, face_color=color_float, size=self._point_size, edge_width=0
        )
        self._has_points = True
        self._points_bounds = (
            (float(world_points[:, 0].min()), float(world_points[:, 0].max())),
            (float(world_points[:, 1].min()), float(world_points[:, 1].max())),
            (float(world_points[:, 2].min()), float(world_points[:, 2].max())),
        )

    def clear_points(self) -> None:
        self._markers.set_data(pos=np.zeros((0, 3), dtype=np.float32))
        self._has_points = False
        self._points_bounds = None

    # -- Camera control ------------------------------------------------------

    def fit_scene(self) -> None:
        """Move the camera to frame the current geometry. Only called by
        the owner on the first valid cloud or an explicit Reset View —
        never automatically on every update (spec §10).

        Computes the framing directly from the point cloud's own
        tracked bounding box rather than calling VisPy's
        ``camera.set_range()`` — that built-in heuristic
        (TurntableCamera._set_range) always fits the FULL 3D diagonal
        of the bounds on the assumption the object must stay framed
        "under any rotation" (``scale_factor = max(rxs, rys)`` where
        ``rys`` includes all three axes and is mathematically always
        >= ``rxs``), and only aspect-corrects two of the three axes for
        a wide viewport, never the vertical one. For a very wide/short
        viewport like this one, that systematically over-zooms-out —
        which is exactly what made the point cloud render as a thin
        sliver instead of filling the view. A direct
        bounding-sphere-radius/FOV calculation is simple, predictable,
        and correct by construction regardless of the viewport's aspect
        ratio.
        """
        if self._has_points and self._points_bounds is not None:
            (x_min, x_max), (y_min, y_max), (z_min, z_max) = self._points_bounds
            center = ((x_min + x_max) / 2.0, (y_min + y_max) / 2.0, (z_min + z_max) / 2.0)
            radius = 0.5 * math.sqrt(
                (x_max - x_min) ** 2 + (y_max - y_min) ** 2 + (z_max - z_min) ** 2
            )
            radius = max(radius, 0.1)  # avoid a degenerate zero-radius cloud

            half_fov_rad = math.radians(self._view.camera.fov / 2.0)
            distance = (radius / math.tan(half_fov_rad)) * 1.2  # 20% headroom

            self._view.camera.center = center
            self._view.camera.distance = distance
        else:
            self._view.camera.center = (0.0, 0.0, 0.0)
            self._view.camera.distance = DEFAULT_CAMERA_DISTANCE

    def reset_view(self) -> None:
        """Reset the camera to its default framing.

        Resets the EXISTING camera's properties rather than assigning a
        new camera object to the view — reassigning ``view.camera``
        turns out to itself trigger VisPy's scene-bounds computation
        (independent of :meth:`fit_scene`), which crashes when the
        Markers visual has never held real data. Resetting properties
        in place avoids that entirely and is more consistent with this
        whole canvas's "create once, never rebuild" design anyway.
        """
        camera = self._view.camera
        camera.center = (0.0, 0.0, 0.0)
        camera.distance = DEFAULT_CAMERA_DISTANCE
        camera.fov = DEFAULT_FOV
        camera.azimuth = 30.0
        camera.elevation = 30.0
        self.fit_scene()

    # -- Render settings -----------------------------------------------------

    def set_point_size(self, size: float) -> None:
        self._point_size = max(0.5, size)

    def set_background(self, color: str) -> None:
        self._canvas.bgcolor = color

    def set_grid_visible(self, visible: bool) -> None:
        self._grid.visible = visible

    # -- Interaction state (drives adaptive point density) --------------------

    def _on_mouse_press(self, _event) -> None:
        self._is_interacting = True
        self.interaction_started.emit()

    def _on_mouse_release(self, _event) -> None:
        self._is_interacting = False
        self.interaction_ended.emit()

    @property
    def is_interacting(self) -> bool:
        return self._is_interacting

    @property
    def has_points(self) -> bool:
        return self._has_points

    def close(self) -> None:
        self._canvas.close()