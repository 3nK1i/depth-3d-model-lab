"""Shared pytest configuration for the Depth & 3D Model Lab test suite.

Automated Qt tests run without a real display server.  A real VisPy
``SceneCanvas`` creates a ``QOpenGLWidget`` and may enter native OpenGL code
while pytest-qt processes events.  The Qt ``offscreen`` plugin does not provide
an OpenGL framebuffer, and some Mesa/driver combinations terminate the Python
process inside ``glDrawArrays`` instead of raising a Python exception.

UI tests therefore replace only the VisPy canvas wrapper with a small Qt test
double.  This keeps controller/signal/layout behaviour testable while leaving
real OpenGL rendering for normal desktop/manual validation.  Production code is
not modified.
"""

from __future__ import annotations

import os
import sys
import types

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("QT_OPENGL", "software")
os.environ.setdefault("LIBGL_ALWAYS_SOFTWARE", "1")
os.environ.setdefault("DEPTH3D_TEST_FAKE_VISPY", "1")


def _install_fake_vispy_canvas() -> None:
    """Install a Qt-only PointCloudCanvas test double before app collection."""
    if os.environ.get("DEPTH3D_TEST_FAKE_VISPY", "1") != "1":
        return

    module_name = "app.widgets.viewport.vispy_canvas"
    if module_name in sys.modules:
        return

    try:
        import numpy as np
        from PyQt6.QtCore import QObject, Qt, pyqtSignal
        from PyQt6.QtWidgets import QWidget
    except ModuleNotFoundError:
        # Pure unit tests may run in a minimal environment without Qt.
        return

    valid_modes = {"select", "orbit", "pan", "zoom"}
    valid_backgrounds = {"dark", "black", "white", "transparent"}

    class _NativeWidget(QWidget):
        def __init__(self, owner: "PointCloudCanvas") -> None:
            super().__init__()
            self._owner = owner
            self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

        def keyPressEvent(self, event) -> None:  # noqa: N802 - Qt override
            if event.key() == Qt.Key.Key_Escape:
                self._owner.escape_pressed.emit()
                event.accept()
                return
            super().keyPressEvent(event)

    class PointCloudCanvas(QObject):
        """API-compatible, non-OpenGL test double for UI/controller tests."""

        interaction_started = pyqtSignal()
        interaction_ended = pyqtSignal()
        escape_pressed = pyqtSignal()

        def __init__(self, parent: QObject | None = None) -> None:
            super().__init__(parent)
            self._native = _NativeWidget(self)
            self._interaction_mode = "orbit"
            self._point_size = 4.0
            self._smoothing = 0.5
            self._background_name = "dark"
            self._grid_visible = True
            self._has_points = False
            self._is_interacting = False
            self._last_points = None
            self._last_colors = None
            self.fit_calls = 0
            self.reset_calls = 0

        @property
        def native(self):
            return self._native

        def update_points(self, points, colors) -> None:
            self._last_points = np.asarray(points).copy()
            self._last_colors = np.asarray(colors).copy()
            self._has_points = bool(self._last_points.shape[0])

        def clear_points(self) -> None:
            self._last_points = None
            self._last_colors = None
            self._has_points = False

        def fit_scene(self) -> None:
            self.fit_calls += 1

        def reset_view(self) -> None:
            self.reset_calls += 1
            self.fit_scene()

        def set_interaction_mode(self, mode: str) -> None:
            normalized = str(mode).strip().lower()
            if normalized not in valid_modes:
                raise ValueError(f"Unsupported viewport interaction mode: {mode!r}")
            self._interaction_mode = normalized

        @property
        def interaction_mode(self) -> str:
            return self._interaction_mode

        def set_point_size(self, size: float) -> None:
            self._point_size = max(0.5, float(size))

        @property
        def point_size(self) -> float:
            return self._point_size

        def set_smoothing(self, value: float) -> None:
            self._smoothing = float(np.clip(value, 0.0, 1.0))

        @property
        def smoothing(self) -> float:
            return self._smoothing

        def set_background(self, background: str) -> None:
            normalized = str(background).strip().lower()
            if normalized not in valid_backgrounds:
                raise ValueError(f"Unsupported viewport background: {background!r}")
            self._background_name = normalized

        @property
        def background_name(self) -> str:
            return self._background_name

        def set_grid_visible(self, visible: bool) -> None:
            self._grid_visible = bool(visible)

        @property
        def grid_visible(self) -> bool:
            return self._grid_visible

        @property
        def is_interacting(self) -> bool:
            return self._is_interacting

        @property
        def has_points(self) -> bool:
            return self._has_points

        def close(self) -> None:
            self._native.close()

    class ReconstructionCanvas(QObject):
        """Qt-only test double for the Phase 8.8C reconstruction canvas."""

        interaction_started = pyqtSignal()
        interaction_ended = pyqtSignal()
        escape_pressed = pyqtSignal()

        MAX_SPARSE_LANDMARK_DISPLAY_POINTS = 8_192
        MAX_KEYFRAME_DISPLAY_FRUSTUMS = 96
        MAX_OFFLINE_CAMERA_PATH_POINTS = 1_024
        MAX_SEMANTIC_OBJECT_DISPLAY = 256

        def __init__(self, parent: QObject | None = None) -> None:
            super().__init__(parent)
            self._native = _NativeWidget(self)
            self._interaction_mode = "orbit"
            self._grid_visible = True
            self._axes_visible = True
            self._trajectory_visible = True
            self._trajectory_positions = []
            self._trajectory_display_count = 0
            self._sparse_landmarks_visible = True
            self._sparse_landmark_positions = np.zeros((0, 3), dtype=np.float32)
            self._sparse_landmark_total_count = 0
            self._sparse_landmark_display_count = 0
            self._sparse_map_revision = None
            self._sparse_landmark_colors = None
            self._offline_visualization_loaded = False
            self._offline_session_id = None
            self._offline_scale_status = None
            self._offline_camera_matrices = np.zeros((0, 4, 4), dtype=np.float64)
            self._offline_camera_positions = np.zeros((0, 3), dtype=np.float32)
            self._offline_camera_ids = ()
            self._offline_camera_path_visible = False
            self._offline_camera_path_display_count = 0
            self._offline_sparse_envelope_visible_requested = True
            self._offline_sparse_envelope_available = False
            self._offline_sparse_envelope_min = None
            self._offline_sparse_envelope_max = None
            self._semantic_objects_visible = True
            self._semantic_objects_available = False
            self._semantic_bounds_min = np.zeros((0, 3), dtype=np.float64)
            self._semantic_bounds_max = np.zeros((0, 3), dtype=np.float64)
            self._semantic_centroids_viewer = np.zeros((0, 3), dtype=np.float32)
            self._semantic_object_ids = ()
            self._semantic_object_total_count = 0
            self._selected_semantic_object_index = None
            self._selected_semantic_object_visible = False
            self._selected_offline_camera_index = None
            self._selected_offline_camera_visible = False
            self._keyframes_visible = True
            self._keyframe_matrices = np.zeros((0, 4, 4), dtype=np.float64)
            self._keyframe_positions = np.zeros((0, 3), dtype=np.float32)
            self._keyframe_ids = ()
            self._keyframe_total_count = 0
            self._keyframe_display_count = 0
            self._keyframe_map_revision = None
            # C3 production axes gained scene-space arrowheads.  The Qt-only
            # test double keeps lightweight visibility sentinels so UI tests
            # can validate the same presentation contract without OpenGL.
            self._axis_x_head = types.SimpleNamespace(visible=True)
            self._axis_y_head = types.SimpleNamespace(visible=True)
            self._axis_z_head = types.SimpleNamespace(visible=True)
            self._world_axis_length = 0.28
            self._current_camera_position = None
            self._has_pose = False
            self._is_interacting = False
            self._intrinsics_matrix = None
            self._image_size = None
            self.fit_calls = 0
            self.focus_calls = 0
            self.reset_calls = 0
            self.closed = False

        @property
        def native(self):
            return self._native

        def configure_camera_geometry(self, *, intrinsics_matrix, image_size) -> None:
            self._intrinsics_matrix = None if intrinsics_matrix is None else np.asarray(intrinsics_matrix).copy()
            self._image_size = image_size

        def update_live_pose(self, T_reconstruction_from_camera) -> None:
            from app.reconstruction_visualization import camera_position_in_viewer

            position = camera_position_in_viewer(T_reconstruction_from_camera).astype(np.float32)
            self._current_camera_position = position
            self._trajectory_positions.append(position.copy())
            self._trajectory_display_count = len(self._trajectory_positions)
            if len(self._trajectory_positions) >= 2:
                from app.reconstruction_visualization import adaptive_world_axis_length
                self._world_axis_length = adaptive_world_axis_length(
                    np.asarray(self._trajectory_positions, dtype=np.float64)
                )
            self._has_pose = True

        def clear_live_state(self) -> None:
            self._trajectory_positions.clear()
            self._trajectory_display_count = 0
            self._sparse_landmark_positions = np.zeros((0, 3), dtype=np.float32)
            self._sparse_landmark_total_count = 0
            self._sparse_landmark_display_count = 0
            self._sparse_map_revision = None
            self._sparse_landmark_colors = None
            self._offline_visualization_loaded = False
            self._offline_session_id = None
            self._offline_scale_status = None
            self._offline_camera_matrices = np.zeros((0, 4, 4), dtype=np.float64)
            self._offline_camera_positions = np.zeros((0, 3), dtype=np.float32)
            self._offline_camera_ids = ()
            self._offline_camera_path_visible = False
            self._offline_camera_path_display_count = 0
            self._offline_sparse_envelope_available = False
            self._offline_sparse_envelope_min = None
            self._offline_sparse_envelope_max = None
            self.clear_semantic_objects()
            self._selected_offline_camera_index = None
            self._selected_offline_camera_visible = False
            self._keyframe_matrices = np.zeros((0, 4, 4), dtype=np.float64)
            self._keyframe_positions = np.zeros((0, 3), dtype=np.float32)
            self._keyframe_ids = ()
            self._keyframe_total_count = 0
            self._keyframe_display_count = 0
            self._keyframe_map_revision = None
            # Visibility preferences intentionally survive a new live session,
            # matching the production C2/C3 canvas behavior.
            self._world_axis_length = 0.28
            self._current_camera_position = None
            self._has_pose = False

        def set_trajectory_visible(self, visible: bool) -> None:
            self._trajectory_visible = bool(visible)

        def update_sparse_landmarks(
            self,
            points_reconstruction,
            *,
            colors_rgb=None,
            total_count: int | None = None,
            revision: int | None = None,
        ) -> None:
            from app.reconstruction_visualization import (
                adaptive_world_axis_length,
                sparse_landmark_display_indices,
                sparse_landmarks_to_viewer,
            )

            points = np.asarray(points_reconstruction, dtype=np.float64)
            if points.size == 0:
                points = np.zeros((0, 3), dtype=np.float64)
            if points.ndim != 2 or points.shape[1:] != (3,) or not np.isfinite(points).all():
                raise ValueError("points_reconstruction must be a finite (N, 3) array.")
            viewer = sparse_landmarks_to_viewer(points)
            indices = sparse_landmark_display_indices(
                viewer.shape[0], max_points=self.MAX_SPARSE_LANDMARK_DISPLAY_POINTS
            )
            display = np.ascontiguousarray(viewer[indices])
            self._sparse_landmark_positions = display
            if colors_rgb is None:
                self._sparse_landmark_colors = None
            else:
                colors = np.asarray(colors_rgb)
                if colors.size == 0:
                    colors = np.zeros((0, 3), dtype=np.uint8)
                if colors.ndim != 2 or colors.shape != points.shape:
                    raise ValueError("colors_rgb must match points_reconstruction shape.")
                self._sparse_landmark_colors = np.asarray(colors[indices]).copy()
            self._sparse_landmark_total_count = max(
                int(total_count) if total_count is not None else int(points.shape[0]),
                int(points.shape[0]),
            )
            self._sparse_landmark_display_count = int(display.shape[0])
            self._sparse_map_revision = None if revision is None else int(revision)

            groups = []
            if self._trajectory_positions:
                groups.append(np.asarray(self._trajectory_positions, dtype=np.float64))
            if self._sparse_landmark_positions.size:
                groups.append(np.asarray(self._sparse_landmark_positions, dtype=np.float64))
            if groups:
                self._world_axis_length = adaptive_world_axis_length(np.vstack(groups))

        def show_offline_sparse_reconstruction(
            self,
            *,
            session_id: str,
            scale_status: str,
            points_reconstruction,
            camera_matrices_reconstruction,
            camera_ids,
            landmark_colors_rgb=None,
            landmark_total: int | None = None,
            registered_camera_total: int | None = None,
        ) -> None:
            self.clear_live_state()
            self.update_sparse_landmarks(
                points_reconstruction,
                colors_rgb=landmark_colors_rgb,
                total_count=landmark_total,
                revision=None,
            )
            matrices = np.asarray(camera_matrices_reconstruction, dtype=np.float64)
            ids = tuple(str(value) for value in camera_ids)
            from app.reconstruction_visualization import keyframe_centers_in_viewer
            self._offline_camera_matrices = np.asarray(matrices).copy()
            self._offline_camera_positions = keyframe_centers_in_viewer(self._offline_camera_matrices)
            self._offline_camera_ids = ids
            self.update_keyframe_poses(
                matrices,
                keyframe_ids=ids,
                total_count=registered_camera_total,
                revision=None,
            )
            self._offline_visualization_loaded = bool(self.has_sparse_map or self.has_keyframes)
            self._offline_camera_path_display_count = min(
                int(self._offline_camera_positions.shape[0]), self.MAX_OFFLINE_CAMERA_PATH_POINTS
            )
            self._offline_camera_path_visible = bool(
                self._offline_visualization_loaded and self._keyframes_visible
                and self._offline_camera_positions.shape[0] >= 2
            )
            self._offline_session_id = str(session_id)
            self._offline_scale_status = str(scale_status)
            self._has_pose = False
            self._current_camera_position = None

        def set_offline_sparse_envelope(self, minimum_reconstruction, maximum_reconstruction) -> None:
            lower = np.asarray(minimum_reconstruction, dtype=np.float64)
            upper = np.asarray(maximum_reconstruction, dtype=np.float64)
            if lower.shape != (3,) or upper.shape != (3,):
                raise ValueError("Offline sparse envelope bounds must be 3-vectors.")
            if not np.isfinite(lower).all() or not np.isfinite(upper).all() or np.any(upper < lower):
                raise ValueError("Offline sparse envelope bounds must be finite and ordered.")
            self._offline_sparse_envelope_min = lower.copy()
            self._offline_sparse_envelope_max = upper.copy()
            self._offline_sparse_envelope_available = True

        def clear_offline_sparse_envelope(self) -> None:
            self._offline_sparse_envelope_available = False
            self._offline_sparse_envelope_min = None
            self._offline_sparse_envelope_max = None

        def set_offline_sparse_envelope_visible(self, visible: bool) -> None:
            self._offline_sparse_envelope_visible_requested = bool(visible)

        def set_semantic_objects(
            self,
            *,
            bounds_min_reconstruction,
            bounds_max_reconstruction,
            centroids_reconstruction,
            object_ids,
            total_count: int | None = None,
        ) -> None:
            from app.offline_reconstruction_semantics import semantic_centroid_in_viewer

            lower = np.asarray(bounds_min_reconstruction, dtype=np.float64)
            upper = np.asarray(bounds_max_reconstruction, dtype=np.float64)
            centroids = np.asarray(centroids_reconstruction, dtype=np.float64)
            if lower.size == 0:
                lower = np.zeros((0, 3), dtype=np.float64)
            if upper.shape != lower.shape or centroids.shape != lower.shape:
                raise ValueError("Semantic bounds and centroids must share shape (N, 3).")
            ids = tuple(str(value).strip() for value in object_ids)
            if len(ids) != lower.shape[0]:
                raise ValueError("object_ids must match semantic objects.")
            self._semantic_bounds_min = lower.copy()
            self._semantic_bounds_max = upper.copy()
            self._semantic_centroids_viewer = np.asarray(
                [semantic_centroid_in_viewer(value) for value in centroids], dtype=np.float32
            ).reshape((-1, 3))
            self._semantic_object_ids = ids
            self._semantic_object_total_count = max(
                int(total_count) if total_count is not None else int(lower.shape[0]),
                int(lower.shape[0]),
            )
            self._semantic_objects_available = bool(lower.shape[0])
            self._selected_semantic_object_index = None
            self._selected_semantic_object_visible = False

        def clear_semantic_objects(self, *, update_canvas: bool = True) -> None:
            self._semantic_objects_available = False
            self._semantic_bounds_min = np.zeros((0, 3), dtype=np.float64)
            self._semantic_bounds_max = np.zeros((0, 3), dtype=np.float64)
            self._semantic_centroids_viewer = np.zeros((0, 3), dtype=np.float32)
            self._semantic_object_ids = ()
            self._semantic_object_total_count = 0
            self._selected_semantic_object_index = None
            self._selected_semantic_object_visible = False

        def set_semantic_objects_visible(self, visible: bool) -> None:
            self._semantic_objects_visible = bool(visible)
            self._selected_semantic_object_visible = bool(
                self._semantic_objects_visible
                and self._semantic_objects_available
                and self._selected_semantic_object_index is not None
            )

        def select_semantic_object(self, object_index: int) -> None:
            index = int(object_index)
            if not self._semantic_objects_available:
                raise ValueError("No semantic object presentation is loaded.")
            if index < 0 or index >= self._semantic_bounds_min.shape[0]:
                raise IndexError("Semantic object index is out of range.")
            self._selected_semantic_object_index = index
            self._selected_semantic_object_visible = self._semantic_objects_visible

        def clear_semantic_object_selection(self) -> None:
            self._selected_semantic_object_index = None
            self._selected_semantic_object_visible = False

        def focus_selected_semantic_object(self) -> None:
            self.focus_calls += 1

        def select_offline_camera(self, camera_index: int) -> None:
            index = int(camera_index)
            if not self._offline_visualization_loaded:
                raise ValueError("No offline sparse reconstruction is loaded.")
            if index < 0 or index >= self._offline_camera_matrices.shape[0]:
                raise IndexError("Offline registered camera index is out of range.")
            self._selected_offline_camera_index = index
            self._selected_offline_camera_visible = self._keyframes_visible

        def clear_offline_camera_selection(self) -> None:
            self._selected_offline_camera_index = None
            self._selected_offline_camera_visible = False

        def focus_selected_offline_camera(self) -> None:
            if self._selected_offline_camera_index is None:
                self.focus_current_camera()
                return
            self.focus_calls += 1

        def set_sparse_landmarks_visible(self, visible: bool) -> None:
            self._sparse_landmarks_visible = bool(visible)

        def update_keyframe_poses(
            self,
            matrices_reconstruction,
            *,
            keyframe_ids=(),
            total_count: int | None = None,
            revision: int | None = None,
        ) -> None:
            from app.reconstruction_visualization import (
                adaptive_world_axis_length,
                keyframe_centers_in_viewer,
                keyframe_display_indices,
            )

            matrices = np.asarray(matrices_reconstruction, dtype=np.float64)
            if matrices.size == 0:
                matrices = np.zeros((0, 4, 4), dtype=np.float64)
            if matrices.ndim != 3 or matrices.shape[1:] != (4, 4) or not np.isfinite(matrices).all():
                raise ValueError("matrices_reconstruction must be a finite (N, 4, 4) array.")
            ids = tuple(str(value).strip() for value in keyframe_ids)
            if ids and len(ids) != matrices.shape[0]:
                raise ValueError("keyframe_ids must match matrices_reconstruction length.")
            indices = keyframe_display_indices(
                matrices.shape[0], max_keyframes=self.MAX_KEYFRAME_DISPLAY_FRUSTUMS
            )
            self._keyframe_matrices = np.ascontiguousarray(matrices[indices])
            self._keyframe_positions = keyframe_centers_in_viewer(self._keyframe_matrices)
            self._keyframe_ids = tuple(ids[index] for index in indices) if ids else ()
            self._keyframe_total_count = max(
                int(total_count) if total_count is not None else int(matrices.shape[0]),
                int(matrices.shape[0]),
            )
            self._keyframe_display_count = int(self._keyframe_matrices.shape[0])
            self._keyframe_map_revision = None if revision is None else int(revision)
            groups = []
            if self._trajectory_positions:
                groups.append(np.asarray(self._trajectory_positions, dtype=np.float64))
            if self._sparse_landmark_positions.size:
                groups.append(np.asarray(self._sparse_landmark_positions, dtype=np.float64))
            if self._keyframe_positions.size:
                groups.append(np.asarray(self._keyframe_positions, dtype=np.float64))
            if groups:
                self._world_axis_length = adaptive_world_axis_length(np.vstack(groups))

        def set_keyframes_visible(self, visible: bool) -> None:
            self._keyframes_visible = bool(visible)
            self._offline_camera_path_visible = bool(
                self._keyframes_visible and self._offline_visualization_loaded
                and self._offline_camera_positions.shape[0] >= 2
            )
            self._selected_offline_camera_visible = bool(
                self._keyframes_visible and self._selected_offline_camera_index is not None
            )

        def fit_scene(self) -> None:
            self.fit_calls += 1

        def focus_current_camera(self) -> None:
            self.focus_calls += 1

        def reset_view(self) -> None:
            self.reset_calls += 1

        def set_interaction_mode(self, mode: str) -> None:
            normalized = str(mode).strip().lower()
            if normalized not in valid_modes:
                raise ValueError(f"Unsupported viewport interaction mode: {mode!r}")
            self._interaction_mode = normalized

        def set_grid_visible(self, visible: bool) -> None:
            self._grid_visible = bool(visible)

        def set_axes_visible(self, visible: bool) -> None:
            self._axes_visible = bool(visible)
            self._axis_x_head.visible = self._axes_visible
            self._axis_y_head.visible = self._axes_visible
            self._axis_z_head.visible = self._axes_visible

        @property
        def interaction_mode(self) -> str:
            return self._interaction_mode

        @property
        def trajectory_count(self) -> int:
            return len(self._trajectory_positions)

        @property
        def trajectory_visible(self) -> bool:
            return self._trajectory_visible

        @property
        def trajectory_display_count(self) -> int:
            return self._trajectory_display_count

        @property
        def sparse_landmarks_visible(self) -> bool:
            return self._sparse_landmarks_visible

        @property
        def sparse_landmark_count(self) -> int:
            return self._sparse_landmark_total_count

        @property
        def sparse_landmark_display_count(self) -> int:
            return self._sparse_landmark_display_count

        @property
        def sparse_map_revision(self) -> int | None:
            return self._sparse_map_revision

        @property
        def has_sparse_map(self) -> bool:
            return self._sparse_landmark_display_count > 0

        @property
        def keyframes_visible(self) -> bool:
            return self._keyframes_visible

        @property
        def keyframe_count(self) -> int:
            return self._keyframe_total_count

        @property
        def keyframe_display_count(self) -> int:
            return self._keyframe_display_count

        @property
        def keyframe_map_revision(self) -> int | None:
            return self._keyframe_map_revision

        @property
        def keyframe_ids(self):
            return self._keyframe_ids

        @property
        def has_keyframes(self) -> bool:
            return self._keyframe_display_count > 0

        @property
        def offline_registered_camera_count(self) -> int:
            return int(self._offline_camera_matrices.shape[0])

        @property
        def offline_camera_ids(self):
            return self._offline_camera_ids

        @property
        def offline_camera_path_visible(self) -> bool:
            return bool(self._offline_camera_path_visible)

        @property
        def offline_camera_path_display_count(self) -> int:
            return int(self._offline_camera_path_display_count)

        @property
        def selected_offline_camera_index(self):
            return self._selected_offline_camera_index

        @property
        def selected_offline_camera_id(self):
            index = self._selected_offline_camera_index
            if index is None or index < 0 or index >= len(self._offline_camera_ids):
                return None
            return self._offline_camera_ids[index]

        @property
        def selected_offline_camera_visible(self) -> bool:
            return bool(self._selected_offline_camera_visible)

        @property
        def offline_sparse_envelope_visible(self) -> bool:
            return bool(
                self._offline_sparse_envelope_visible_requested
                and self._offline_sparse_envelope_available
                and self._offline_visualization_loaded
            )

        @property
        def offline_sparse_envelope_available(self) -> bool:
            return bool(self._offline_sparse_envelope_available)

        @property
        def offline_sparse_envelope_bounds(self):
            if self._offline_sparse_envelope_min is None or self._offline_sparse_envelope_max is None:
                return None
            return (self._offline_sparse_envelope_min.copy(), self._offline_sparse_envelope_max.copy())

        @property
        def offline_visualization_loaded(self) -> bool:
            return bool(self._offline_visualization_loaded)

        @property
        def offline_session_id(self):
            return self._offline_session_id

        @property
        def offline_scale_status(self):
            return self._offline_scale_status

        @property
        def semantic_objects_available(self) -> bool:
            return bool(self._semantic_objects_available)

        @property
        def semantic_objects_visible(self) -> bool:
            return bool(self._semantic_objects_visible and self._semantic_objects_available)

        @property
        def semantic_object_count(self) -> int:
            return int(self._semantic_object_total_count)

        @property
        def semantic_object_display_count(self) -> int:
            return int(self._semantic_bounds_min.shape[0])

        @property
        def semantic_object_ids(self):
            return self._semantic_object_ids

        @property
        def selected_semantic_object_index(self):
            return self._selected_semantic_object_index

        @property
        def selected_semantic_object_id(self):
            index = self._selected_semantic_object_index
            if index is None or index < 0 or index >= len(self._semantic_object_ids):
                return None
            return self._semantic_object_ids[index]

        @property
        def selected_semantic_object_visible(self) -> bool:
            return bool(self._selected_semantic_object_visible)

        @property
        def world_axis_length(self) -> float:
            return float(self._world_axis_length)

        def orientation_gizmo_axes(self):
            from app.reconstruction_visualization import orientation_gizmo_projection
            return orientation_gizmo_projection(32.0, 24.0)

        @property
        def current_camera_position(self):
            return None if self._current_camera_position is None else self._current_camera_position.copy()

        @property
        def has_pose(self) -> bool:
            return self._has_pose

        @property
        def grid_visible(self) -> bool:
            return self._grid_visible

        @property
        def axes_visible(self) -> bool:
            return self._axes_visible

        @property
        def is_interacting(self) -> bool:
            return self._is_interacting

        def close(self) -> None:
            self.closed = True
            self._native.close()

    fake_module = types.ModuleType(module_name)
    fake_module.PointCloudCanvas = PointCloudCanvas
    fake_module.ReconstructionCanvas = ReconstructionCanvas
    fake_module.DEFAULT_INTERACTION_MODE = "orbit"
    fake_module.DEFAULT_POINT_SIZE = 4.0
    sys.modules[module_name] = fake_module


_install_fake_vispy_canvas()

import pytest


@pytest.fixture(scope="session", autouse=True)
def isolated_qsettings(tmp_path_factory):
    """Redirect QSettings to a temporary INI directory for the test process."""
    try:
        from PyQt6.QtCore import QSettings
    except ModuleNotFoundError:
        yield
        return

    settings_dir = tmp_path_factory.mktemp("qsettings")
    QSettings.setDefaultFormat(QSettings.Format.IniFormat)
    QSettings.setPath(
        QSettings.Format.IniFormat,
        QSettings.Scope.UserScope,
        str(settings_dir),
    )
    settings = QSettings("EIA Tech", "Depth3DModelLab")
    settings.clear()
    settings.sync()
    yield
    settings.clear()
    settings.sync()


def _install_fake_inspection_canvas() -> None:
    """Install a Qt-only inspection 3D canvas before Step 8 UI collection."""
    if os.environ.get("DEPTH3D_TEST_FAKE_VISPY", "1") != "1":
        return
    module_name = "app.widgets.inspection_3d_canvas"
    if module_name in sys.modules:
        return
    try:
        import numpy as np
        from PyQt6.QtCore import QObject, pyqtSignal
        from PyQt6.QtWidgets import QWidget
    except ModuleNotFoundError:
        return

    class Inspection3DCanvas(QObject):
        escape_pressed = pyqtSignal()

        def __init__(self, parent: QObject | None = None) -> None:
            super().__init__(parent)
            self._native = QWidget()
            self.last_object = None
            self.last_arrays = {}
            self.point_mode = None
            self.box_method = None
            self.show_boxes = True
            self.fit_calls = 0
            self.reset_calls = 0
            self.closed = False

        @property
        def native(self):
            return self._native

        def close(self) -> None:
            self.closed = True
            self._native.close()

        def clear(self) -> None:
            self.last_object = None
            self.last_arrays = {}

        def set_object(self, obj, arrays, *, point_mode, box_method, show_boxes) -> None:
            self.last_object = obj
            self.last_arrays = {key: np.asarray(value) for key, value in arrays.items()}
            self.point_mode = point_mode
            self.box_method = box_method
            self.show_boxes = bool(show_boxes)

        def refresh_style(self, *, point_size=None, show_boxes=None) -> None:
            if show_boxes is not None:
                self.show_boxes = bool(show_boxes)

        def fit_scene(self) -> None:
            self.fit_calls += 1

        def reset_view(self) -> None:
            self.reset_calls += 1
            self.fit_scene()

    fake_module = types.ModuleType(module_name)
    fake_module.Inspection3DCanvas = Inspection3DCanvas
    sys.modules[module_name] = fake_module


_install_fake_inspection_canvas()
