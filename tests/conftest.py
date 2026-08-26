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

    valid_modes = {"select", "orbit", "pan", "zoom", "measure"}
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
        measurement_point_picked = pyqtSignal(object)
        measurement_cancel_requested = pyqtSignal()

        MAX_SPARSE_LANDMARK_DISPLAY_POINTS = 8_192
        MAX_KEYFRAME_DISPLAY_FRUSTUMS = 96
        MAX_OFFLINE_CAMERA_PATH_POINTS = 1_024
        MAX_CANONICAL_CAMERA_FRUSTUMS = 192
        MAX_CANONICAL_CAMERA_TRAJECTORY_POINTS = 4_096
        MAX_DENSE_RGB_DISPLAY_POINTS = 250_000
        MAX_SEMANTIC_OBJECT_DISPLAY = 256
        MAX_CANONICAL_SEMANTIC_LABEL_DISPLAY = 64
        CANONICAL_INSPECTION_OVERLAY_FONT_SIZE = 14.0

        def __init__(self, parent: QObject | None = None) -> None:
            super().__init__(parent)
            self._native = _NativeWidget(self)
            self._interaction_mode = "orbit"
            self._grid_visible = True
            self._axes_visible = True
            self._canonical_mesh_layer_visible = True
            self._dense_rgb_points_layer_visible = True
            self._canonical_camera_frustums_layer_visible = True
            self._canonical_camera_trajectory_layer_visible = True
            self._semantic_support_layer_visible = True
            self._canonical_semantic_labels_layer_visible = True
            self._canonical_semantic_selection_layer_visible = True
            self._canonical_inspection_overlay_layer_visible = True
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
            self._canonical_mesh_loaded = False
            self._canonical_mesh_visible = False
            self._canonical_mesh_scene_id = ""
            self._canonical_mesh_artifact_id = ""
            self._canonical_mesh_source_path = ""
            self._canonical_mesh_vertices = np.zeros((0, 3), dtype=np.float32)
            self._canonical_mesh_faces = np.zeros((0, 3), dtype=np.uint32)
            self._canonical_mesh_colors = np.zeros((0, 4), dtype=np.float32)
            self._canonical_mesh_bounds_min = None
            self._canonical_mesh_bounds_max = None
            self._dense_rgb_loaded = False
            self._dense_rgb_visible = False
            self._dense_rgb_scene_id = ""
            self._dense_rgb_representation = ""
            self._dense_rgb_source_path = ""
            self._dense_rgb_points_reconstruction = np.zeros((0, 3), dtype=np.float64)
            self._dense_rgb_colors_rgb = np.zeros((0, 3), dtype=np.uint8)
            self._dense_rgb_total_count = 0
            self._dense_rgb_display_count = 0
            self._dense_rgb_bounds_min = None
            self._dense_rgb_bounds_max = None
            self._interactive_measurement_points = np.zeros((0, 3), dtype=np.float64)
            self._interactive_gaussian_loaded = False
            self._interactive_gaussian_positions = np.zeros((0, 3), dtype=np.float32)
            self._interactive_gaussian_colors = np.zeros((0, 4), dtype=np.float32)
            self._interactive_gaussian_sizes = np.zeros((0,), dtype=np.float32)
            self._interactive_gaussian_total_count = 0
            self._canonical_camera_loaded = False
            self._canonical_camera_scene_id = ""
            self._canonical_camera_artifact_id = ""
            self._canonical_camera_provider_id = ""
            self._canonical_camera_matrices = np.zeros((0, 4, 4), dtype=np.float64)
            self._canonical_camera_positions = np.zeros((0, 3), dtype=np.float32)
            self._canonical_camera_ids = ()
            self._canonical_camera_total_count = 0
            self._canonical_camera_frustum_display_count = 0
            self._canonical_camera_trajectory_display_count = 0
            self._canonical_camera_frustums_visible = False
            self._canonical_camera_trajectory_visible = False
            self._canonical_semantic_loaded = False
            self._canonical_semantic_scene_id = ""
            self._canonical_semantic_artifact_id = ""
            self._canonical_semantic_canonical_artifact_id = ""
            self._canonical_semantic_object_ids = ()
            self._canonical_semantic_labels = ()
            self._canonical_semantic_label_positions = np.zeros((0, 3), dtype=np.float32)
            self._canonical_semantic_surface_vertices = ()
            self._canonical_semantic_surface_faces = ()
            self._canonical_semantic_label_display_count = 0
            self._selected_canonical_semantic_surface_visible = False
            self._selected_canonical_semantic_label_visible = False
            self._canonical_inspection_overlay_visible = False
            self._canonical_inspection_overlay_scene_id = ""
            self._canonical_inspection_overlay_object_id = ""
            self._canonical_inspection_overlay_text = ""
            self._canonical_inspection_overlay_anchor = np.zeros((3,), dtype=np.float32)
            self._canonical_registered_dimension_count = 0
            self._canonical_registered_dimension_scene_id = ""
            self._canonical_registered_dimension_object_id = ""
            self._canonical_registered_dimension_text = ()
            self._canonical_registered_dimensions_visible = False
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
            self._refresh_canonical_camera_layer_visibility()

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

        def set_dense_rgb_points(
            self, points_reconstruction, colors_rgb, *, scene_id: str, representation: str, source_path: str
        ) -> None:
            points = np.asarray(points_reconstruction, dtype=np.float64)
            colors = np.asarray(colors_rgb, dtype=np.uint8)
            if points.ndim != 2 or points.shape[1:] != (3,) or points.shape[0] < 1:
                raise ValueError("Dense RGB points must be non-empty (N, 3).")
            if colors.shape != points.shape:
                raise ValueError("Dense RGB colors must align with points.")
            rep = str(representation).strip().lower()
            if rep not in {"extended", "core"}:
                raise ValueError("Dense RGB representation must be extended/core.")
            self._dense_rgb_points_reconstruction = points.copy()
            self._dense_rgb_colors_rgb = colors.copy()
            self._dense_rgb_scene_id = str(scene_id)
            self._dense_rgb_representation = rep
            self._dense_rgb_source_path = str(source_path)
            self._dense_rgb_total_count = int(points.shape[0])
            self._dense_rgb_display_count = min(int(points.shape[0]), self.MAX_DENSE_RGB_DISPLAY_POINTS)
            self._dense_rgb_bounds_min = points.min(axis=0).copy()
            self._dense_rgb_bounds_max = points.max(axis=0).copy()
            self._dense_rgb_loaded = True
            self._dense_rgb_visible = bool(self._dense_rgb_points_layer_visible)

        def clear_dense_rgb_points(self, *, update_canvas: bool = True) -> None:
            self._dense_rgb_loaded = False
            self._dense_rgb_visible = False
            self._dense_rgb_scene_id = ""
            self._dense_rgb_representation = ""
            self._dense_rgb_source_path = ""
            self._dense_rgb_points_reconstruction = np.zeros((0, 3), dtype=np.float64)
            self._dense_rgb_colors_rgb = np.zeros((0, 3), dtype=np.uint8)
            self._dense_rgb_total_count = 0
            self._dense_rgb_display_count = 0
            self._dense_rgb_bounds_min = None
            self._dense_rgb_bounds_max = None

        def set_dense_rgb_points_visible(self, visible: bool) -> None:
            self._dense_rgb_points_layer_visible = bool(visible)
            self._dense_rgb_visible = bool(self._dense_rgb_loaded and self._dense_rgb_points_layer_visible)

        def set_canonical_mesh(
            self,
            *,
            vertices_viewer,
            faces,
            vertex_colors_rgba,
            scene_id: str,
            artifact_id: str,
            source_path: str,
        ) -> None:
            vertices = np.asarray(vertices_viewer, dtype=np.float32)
            triangles = np.asarray(faces, dtype=np.uint32)
            colors = np.asarray(vertex_colors_rgba, dtype=np.float32)
            if vertices.ndim != 2 or vertices.shape[1:] != (3,) or vertices.shape[0] < 3:
                raise ValueError("vertices_viewer must contain at least three (N, 3) vertices.")
            if triangles.ndim != 2 or triangles.shape[1:] != (3,) or triangles.shape[0] < 1:
                raise ValueError("faces must contain at least one triangular face.")
            if colors.shape != (vertices.shape[0], 4):
                raise ValueError("vertex_colors_rgba must have shape (N, 4).")
            self._canonical_mesh_vertices = vertices.copy()
            self._canonical_mesh_faces = triangles.copy()
            self._canonical_mesh_colors = colors.copy()
            self._canonical_mesh_bounds_min = vertices.min(axis=0).copy()
            self._canonical_mesh_bounds_max = vertices.max(axis=0).copy()
            self._canonical_mesh_scene_id = str(scene_id)
            self._canonical_mesh_artifact_id = str(artifact_id)
            self._canonical_mesh_source_path = str(source_path)
            self._canonical_mesh_loaded = True
            self._canonical_mesh_visible = bool(self._canonical_mesh_layer_visible)

        def clear_canonical_mesh(self, *, update_canvas: bool = True) -> None:
            self.clear_canonical_semantic_objects(update_canvas=False)
            self._canonical_mesh_loaded = False
            self._canonical_mesh_visible = False
            self._canonical_mesh_scene_id = ""
            self._canonical_mesh_artifact_id = ""
            self._canonical_mesh_source_path = ""
            self._canonical_mesh_vertices = np.zeros((0, 3), dtype=np.float32)
            self._canonical_mesh_faces = np.zeros((0, 3), dtype=np.uint32)
            self._canonical_mesh_colors = np.zeros((0, 4), dtype=np.float32)
            self._canonical_mesh_bounds_min = None
            self._canonical_mesh_bounds_max = None
            self._interactive_measurement_points = np.zeros((0, 3), dtype=np.float64)
            self.clear_interactive_gaussian_scene(update_canvas=False)

        def set_interactive_measurement_overlay(self, points_reconstruction, *, update_canvas: bool = True) -> None:
            points = np.asarray(points_reconstruction, dtype=np.float64)
            if points.ndim != 2 or points.shape[1:] != (3,) or points.shape[0] not in {1, 2}:
                raise ValueError("Interactive measurement overlay requires one or two 3D points.")
            self._interactive_measurement_points = points.copy()

        def clear_interactive_measurement_overlay(self, *, update_canvas: bool = True) -> None:
            self._interactive_measurement_points = np.zeros((0, 3), dtype=np.float64)

        def set_interactive_gaussian_scene(
            self, positions_reconstruction, colors_rgba, sizes_px, *, scene_id: str, artifact_id: str, total_count=None
        ) -> None:
            positions = np.asarray(positions_reconstruction, dtype=np.float32)
            colors = np.asarray(colors_rgba, dtype=np.float32)
            sizes = np.asarray(sizes_px, dtype=np.float32).reshape(-1)
            if positions.ndim != 2 or positions.shape[1] != 3 or positions.shape[0] < 1:
                raise ValueError("positions_reconstruction must be non-empty (N, 3).")
            if colors.shape != (positions.shape[0], 4) or sizes.shape != (positions.shape[0],):
                raise ValueError("Gaussian color/size buffers must match position count.")
            self._interactive_gaussian_positions = positions.copy()
            self._interactive_gaussian_colors = colors.copy()
            self._interactive_gaussian_sizes = sizes.copy()
            self._interactive_gaussian_total_count = max(
                int(total_count) if total_count is not None else int(positions.shape[0]),
                int(positions.shape[0]),
            )
            self._interactive_gaussian_loaded = True

        def clear_interactive_gaussian_scene(self, *, update_canvas: bool = True) -> None:
            self._interactive_gaussian_loaded = False
            self._interactive_gaussian_positions = np.zeros((0, 3), dtype=np.float32)
            self._interactive_gaussian_colors = np.zeros((0, 4), dtype=np.float32)
            self._interactive_gaussian_sizes = np.zeros((0,), dtype=np.float32)
            self._interactive_gaussian_total_count = 0

        def set_canonical_mesh_visible(self, visible: bool) -> None:
            self._canonical_mesh_layer_visible = bool(visible)
            self._canonical_mesh_visible = bool(
                self._canonical_mesh_loaded and self._canonical_mesh_layer_visible
            )

        def set_canonical_camera_frustums_visible(self, visible: bool) -> None:
            self._canonical_camera_frustums_layer_visible = bool(visible)
            self._refresh_canonical_camera_layer_visibility()

        def set_canonical_camera_trajectory_visible(self, visible: bool) -> None:
            self._canonical_camera_trajectory_layer_visible = bool(visible)
            self._refresh_canonical_camera_layer_visibility()

        def _refresh_canonical_camera_layer_visibility(self) -> None:
            self._canonical_camera_frustums_visible = bool(
                self._canonical_camera_loaded
                and self._keyframes_visible
                and self._canonical_camera_frustums_layer_visible
            )
            self._canonical_camera_trajectory_visible = bool(
                self._canonical_camera_loaded
                and self._trajectory_visible
                and self._canonical_camera_trajectory_layer_visible
                and self._canonical_camera_trajectory_display_count >= 2
            )

        def set_canonical_camera_trajectory(
            self,
            *,
            matrices_reconstruction,
            camera_ids,
            scene_id: str,
            artifact_id: str,
            provider_id: str,
            total_count: int | None = None,
        ) -> None:
            from app.reconstruction_visualization import (
                keyframe_centers_in_viewer,
                keyframe_display_indices,
                trajectory_display_indices,
            )

            matrices = np.asarray(matrices_reconstruction, dtype=np.float64)
            ids = tuple(str(value).strip() for value in camera_ids)
            if matrices.ndim != 3 or matrices.shape[1:] != (4, 4) or matrices.shape[0] < 1:
                raise ValueError("matrices_reconstruction must have shape (N, 4, 4) with N >= 1.")
            if len(ids) != matrices.shape[0] or any(not value for value in ids):
                raise ValueError("camera_ids must match canonical camera poses.")
            positions = keyframe_centers_in_viewer(matrices)
            frustum_indices = keyframe_display_indices(
                matrices.shape[0], max_keyframes=self.MAX_CANONICAL_CAMERA_FRUSTUMS
            )
            trajectory_indices = trajectory_display_indices(
                positions.shape[0], max_points=self.MAX_CANONICAL_CAMERA_TRAJECTORY_POINTS
            )
            self._canonical_camera_loaded = True
            self._canonical_camera_scene_id = str(scene_id)
            self._canonical_camera_artifact_id = str(artifact_id)
            self._canonical_camera_provider_id = str(provider_id)
            self._canonical_camera_matrices = matrices.copy()
            self._canonical_camera_positions = positions.copy()
            self._canonical_camera_ids = ids
            self._canonical_camera_total_count = (
                matrices.shape[0] if total_count is None else int(total_count)
            )
            self._canonical_camera_frustum_display_count = int(frustum_indices.shape[0])
            self._canonical_camera_trajectory_display_count = int(trajectory_indices.shape[0])
            self._refresh_canonical_camera_layer_visibility()

        def clear_canonical_camera_trajectory(self, *, update_canvas: bool = True) -> None:
            self._canonical_camera_loaded = False
            self._canonical_camera_scene_id = ""
            self._canonical_camera_artifact_id = ""
            self._canonical_camera_provider_id = ""
            self._canonical_camera_matrices = np.zeros((0, 4, 4), dtype=np.float64)
            self._canonical_camera_positions = np.zeros((0, 3), dtype=np.float32)
            self._canonical_camera_ids = ()
            self._canonical_camera_total_count = 0
            self._canonical_camera_frustum_display_count = 0
            self._canonical_camera_trajectory_display_count = 0
            self._canonical_camera_frustums_visible = False
            self._canonical_camera_trajectory_visible = False

        def set_canonical_semantic_objects(
            self,
            *,
            object_ids,
            labels,
            label_positions_viewer,
            surface_vertices_viewer,
            surface_faces,
            scene_id: str,
            semantic_artifact_id: str,
            canonical_artifact_id: str,
            preserve_selected_mesh_visual: bool = False,
        ) -> None:
            ids = tuple(str(value).strip() for value in object_ids)
            labels = tuple(str(value).strip() for value in labels)
            positions = np.asarray(label_positions_viewer, dtype=np.float32)
            vertices = tuple(np.asarray(value, dtype=np.float32) for value in surface_vertices_viewer)
            faces = tuple(np.asarray(value, dtype=np.uint32) for value in surface_faces)
            count = len(ids)
            if not self._canonical_mesh_loaded:
                raise ValueError("Canonical semantic binding requires a loaded canonical mesh.")
            if str(scene_id) != self._canonical_mesh_scene_id:
                raise ValueError("Canonical semantic scene identity does not match the loaded mesh.")
            if str(canonical_artifact_id) != self._canonical_mesh_artifact_id:
                raise ValueError("Canonical semantic binding targets a different canonical artifact.")
            if self._semantic_objects_available and ids != self._semantic_object_ids:
                raise ValueError("Canonical semantic object ordering must match the active semantic presentation.")
            if len(labels) != count or positions.shape != (count, 3):
                raise ValueError("Canonical semantic labels and positions must match object_ids.")
            if len(vertices) != count or len(faces) != count:
                raise ValueError("Canonical semantic surface arrays must match object_ids.")
            self.clear_canonical_semantic_objects(
                update_canvas=False,
                preserve_selected_mesh_visual=preserve_selected_mesh_visual,
            )
            self._canonical_semantic_loaded = bool(count)
            self._canonical_semantic_scene_id = str(scene_id)
            self._canonical_semantic_artifact_id = str(semantic_artifact_id)
            self._canonical_semantic_canonical_artifact_id = str(canonical_artifact_id)
            self._canonical_semantic_object_ids = ids
            self._canonical_semantic_labels = labels
            self._canonical_semantic_label_positions = positions.copy()
            self._canonical_semantic_surface_vertices = tuple(value.copy() for value in vertices)
            self._canonical_semantic_surface_faces = tuple(value.copy() for value in faces)
            self._canonical_semantic_label_display_count = min(
                count, self.MAX_CANONICAL_SEMANTIC_LABEL_DISPLAY
            )
            if self._selected_semantic_object_index is not None:
                self._refresh_selected_canonical_semantic_state()

        def clear_canonical_semantic_objects(
            self,
            *,
            update_canvas: bool = True,
            preserve_selected_mesh_visual: bool = False,
        ) -> None:
            self.clear_canonical_inspection_overlay(update_canvas=False)
            self.clear_canonical_registered_dimensions(update_canvas=False)
            self._canonical_semantic_loaded = False
            self._canonical_semantic_scene_id = ""
            self._canonical_semantic_artifact_id = ""
            self._canonical_semantic_canonical_artifact_id = ""
            self._canonical_semantic_object_ids = ()
            self._canonical_semantic_labels = ()
            self._canonical_semantic_label_positions = np.zeros((0, 3), dtype=np.float32)
            self._canonical_semantic_surface_vertices = ()
            self._canonical_semantic_surface_faces = ()
            self._canonical_semantic_label_display_count = 0
            self._selected_canonical_semantic_surface_visible = False
            self._selected_canonical_semantic_label_visible = False

        def set_canonical_inspection_overlay(
            self, *, scene_id: str, object_id: str, anchor_viewer, text: str
        ) -> None:
            if not self._canonical_semantic_loaded:
                raise ValueError("Canonical inspection overlay requires loaded canonical semantics.")
            scene_id = str(scene_id).strip()
            object_id = str(object_id).strip()
            text = str(text).strip()
            if scene_id != self._canonical_semantic_scene_id:
                raise ValueError("Canonical inspection overlay scene does not match loaded semantics.")
            if object_id not in self._canonical_semantic_object_ids:
                raise ValueError("Canonical inspection overlay targets an unknown semantic object.")
            if self.selected_semantic_object_id != object_id:
                raise ValueError("Canonical inspection overlay must target the currently selected object.")
            anchor = np.asarray(anchor_viewer, dtype=np.float32)
            if anchor.shape != (3,) or not np.isfinite(anchor).all():
                raise ValueError("Canonical inspection overlay anchor must be one finite 3D point.")
            if not text:
                raise ValueError("Canonical inspection overlay text must be non-empty.")
            self._canonical_inspection_overlay_scene_id = scene_id
            self._canonical_inspection_overlay_object_id = object_id
            self._canonical_inspection_overlay_text = text
            self._canonical_inspection_overlay_anchor = anchor.copy()
            self._refresh_canonical_inspection_overlay_visibility()

        def clear_canonical_inspection_overlay(self, *, update_canvas: bool = True) -> None:
            self._canonical_inspection_overlay_visible = False
            self._canonical_inspection_overlay_scene_id = ""
            self._canonical_inspection_overlay_object_id = ""
            self._canonical_inspection_overlay_text = ""
            self._canonical_inspection_overlay_anchor = np.zeros((3,), dtype=np.float32)

        def _refresh_canonical_inspection_overlay_visibility(self) -> None:
            self._canonical_inspection_overlay_visible = bool(
                self._canonical_inspection_overlay_layer_visible
                and self._semantic_objects_visible
                and self._semantic_objects_available
                and self._canonical_semantic_loaded
                and self._canonical_inspection_overlay_text
                and self.selected_semantic_object_id == self._canonical_inspection_overlay_object_id
                and self._canonical_inspection_overlay_scene_id == self._canonical_semantic_scene_id
            )

        def set_canonical_registered_dimensions(
            self, *, scene_id, object_id, starts_reconstruction, ends_reconstruction, labels
        ) -> None:
            if not self._canonical_semantic_loaded:
                raise ValueError("Canonical registered dimensions require loaded canonical semantics.")
            starts = np.asarray(starts_reconstruction, dtype=np.float64)
            ends = np.asarray(ends_reconstruction, dtype=np.float64)
            texts = tuple(str(value).strip() for value in labels)
            if starts.ndim != 2 or starts.shape[1:] != (3,) or ends.shape != starts.shape:
                raise ValueError("Registered dimension endpoints must have shape (N, 3).")
            if not 1 <= len(starts) <= 3 or len(texts) != len(starts):
                raise ValueError("Registered dimensions require 1-3 labels.")
            if str(scene_id) != self._canonical_semantic_scene_id:
                raise ValueError("Registered dimension scene mismatch.")
            if str(object_id) != self.selected_semantic_object_id:
                raise ValueError("Registered dimensions require selected object.")
            self._canonical_registered_dimension_count = len(texts)
            self._canonical_registered_dimension_scene_id = str(scene_id)
            self._canonical_registered_dimension_object_id = str(object_id)
            self._canonical_registered_dimension_text = texts
            self._refresh_canonical_registered_dimension_visibility()

        def clear_canonical_registered_dimensions(self, *, update_canvas: bool = True) -> None:
            self._canonical_registered_dimension_count = 0
            self._canonical_registered_dimension_scene_id = ""
            self._canonical_registered_dimension_object_id = ""
            self._canonical_registered_dimension_text = ()
            self._canonical_registered_dimensions_visible = False

        def _refresh_canonical_registered_dimension_visibility(self) -> None:
            self._canonical_registered_dimensions_visible = bool(
                self._canonical_inspection_overlay_layer_visible
                and self._semantic_objects_visible
                and self._semantic_objects_available
                and self._canonical_semantic_loaded
                and self._canonical_registered_dimension_count
                and self.selected_semantic_object_id == self._canonical_registered_dimension_object_id
                and self._canonical_registered_dimension_scene_id == self._canonical_semantic_scene_id
            )

        def _refresh_selected_canonical_semantic_state(self) -> None:
            index = self._selected_semantic_object_index
            active = bool(
                self._semantic_objects_visible
                and self._semantic_objects_available
                and self._canonical_semantic_loaded
                and index is not None
                and 0 <= index < len(self._canonical_semantic_object_ids)
                and index < len(self._semantic_object_ids)
                and self._canonical_semantic_object_ids[index] == self._semantic_object_ids[index]
            )
            self._selected_canonical_semantic_label_visible = bool(
                active and self._canonical_semantic_labels_layer_visible
            )
            self._selected_canonical_semantic_surface_visible = bool(
                active
                and self._canonical_semantic_selection_layer_visible
                and self._canonical_semantic_surface_faces[index].shape[0]
            ) if active else False

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
            self.clear_canonical_semantic_objects(update_canvas=False)
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
            self._refresh_semantic_layer_state()

        def set_semantic_support_visible(self, visible: bool) -> None:
            self._semantic_support_layer_visible = bool(visible)
            self._refresh_semantic_layer_state()

        def set_canonical_semantic_labels_visible(self, visible: bool) -> None:
            self._canonical_semantic_labels_layer_visible = bool(visible)
            self._refresh_semantic_layer_state()

        def set_canonical_semantic_selection_visible(self, visible: bool) -> None:
            self._canonical_semantic_selection_layer_visible = bool(visible)
            self._refresh_semantic_layer_state()

        def set_canonical_inspection_overlay_visible(self, visible: bool) -> None:
            self._canonical_inspection_overlay_layer_visible = bool(visible)
            self._refresh_canonical_inspection_overlay_visibility()
            self._refresh_canonical_registered_dimension_visibility()

        def _refresh_semantic_layer_state(self) -> None:
            self._selected_semantic_object_visible = bool(
                self._semantic_objects_visible
                and self._semantic_support_layer_visible
                and self._semantic_objects_available
                and self._selected_semantic_object_index is not None
            )
            self._refresh_selected_canonical_semantic_state()
            self._refresh_canonical_inspection_overlay_visibility()
            self._refresh_canonical_registered_dimension_visibility()

        def select_semantic_object(self, object_index: int) -> None:
            index = int(object_index)
            if not self._semantic_objects_available:
                raise ValueError("No semantic object presentation is loaded.")
            if index < 0 or index >= self._semantic_bounds_min.shape[0]:
                raise IndexError("Semantic object index is out of range.")
            self._selected_semantic_object_index = index
            self._selected_semantic_object_visible = bool(
                self._semantic_objects_visible and self._semantic_support_layer_visible
            )
            self._refresh_selected_canonical_semantic_state()
            self._refresh_canonical_inspection_overlay_visibility()
            self._refresh_canonical_registered_dimension_visibility()

        def clear_semantic_object_selection(self) -> None:
            self._selected_semantic_object_index = None
            self._selected_semantic_object_visible = False
            self._selected_canonical_semantic_surface_visible = False
            self._selected_canonical_semantic_label_visible = False
            self.clear_canonical_inspection_overlay(update_canvas=False)

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
            self._refresh_canonical_camera_layer_visibility()

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
        def canonical_mesh_layer_visible(self) -> bool:
            return bool(self._canonical_mesh_layer_visible)

        @property
        def canonical_camera_frustums_layer_visible(self) -> bool:
            return bool(self._canonical_camera_frustums_layer_visible)

        @property
        def canonical_camera_trajectory_layer_visible(self) -> bool:
            return bool(self._canonical_camera_trajectory_layer_visible)

        @property
        def semantic_support_layer_visible(self) -> bool:
            return bool(self._semantic_support_layer_visible)

        @property
        def canonical_semantic_labels_layer_visible(self) -> bool:
            return bool(self._canonical_semantic_labels_layer_visible)

        @property
        def canonical_semantic_selection_layer_visible(self) -> bool:
            return bool(self._canonical_semantic_selection_layer_visible)

        @property
        def canonical_inspection_overlay_layer_visible(self) -> bool:
            return bool(self._canonical_inspection_overlay_layer_visible)

        @property
        def dense_rgb_points_loaded(self) -> bool:
            return bool(self._dense_rgb_loaded)

        @property
        def dense_rgb_points_visible(self) -> bool:
            return bool(self._dense_rgb_loaded and self._dense_rgb_visible)

        @property
        def dense_rgb_points_layer_visible(self) -> bool:
            return bool(self._dense_rgb_points_layer_visible)

        @property
        def dense_rgb_scene_id(self) -> str:
            return self._dense_rgb_scene_id

        @property
        def dense_rgb_representation(self) -> str:
            return self._dense_rgb_representation

        @property
        def dense_rgb_source_path(self) -> str:
            return self._dense_rgb_source_path

        @property
        def dense_rgb_total_count(self) -> int:
            return int(self._dense_rgb_total_count)

        @property
        def dense_rgb_display_count(self) -> int:
            return int(self._dense_rgb_display_count)

        @property
        def dense_rgb_bounds(self):
            if self._dense_rgb_bounds_min is None or self._dense_rgb_bounds_max is None:
                return None
            return self._dense_rgb_bounds_min.copy(), self._dense_rgb_bounds_max.copy()

        @property
        def canonical_mesh_loaded(self) -> bool:
            return bool(self._canonical_mesh_loaded)

        @property
        def interactive_gaussian_visible(self) -> bool:
            return bool(self._interactive_gaussian_loaded)

        @property
        def interactive_gaussian_display_count(self) -> int:
            return int(self._interactive_gaussian_positions.shape[0])

        @property
        def interactive_gaussian_total_count(self) -> int:
            return int(self._interactive_gaussian_total_count)

        @property
        def canonical_mesh_visible(self) -> bool:
            return bool(self._canonical_mesh_visible and self._canonical_mesh_loaded)

        @property
        def canonical_mesh_vertex_count(self) -> int:
            return int(self._canonical_mesh_vertices.shape[0])

        @property
        def canonical_mesh_face_count(self) -> int:
            return int(self._canonical_mesh_faces.shape[0])

        @property
        def canonical_mesh_scene_id(self) -> str:
            return self._canonical_mesh_scene_id

        @property
        def canonical_mesh_artifact_id(self) -> str:
            return self._canonical_mesh_artifact_id

        @property
        def canonical_mesh_source_path(self) -> str:
            return self._canonical_mesh_source_path

        @property
        def canonical_mesh_bounds(self):
            if self._canonical_mesh_bounds_min is None or self._canonical_mesh_bounds_max is None:
                return None
            return self._canonical_mesh_bounds_min.copy(), self._canonical_mesh_bounds_max.copy()

        @property
        def canonical_camera_loaded(self) -> bool:
            return self._canonical_camera_loaded

        @property
        def canonical_camera_scene_id(self) -> str:
            return self._canonical_camera_scene_id

        @property
        def canonical_camera_artifact_id(self) -> str:
            return self._canonical_camera_artifact_id

        @property
        def canonical_camera_provider_id(self) -> str:
            return self._canonical_camera_provider_id

        @property
        def canonical_camera_ids(self) -> tuple[str, ...]:
            return self._canonical_camera_ids

        @property
        def canonical_camera_count(self) -> int:
            return int(self._canonical_camera_total_count)

        @property
        def canonical_camera_persisted_count(self) -> int:
            return int(self._canonical_camera_matrices.shape[0])

        @property
        def canonical_camera_frustum_display_count(self) -> int:
            return int(self._canonical_camera_frustum_display_count)

        @property
        def canonical_camera_trajectory_display_count(self) -> int:
            return int(self._canonical_camera_trajectory_display_count)

        @property
        def canonical_camera_frustums_visible(self) -> bool:
            return bool(self._canonical_camera_loaded and self._canonical_camera_frustums_visible)

        @property
        def canonical_camera_trajectory_visible(self) -> bool:
            return bool(self._canonical_camera_loaded and self._canonical_camera_trajectory_visible)

        @property
        def canonical_semantic_loaded(self) -> bool:
            return bool(self._canonical_semantic_loaded)

        @property
        def canonical_semantic_scene_id(self) -> str:
            return self._canonical_semantic_scene_id

        @property
        def canonical_semantic_artifact_id(self) -> str:
            return self._canonical_semantic_artifact_id

        @property
        def canonical_semantic_object_ids(self):
            return self._canonical_semantic_object_ids

        @property
        def canonical_semantic_surface_binding_count(self) -> int:
            return sum(bool(value.shape[0]) for value in self._canonical_semantic_surface_faces)

        @property
        def canonical_semantic_label_display_count(self) -> int:
            return int(self._canonical_semantic_label_display_count)

        @property
        def canonical_semantic_labels_visible(self) -> bool:
            return bool(
                self._canonical_semantic_loaded
                and self._canonical_semantic_labels_layer_visible
                and self._semantic_objects_visible
                and self._semantic_objects_available
                and self._canonical_semantic_label_display_count
            )

        @property
        def selected_canonical_semantic_surface_visible(self) -> bool:
            return bool(self._selected_canonical_semantic_surface_visible)

        @property
        def selected_canonical_semantic_label_visible(self) -> bool:
            return bool(self._selected_canonical_semantic_label_visible)

        @property
        def canonical_inspection_overlay_visible(self) -> bool:
            return bool(self._canonical_inspection_overlay_visible)

        @property
        def canonical_inspection_overlay_text(self) -> str:
            return self._canonical_inspection_overlay_text

        @property
        def canonical_inspection_overlay_object_id(self) -> str:
            return self._canonical_inspection_overlay_object_id

        @property
        def canonical_registered_dimension_count(self) -> int:
            return int(self._canonical_registered_dimension_count)

        @property
        def canonical_registered_dimensions_visible(self) -> bool:
            return bool(self._canonical_registered_dimensions_visible)

        @property
        def canonical_registered_dimension_object_id(self) -> str:
            return self._canonical_registered_dimension_object_id

        @property
        def canonical_registered_dimension_text(self) -> tuple[str, ...]:
            return self._canonical_registered_dimension_text

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
            self.clear_canonical_camera_trajectory(update_canvas=False)
            self.clear_canonical_semantic_objects(update_canvas=False)
            self.clear_dense_rgb_points(update_canvas=False)
            self.clear_canonical_mesh(update_canvas=False)
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
