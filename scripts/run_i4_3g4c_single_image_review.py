#!/usr/bin/env python3
"""Prepare, reveal, and formally close I4.3G4C single-image review evidence.

Prepare is evidence-only: it consumes frozen G4B artifacts, creates blinded
A-F native-depth and canonical-geometry review panels, and writes an editable
manual-review template. Reveal is blocked until every frozen G1 review
dimension is explicitly confirmed. Close requires all three scenario reveals.

No command reruns PyCOLMAP or a depth provider.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time
from typing import Any, Mapping

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.geometry.sampling import enforce_point_budget  # noqa: E402
from core.reconstruction.errors import ReconstructionValidationError  # noqa: E402
from core.reconstruction.mvs_da3_matched_view import (  # noqa: E402
    MATCHED_VIEW_FRAMINGS,
    MATCHED_VIEW_VIEWS,
    derive_shared_visual_frame,
)
from core.reconstruction.single_image_provider_matched_view_review import (  # noqa: E402
    I43G4C_BLIND_LABELS,
    I43G4C_PHASE,
    I43G4C_PROVIDER_IMPLEMENTATIONS,
    I43G4C_REVIEW_DIMENSIONS,
    I43G4C_SCENARIOS,
    assignment_commitment,
    blind_assignment,
    build_controlled_reveal,
    build_i4_3g4c_final_closure,
    build_manual_review_template,
    load_g4b_matched_view_sources,
    read_json_object,
    review_is_complete,
    sha256_file,
    verify_g1_single_image_review_requirement,
    verify_sha256,
)

_CANONICAL_TILE_W = 640
_CANONICAL_TILE_H = 480
_NATIVE_TILE_W = 480
_NATIVE_TILE_H = 360


def _write_json(path: Path, payload: Mapping[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    return path


def _artifact(path: Path) -> dict[str, Any]:
    source = path.expanduser().resolve()
    return {"path": str(source), "sha256": sha256_file(source)}


def _load_native_depth(path: Path) -> tuple[np.ndarray, np.ndarray]:
    try:
        with np.load(path, allow_pickle=False) as archive:
            if "depth" in archive.files:
                depth = np.asarray(archive["depth"], dtype=np.float32)
            elif "relative_depth" in archive.files:
                depth = np.asarray(archive["relative_depth"], dtype=np.float32)
            else:
                raise ReconstructionValidationError(f"I4.3G4C native depth NPZ lacks depth/relative_depth: {path}")
            if "valid_mask" in archive.files:
                valid = np.asarray(archive["valid_mask"], dtype=bool)
            else:
                valid = np.isfinite(depth)
    except (OSError, ValueError) as exc:
        raise ReconstructionValidationError(f"I4.3G4C could not load native depth evidence {path}: {exc}") from exc
    if depth.ndim != 2 or valid.shape != depth.shape:
        raise ReconstructionValidationError(f"I4.3G4C native depth grid is invalid: {path}")
    if not np.isfinite(depth[valid]).all():
        raise ReconstructionValidationError(f"I4.3G4C native depth contains non-finite valid pixels: {path}")
    return depth, valid


def _provider_display_range(items: list[tuple[np.ndarray, np.ndarray]]) -> tuple[float, float]:
    samples: list[np.ndarray] = []
    for depth, valid in items:
        sampled = depth[::8, ::8]
        sampled_valid = valid[::8, ::8] & np.isfinite(sampled)
        values = sampled[sampled_valid]
        if values.size:
            samples.append(values.astype(np.float64, copy=False))
    if not samples:
        raise ReconstructionValidationError("I4.3G4C provider native-depth display range has no valid samples.")
    merged = np.concatenate(samples)
    low, high = np.quantile(merged, [0.02, 0.98])
    low_value = float(low)
    high_value = float(high)
    if not np.isfinite(low_value) or not np.isfinite(high_value) or high_value <= low_value:
        low_value = float(np.min(merged)); high_value = float(np.max(merged))
    if high_value <= low_value:
        raise ReconstructionValidationError("I4.3G4C provider native-depth display range is degenerate.")
    return low_value, high_value


def _resize_with_letterbox(image: np.ndarray, width: int, height: int) -> np.ndarray:
    h, w = image.shape[:2]
    scale = min(width / float(w), height / float(h))
    tw, th = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    resized = cv2.resize(image, (tw, th), interpolation=cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR)
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    x = (width - tw) // 2; y = (height - th) // 2
    canvas[y:y + th, x:x + tw] = resized
    return canvas


def _label_tile(image: np.ndarray, title: str, subtitle: str = "") -> np.ndarray:
    canvas = image.copy()
    cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 52), (18, 18, 18), thickness=-1)
    cv2.putText(canvas, title, (12, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (245, 245, 245), 1, cv2.LINE_AA)
    if subtitle:
        cv2.putText(canvas, subtitle, (12, 43), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (205, 205, 205), 1, cv2.LINE_AA)
    return canvas


def _depth_tile(depth: np.ndarray, valid: np.ndarray, *, low: float, high: float, label: str) -> np.ndarray:
    normalized = np.clip((depth.astype(np.float64) - low) / (high - low), 0.0, 1.0)
    gray = np.asarray(np.round(normalized * 255.0), dtype=np.uint8)
    colored = cv2.applyColorMap(gray, cv2.COLORMAP_TURBO)
    colored[~valid] = 0
    tile = _resize_with_letterbox(colored, _NATIVE_TILE_W, _NATIVE_TILE_H)
    return _label_tile(tile, f"BLIND {label}", "provider-native depth | fixed scenario range")


def _rgb_tile(path: Path, *, keyframe_id: str) -> np.ndarray:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ReconstructionValidationError(f"I4.3G4C could not decode authoritative RGB: {path}")
    tile = _resize_with_letterbox(bgr, _NATIVE_TILE_W, _NATIVE_TILE_H)
    return _label_tile(tile, "SOURCE RGB", f"authoritative SfM keyframe {keyframe_id}")


def _policy_tile() -> np.ndarray:
    tile = np.zeros((_NATIVE_TILE_H, _NATIVE_TILE_W, 3), dtype=np.uint8)
    lines = [
        "I4.3G4C BLINDED NATIVE DEPTH",
        "Same frozen keyframe for A-F",
        "One fixed display range/provider/scenario",
        "No per-frame auto-fit",
        "Color values are NOT cross-provider metric scale",
        "Production status remains primary evidence",
    ]
    for index, text in enumerate(lines):
        cv2.putText(tile, text, (20, 48 + index * 44), cv2.FONT_HERSHEY_SIMPLEX, 0.53 if index == 0 else 0.45, (235, 235, 235), 1, cv2.LINE_AA)
    return tile


def _grid(tiles: list[np.ndarray], *, columns: int, output: Path) -> None:
    if not tiles:
        raise ReconstructionValidationError("I4.3G4C cannot create an empty review grid.")
    h, w = tiles[0].shape[:2]
    if any(tile.shape[:2] != (h, w) for tile in tiles):
        raise ReconstructionValidationError("I4.3G4C review grid tiles do not share one raster extent.")
    rows = (len(tiles) + columns - 1) // columns
    blank = np.zeros_like(tiles[0])
    padded = tiles + [blank] * (rows * columns - len(tiles))
    stacked = np.vstack([np.hstack(padded[row * columns:(row + 1) * columns]) for row in range(rows)])
    output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output), stacked):
        raise ReconstructionValidationError(f"I4.3G4C could not write review grid: {output}")


def _workspace_rgb_map(sfm: Mapping[str, Any], *, common_keyframes: tuple[str, ...]) -> dict[str, Path]:
    workspace = sfm.get("workspace")
    if not isinstance(workspace, Mapping):
        raise ReconstructionValidationError("I4.3G4C authoritative SfM workspace is missing.")
    root = Path(str(workspace.get("workspace_root", ""))).expanduser().resolve()
    images = workspace.get("images")
    if not isinstance(images, list):
        raise ReconstructionValidationError("I4.3G4C authoritative SfM workspace images are missing.")
    by_id: dict[str, Path] = {}
    for raw in images:
        if not isinstance(raw, Mapping):
            continue
        keyframe_id = str(raw.get("keyframe_id", ""))
        if keyframe_id not in common_keyframes:
            continue
        path = (root / str(raw.get("staged_image_path", ""))).resolve()
        if not path.is_file():
            raise ReconstructionValidationError(f"I4.3G4C authoritative RGB is missing: {path}")
        expected = str(raw.get("staged_sha256", ""))
        verify_sha256(path, expected, label=f"authoritative RGB {keyframe_id}")
        by_id[keyframe_id] = path
    if tuple(key for key in common_keyframes if key in by_id) != common_keyframes:
        raise ReconstructionValidationError("I4.3G4C authoritative RGB coverage is incomplete for common matched views.")
    return by_id


def _load_canonical_cloud(path: Path) -> tuple[np.ndarray, np.ndarray]:
    try:
        with np.load(path, allow_pickle=False) as archive:
            points = np.asarray(archive["points_reconstruction"], dtype=np.float32)
            colors = np.asarray(archive["colors_rgb"], dtype=np.uint8)
    except (OSError, KeyError, ValueError) as exc:
        raise ReconstructionValidationError(f"I4.3G4C could not load canonical geometry {path}: {exc}") from exc
    if points.ndim != 2 or points.shape[1] != 3 or colors.shape != points.shape:
        raise ReconstructionValidationError(f"I4.3G4C canonical geometry shape is invalid: {path}")
    if len(points) < 3 or not np.isfinite(points).all():
        raise ReconstructionValidationError(f"I4.3G4C canonical geometry contains insufficient/non-finite points: {path}")
    return points, colors


def _render_cloud_tile(
    *,
    app: Any,
    points: np.ndarray,
    colors: np.ndarray,
    blind_label: str,
    view: str,
    framing: str,
    output: Path,
    center_viewer: tuple[float, float, float],
    distance: float,
    point_size: float,
) -> None:
    os.environ.pop("DEPTH3D_TEST_FAKE_VISPY", None)
    from PyQt6.QtCore import Qt
    from PyQt6.QtWidgets import QLabel, QMainWindow, QVBoxLayout, QWidget
    from app.widgets.viewport.vispy_canvas import PointCloudCanvas

    window = QMainWindow()
    central = QWidget()
    layout = QVBoxLayout(central)
    label = QLabel(f"I4.3G4C BLIND {blind_label} | {view} | {framing} | SAME SfM FRAME + SAME CAMERA + SAME DRAW COUNT")
    label.setAlignment(Qt.AlignmentFlag.AlignCenter)
    layout.addWidget(label)
    canvas = PointCloudCanvas(window)
    canvas.set_point_size(float(point_size))
    canvas.set_grid_visible(True)
    canvas.update_points(points, colors)
    camera = getattr(canvas, "_camera", None)
    if camera is None:
        canvas.close(); window.close()
        raise ReconstructionValidationError("I4.3G4C canonical render requires PointCloudCanvas camera access.")
    camera.center = tuple(float(v) for v in center_viewer)
    camera.distance = float(distance)
    camera.fov = 45.0
    camera.azimuth, camera.elevation = map(float, MATCHED_VIEW_VIEWS[view])
    changed = getattr(camera, "view_changed", None)
    if callable(changed):
        changed()
    layout.addWidget(canvas.native, 1)
    window.setCentralWidget(central)
    window.setFixedSize(_CANONICAL_TILE_W, _CANONICAL_TILE_H)
    window.show()
    for _ in range(18):
        app.processEvents(); time.sleep(0.02)
    image = window.grab().toImage()
    if image.isNull():
        canvas.close(); window.close()
        raise ReconstructionValidationError(f"I4.3G4C could not capture canonical render: {output}")
    if (image.width(), image.height()) != (_CANONICAL_TILE_W, _CANONICAL_TILE_H):
        image = image.scaled(_CANONICAL_TILE_W, _CANONICAL_TILE_H, Qt.AspectRatioMode.IgnoreAspectRatio, Qt.TransformationMode.SmoothTransformation)
    output.parent.mkdir(parents=True, exist_ok=True)
    if not image.save(str(output), "PNG"):
        canvas.close(); window.close()
        raise ReconstructionValidationError(f"I4.3G4C could not save canonical render: {output}")
    canvas.close(); window.close(); app.processEvents()


def _canonical_placeholder(*, blind_label: str, output: Path) -> None:
    image = np.zeros((_CANONICAL_TILE_H, _CANONICAL_TILE_W, 3), dtype=np.uint8)
    lines = [
        f"BLIND {blind_label}",
        "NO QUALIFIED CANONICAL G4B GEOMETRY",
        "Frozen production terminal outcome preserved",
        "Use native-depth panels where assessable",
        "Do not infer a replacement geometry",
    ]
    for i, text in enumerate(lines):
        cv2.putText(image, text, (28, 90 + i * 58), cv2.FONT_HERSHEY_SIMPLEX, 0.66 if i == 0 else 0.52, (235, 235, 235), 1, cv2.LINE_AA)
    output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output), image):
        raise ReconstructionValidationError(f"I4.3G4C could not write canonical placeholder: {output}")


def _review_guide(path: Path, *, scenario_id: str) -> Path:
    text = f"""I4.3G4C BLINDED MANUAL REVIEW — {scenario_id}

Review the generated native_depth_panels/ and canonical_geometry_panels/ before editing manual_review.json.

For every one of the seven dimensions, fill preferred_blind_labels with zero or more A-F labels. Multiple labels are allowed and mean no forced single winner. Fill not_assessable_blind_labels only where the frozen evidence genuinely does not support a judgement. Then set that dimension's dimension_confirmation to true.

Dimensions:
- global_completeness
- local_shape_plausibility
- edge_definition
- small_detail_preservation
- texture_sensitivity
- cross_view_stability_after_sfm_alignment
- false_geometry_and_floaters

Rules:
- Do not try to infer provider identities from A-F.
- Native depth uses one fixed provider-internal display range for the whole scenario; absolute colors are not comparable as metric scale across providers.
- Canonical 3D uses the same PyCOLMAP-derived frame/camera and equal rendered point count for every available cloud.
- A missing canonical tile is intentional when G4B did not produce qualified canonical geometry. Do not fabricate or rerun it.
- Production PASS/DEGRADED/REJECTED status is primary and cannot be overridden by visual preference.
- No provider ranking, universal winner, Auto policy, hybrid policy, metric-scale claim, or metrology claim is authorized here.

After all seven dimension_confirmation values are true, set reviewer_confirmation to true. Leave blind_key_revealed=false. The reveal command will fail closed otherwise.
"""
    path.write_text(text, encoding="utf-8")
    return path


def _prepare(args: argparse.Namespace) -> int:
    out = args.output_dir.expanduser().resolve()
    if out.exists() and any(out.iterdir()):
        raise ReconstructionValidationError(f"I4.3G4C prepare output must be empty or absent: {out}")

    g1_path = args.g1_snapshot.expanduser().resolve()
    g1_sha = verify_sha256(g1_path, args.g1_sha256, label="G1 qualification snapshot")
    g1 = read_json_object(g1_path, label="G1 qualification snapshot")
    g1_entry = verify_g1_single_image_review_requirement(g1, scenario_id=args.scenario)
    sources = load_g4b_matched_view_sources(
        args.production_snapshot,
        expected_sha256=args.production_sha256,
        scenario_id=args.scenario,
    )
    sfm = read_json_object(sources.authoritative_sfm_path, label="authoritative SfM snapshot")
    shared_frame = derive_shared_visual_frame(sfm)
    rgb_paths = _workspace_rgb_map(sfm, common_keyframes=sources.common_keyframe_ids)

    assignment = blind_assignment(sources.production_snapshot_sha256, scenario_id=args.scenario)
    commitment = assignment_commitment(
        assignment,
        seed_sha256=sources.production_snapshot_sha256,
        scenario_id=args.scenario,
    )
    provider_to_blind = {provider: label for label, provider in assignment.items()}

    # Native-frame review is supplemental and admitted only when G4B itself
    # content-addressed every selected frame artifact. In the current G4B
    # evidence this is the rejection path. Successful providers are reviewed
    # from their frozen canonical fused geometry instead of unhashed worker
    # frame files.
    native_data: dict[str, dict[str, tuple[np.ndarray, np.ndarray]]] = {}
    display_ranges: dict[str, tuple[float, float]] = {}
    for implementation in I43G4C_PROVIDER_IMPLEMENTATIONS:
        source = sources.providers[implementation]
        if not source.native_depth_artifacts:
            continue
        per_view: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for keyframe_id in sources.common_keyframe_ids:
            if keyframe_id not in source.native_depth_artifacts:
                raise ReconstructionValidationError(
                    f"I4.3G4C hash-bound native review coverage is incomplete for {implementation}."
                )
            path, expected = source.native_depth_artifacts[keyframe_id]
            verify_sha256(path, expected, label=f"{implementation} native depth {keyframe_id}")
            per_view[keyframe_id] = _load_native_depth(path)
        native_data[implementation] = per_view
        display_ranges[implementation] = _provider_display_range(list(per_view.values()))

    canonical_clouds: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    canonical_counts: dict[str, int] = {}
    for implementation in I43G4C_PROVIDER_IMPLEMENTATIONS:
        source = sources.providers[implementation]
        if source.canonical_geometry_path is None:
            continue
        verify_sha256(source.canonical_geometry_path, str(source.canonical_geometry_sha256), label=f"{implementation} canonical geometry")
        points, colors = _load_canonical_cloud(source.canonical_geometry_path)
        canonical_clouds[implementation] = (points, colors)
        canonical_counts[implementation] = len(points)
    if not canonical_clouds:
        raise ReconstructionValidationError("I4.3G4C requires at least one qualified canonical G4B cloud for 3D review.")
    equal_draw_count = min(int(args.point_budget), *(canonical_counts.values()))
    if equal_draw_count <= 0:
        raise ReconstructionValidationError("I4.3G4C equal canonical draw count is invalid.")
    canonical_draw: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for implementation, (points, colors) in canonical_clouds.items():
        draw_points, draw_colors = enforce_point_budget(points, colors, equal_draw_count, seed=0)
        if len(draw_points) != equal_draw_count:
            raise ReconstructionValidationError("I4.3G4C equal canonical render density contract failed.")
        canonical_draw[implementation] = (draw_points, draw_colors)

    out.mkdir(parents=True, exist_ok=False)
    native_dir = out / "native_depth_panels"; native_dir.mkdir()
    canonical_dir = out / "canonical_geometry_panels"; canonical_dir.mkdir()

    native_artifacts: dict[str, Any] = {}
    native_blind_labels = [label for label in I43G4C_BLIND_LABELS if assignment[label] in native_data]
    if native_blind_labels:
        for keyframe_id in sources.common_keyframe_ids:
            tiles = [_rgb_tile(rgb_paths[keyframe_id], keyframe_id=keyframe_id)]
            for label in native_blind_labels:
                implementation = assignment[label]
                low, high = display_ranges[implementation]
                depth, valid = native_data[implementation][keyframe_id]
                tiles.append(_depth_tile(depth, valid, low=low, high=high, label=label))
            tiles.append(_policy_tile())
            target = native_dir / f"keyframe_{keyframe_id}_blind_native_depth_grid.png"
            _grid(tiles, columns=min(4, len(tiles)), output=target)
            native_artifacts[keyframe_id] = _artifact(target)
            print(f"g4c_native_depth {args.scenario} {keyframe_id}: {target}")

    from PyQt6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication(sys.argv[:1])
    canonical_artifacts: dict[str, Any] = {}
    for framing in args.framings:
        for view in args.views:
            tiles: list[np.ndarray] = []
            tile_refs: dict[str, Any] = {}
            for label in I43G4C_BLIND_LABELS:
                implementation = assignment[label]
                target = canonical_dir / f"{framing}_{view}_blind_{label.lower()}.png"
                if implementation in canonical_draw:
                    points, colors = canonical_draw[implementation]
                    _render_cloud_tile(
                        app=app,
                        points=points,
                        colors=colors,
                        blind_label=label,
                        view=view,
                        framing=framing,
                        output=target,
                        center_viewer=shared_frame.center_viewer,
                        distance=shared_frame.camera_distance(framing),
                        point_size=args.point_size,
                    )
                else:
                    _canonical_placeholder(blind_label=label, output=target)
                image = cv2.imread(str(target), cv2.IMREAD_COLOR)
                if image is None or image.shape[:2] != (_CANONICAL_TILE_H, _CANONICAL_TILE_W):
                    raise ReconstructionValidationError(f"I4.3G4C canonical review tile is unreadable: {target}")
                tiles.append(image)
                tile_refs[label] = _artifact(target)
            grid_path = canonical_dir / f"{framing}_{view}_blind_grid.png"
            _grid(tiles, columns=3, output=grid_path)
            canonical_artifacts[f"{framing}:{view}"] = {
                "grid": _artifact(grid_path),
                "blind_tiles": tile_refs,
            }
            print(f"g4c_canonical {args.scenario} {framing}/{view}: {grid_path}")

    commitment_path = _write_json(
        out / "blind_assignment_commitment.json",
        {
            "schema_version": 1,
            "phase": I43G4C_PHASE,
            "scenario_id": args.scenario,
            "blind_labels": list(I43G4C_BLIND_LABELS),
            "assignment_commitment_sha256": commitment,
            "mapping_in_public_review_package": False,
            "reveal_only_after_completed_manual_review": True,
        },
    )
    canonical_available_labels = [label for label in I43G4C_BLIND_LABELS if assignment[label] in canonical_clouds]
    review_path = _write_json(
        out / "manual_review.json",
        build_manual_review_template(
            scenario_id=args.scenario,
            assignment_commitment_sha256=commitment,
            common_keyframe_ids=sources.common_keyframe_ids,
            canonical_available_labels=canonical_available_labels,
        ),
    )
    guide_path = _review_guide(out / "REVIEW_GUIDE.txt", scenario_id=args.scenario)

    status_counts = {status: sum(source.status == status for source in sources.providers.values()) for status in ("pass", "degraded", "rejected")}
    summary = {
        "schema_version": 1,
        "phase": I43G4C_PHASE,
        "status": "ready_for_blinded_manual_review",
        "pass": True,
        "scenario_id": args.scenario,
        "comparison_mode": "matched_view_diagnostic",
        "g1_requirement": {
            "path": str(g1_path),
            "sha256": g1_sha,
            "manual_review_required": g1_entry.get("manual_review_required"),
            "manual_review_dimensions": list(I43G4C_REVIEW_DIMENSIONS),
        },
        "g4b_production_snapshot": {
            "path": str(sources.production_snapshot_path),
            "sha256": sources.production_snapshot_sha256,
        },
        "production_status_counts": status_counts,
        "authoritative_sfm": {
            "path": str(sources.authoritative_sfm_path),
            "sha256": sources.authoritative_sfm_sha256,
            "frame_id": sources.authoritative_sfm_frame_id,
        },
        "matched_views": {
            "selection_policy": "intersection_of_all_six_frozen_G4B_adaptive_selected_view_ids",
            "common_view_count": len(sources.common_keyframe_ids),
            "keyframe_ids": list(sources.common_keyframe_ids),
            "same_frozen_source_keyframes_for_all_six": True,
        },
        "native_depth_review": {
            "scope": "supplemental_only_for_provider_frames_individually_hash_bound_by_G4B",
            "blind_labels_included": native_blind_labels,
            "unhashed_successful_provider_worker_frames_consumed": False,
            "provider_native_semantics_preserved": True,
            "one_fixed_display_range_per_included_provider_over_full_common_scenario_view_set": True,
            "display_quantiles": [0.02, 0.98],
            "per_frame_auto_fit": False,
            "cross_provider_absolute_color_value_comparison_permitted": False,
            "display_ranges_hidden_until_reveal": True,
            "panels": native_artifacts,
        },
        "canonical_geometry_review": {
            "shared_visual_frame": shared_frame.as_dict(),
            "views": list(args.views),
            "framings": list(args.framings),
            "requested_point_budget_per_available_cloud": int(args.point_budget),
            "equal_rendered_point_count_per_available_cloud": int(equal_draw_count),
            "equal_render_density_enforced": True,
            "independent_provider_auto_fit": False,
            "canonical_geometry_available_blind_labels": canonical_available_labels,
            "missing_canonical_geometry_is_not_synthesized": True,
            "panels": canonical_artifacts,
        },
        "blinding": {
            "mapping_written_to_review_package": False,
            "assignment_commitment_sha256": commitment,
            "commitment": _artifact(commitment_path),
            "reveal_only_after_completed_manual_review": True,
        },
        "manual_review": {
            "path": str(review_path),
            "initial_sha256": sha256_file(review_path),
            "editable": True,
            "guide": _artifact(guide_path),
            "required_dimensions": list(I43G4C_REVIEW_DIMENSIONS),
            "completed": False,
        },
        "scope": {
            "sfm_rerun": False,
            "provider_rerun": False,
            "provider_outputs_modified": False,
            "matched_view_diagnostic_secondary": True,
            "production_native_evidence_primary": True,
            "manual_review_started": False,
            "blind_key_revealed": False,
        },
        "authority": {
            "camera_pose_authority_owner": "pycolmap_sfm",
            "authoritative_sfm_modified": False,
            "provider_selected": False,
            "provider_promoted": False,
            "auto_policy_implemented": False,
            "hybrid_authorized": False,
            "metric_scale_claimed": False,
            "measurement_authoritative": False,
            "certified_metrology": False,
            "surface_mesh_generated": False,
            "permanent_global_view_cap": None,
        },
    }
    summary_path = _write_json(out / "i4_3g4c_prepare_summary.json", summary)
    print("I4.3G4C single-image matched-view: READY FOR BLINDED MANUAL REVIEW")
    print(f"scenario={args.scenario} common_views={len(sources.common_keyframe_ids)} status_counts={status_counts}")
    print(f"canonical_available_blind_labels={','.join(canonical_available_labels)} equal_draw_count={equal_draw_count}")
    print("sfm_rerun=False provider_rerun=False provider_selected=False mapping_written=False")
    print(f"manual_review={review_path}")
    print(f"review_guide={guide_path}")
    print(f"prepare_summary={summary_path}")
    return 0


def _reveal(args: argparse.Namespace) -> int:
    summary_path = args.prepare_summary.expanduser().resolve()
    summary_sha = verify_sha256(summary_path, args.prepare_sha256, label="G4C prepare summary")
    summary = read_json_object(summary_path, label="G4C prepare summary")
    if summary.get("phase") != I43G4C_PHASE or summary.get("status") != "ready_for_blinded_manual_review" or summary.get("pass") is not True:
        raise ReconstructionValidationError("I4.3G4C reveal requires a ready passing prepare summary.")
    review_path = args.manual_review.expanduser().resolve()
    review = read_json_object(review_path, label="completed G4C manual review")
    if not review_is_complete(review):
        raise ReconstructionValidationError("I4.3G4C manual review is incomplete; confirm all seven dimensions and reviewer_confirmation=true.")
    if review.get("scenario_id") != summary.get("scenario_id"):
        raise ReconstructionValidationError("I4.3G4C manual review scenario does not match prepare summary.")
    prod_ref = _mapping_for_cli(summary.get("g4b_production_snapshot"), label="G4B production snapshot reference")
    prod_path = Path(str(prod_ref.get("path", ""))).expanduser().resolve()
    prod_sha = str(prod_ref.get("sha256", ""))
    sources = load_g4b_matched_view_sources(
        prod_path,
        expected_sha256=prod_sha,
        scenario_id=str(summary.get("scenario_id")),
    )
    payload = build_controlled_reveal(
        review=review,
        production_snapshot_sha256=sources.production_snapshot_sha256,
        production_statuses=sources.production_statuses,
        prepare_summary_sha256=summary_sha,
    )
    payload["evidence_bindings"] = {
        "prepare_summary": {"path": str(summary_path), "sha256": summary_sha},
        "manual_review": {"path": str(review_path), "sha256": sha256_file(review_path)},
        "g4b_production_snapshot": {"path": str(prod_path), "sha256": sources.production_snapshot_sha256},
    }
    output = args.output.expanduser().resolve()
    if output.exists():
        raise ReconstructionValidationError(f"I4.3G4C controlled reveal output already exists: {output}")
    _write_json(output, payload)
    print("I4.3G4C controlled blind reveal: PASS")
    print(f"scenario={payload['scenario_id']} production_statuses={payload['production_statuses']}")
    print("production_rejection_overridden=False provider_selected=False provider_ranked=False universal_winner=False")
    print(f"controlled_reveal={output}")
    return 0


def _mapping_for_cli(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ReconstructionValidationError(f"I4.3G4C requires mapping section: {label}")
    return value


def _close(args: argparse.Namespace) -> int:
    refs = (
        ("g4_single_image_complete_room", args.complete_room_reveal, args.complete_room_sha256),
        ("g4_single_image_low_texture", args.low_texture_reveal, args.low_texture_sha256),
        ("g4_single_image_weak_overlap", args.weak_overlap_reveal, args.weak_overlap_sha256),
    )
    payloads: list[Mapping[str, Any]] = []
    bindings: dict[str, Any] = {}
    for scenario_id, path_value, expected_sha in refs:
        path = path_value.expanduser().resolve()
        actual = verify_sha256(path, expected_sha, label=f"{scenario_id} controlled reveal")
        payload = read_json_object(path, label=f"{scenario_id} controlled reveal")
        if payload.get("scenario_id") != scenario_id:
            raise ReconstructionValidationError(f"I4.3G4C controlled reveal scenario mismatch for {scenario_id}.")
        payloads.append(payload)
        bindings[scenario_id] = {"path": str(path), "sha256": actual}
    closure = build_i4_3g4c_final_closure(payloads)
    closure["evidence_bindings"] = bindings
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ReconstructionValidationError(f"I4.3G4C closure output must be empty or absent: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    closure_path = _write_json(output_dir / "i4_3g4c_final_closure.json", closure)
    print("I4.3G4 single-image provider qualification: FORMALLY CLOSED")
    print("scenario_count=3 all_manual_reviews_complete=True i4_3g4_closed=True i4_3g5_authorized=True")
    print("provider_selected=False provider_promoted=False provider_ranked=False universal_winner=False metric_scale=False")
    print(f"closure={closure_path}")
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    prepare = sub.add_parser("prepare", help="Create one blinded G4C review package from frozen G4B evidence.")
    prepare.add_argument("--scenario", choices=I43G4C_SCENARIOS, required=True)
    prepare.add_argument("--g1-snapshot", type=Path, required=True)
    prepare.add_argument("--g1-sha256", required=True)
    prepare.add_argument("--production-snapshot", type=Path, required=True)
    prepare.add_argument("--production-sha256", required=True)
    prepare.add_argument("--output-dir", type=Path, required=True)
    prepare.add_argument("--point-budget", type=int, default=180_000)
    prepare.add_argument("--point-size", type=float, default=2.0)
    prepare.add_argument("--views", default="oblique,front,side,top")
    prepare.add_argument("--framings", default="detail,context")
    prepare.set_defaults(func=_prepare)

    reveal = sub.add_parser("reveal", help="Reveal A-F mapping only after the seven-dimension blinded review is complete.")
    reveal.add_argument("--prepare-summary", type=Path, required=True)
    reveal.add_argument("--prepare-sha256", required=True)
    reveal.add_argument("--manual-review", type=Path, required=True)
    reveal.add_argument("--output", type=Path, required=True)
    reveal.set_defaults(func=_reveal)

    close = sub.add_parser("close", help="Formally close G4 after all three controlled scenario reveals exist.")
    close.add_argument("--complete-room-reveal", type=Path, required=True)
    close.add_argument("--complete-room-sha256", required=True)
    close.add_argument("--low-texture-reveal", type=Path, required=True)
    close.add_argument("--low-texture-sha256", required=True)
    close.add_argument("--weak-overlap-reveal", type=Path, required=True)
    close.add_argument("--weak-overlap-sha256", required=True)
    close.add_argument("--output-dir", type=Path, required=True)
    close.set_defaults(func=_close)
    return parser


def main() -> int:
    args = _parser().parse_args()
    if getattr(args, "point_budget", 1) <= 0 or getattr(args, "point_size", 1.0) <= 0:
        raise SystemExit("point budget and point size must be positive")
    if hasattr(args, "views"):
        args.views = [value.strip() for value in args.views.split(",") if value.strip()]
        args.framings = [value.strip() for value in args.framings.split(",") if value.strip()]
        if not args.views or any(value not in MATCHED_VIEW_VIEWS for value in args.views):
            raise SystemExit("--views contains an unsupported fixed view")
        if not args.framings or any(value not in MATCHED_VIEW_FRAMINGS for value in args.framings):
            raise SystemExit("--framings contains an unsupported shared framing")
        if len(set(args.views)) != len(args.views) or len(set(args.framings)) != len(args.framings):
            raise SystemExit("views/framings must be unique")
    try:
        return int(args.func(args))
    except Exception as exc:  # noqa: BLE001 - CLI boundary
        print(f"I4.3G4C ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
