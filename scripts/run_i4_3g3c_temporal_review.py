#!/usr/bin/env python3
"""Prepare, reveal, and close I4.3G3C temporal matched-view review evidence.

This runner is evidence-only: it consumes frozen I4.3G3B provider outputs and
never starts PyCOLMAP, RollingDepth, or Video Depth Anything.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any, Mapping

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.reconstruction.errors import ReconstructionValidationError
from core.reconstruction.temporal_provider_matched_view_review import (
    I43G3C_DEFAULT_WINDOW_FRAMES,
    I43G3C_PHASE,
    I43G3C_SCENARIOS,
    assignment_commitment,
    blind_assignment,
    build_controlled_reveal,
    build_i4_3g3c_final_closure,
    build_manual_review_template,
    centered_contiguous_window,
    load_depth_stack,
    load_g3b_matched_view_sources,
    read_json_object,
    review_is_complete,
    rolling_depth_display_multiplier,
    sequence_global_display_range,
    sha256_file,
    verify_g1_temporal_review_requirement,
    verify_sha256,
)

PANEL_WIDTH = 480
PANEL_HEIGHT = 320
HEADER_HEIGHT = 40
FRAME_WIDTH = PANEL_WIDTH * 3
FRAME_HEIGHT = PANEL_HEIGHT + HEADER_HEIGHT


def _write_json(path: Path, payload: Mapping[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    return path


def _artifact(path: Path) -> dict[str, Any]:
    source = path.expanduser().resolve()
    if not source.is_file():
        raise ReconstructionValidationError(f"I4.3G3C expected artifact is missing: {source}")
    return {"path": str(source), "sha256": sha256_file(source), "size_bytes": source.stat().st_size}


def _fit_panel(image: np.ndarray) -> np.ndarray:
    if image.ndim != 3 or image.shape[2] != 3:
        raise ReconstructionValidationError("I4.3G3C render panel requires HxWx3 image data.")
    height, width = image.shape[:2]
    if width <= 0 or height <= 0:
        raise ReconstructionValidationError("I4.3G3C render panel has invalid dimensions.")
    scale = min(PANEL_WIDTH / float(width), PANEL_HEIGHT / float(height))
    resized = cv2.resize(
        image,
        (max(1, int(round(width * scale))), max(1, int(round(height * scale)))),
        interpolation=cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR,
    )
    canvas = np.zeros((PANEL_HEIGHT, PANEL_WIDTH, 3), dtype=np.uint8)
    y = (PANEL_HEIGHT - resized.shape[0]) // 2
    x = (PANEL_WIDTH - resized.shape[1]) // 2
    canvas[y : y + resized.shape[0], x : x + resized.shape[1]] = resized
    return canvas


def _depth_bgr(depth: np.ndarray, *, low: float, high: float, multiplier: float) -> np.ndarray:
    values = np.asarray(depth, dtype=np.float32) * float(multiplier)
    normalized = np.clip((values - float(low)) / (float(high) - float(low)), 0.0, 1.0)
    image = np.asarray(np.rint(normalized * 255.0), dtype=np.uint8)
    return cv2.applyColorMap(image, cv2.COLORMAP_TURBO)


def _header(label: str, *, width: int = PANEL_WIDTH) -> np.ndarray:
    canvas = np.zeros((HEADER_HEIGHT, width, 3), dtype=np.uint8)
    cv2.putText(
        canvas,
        label,
        (12, 27),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        (235, 235, 235),
        1,
        cv2.LINE_AA,
    )
    return canvas


def _compose_frame(
    *,
    rgb_bgr: np.ndarray,
    blind_a_bgr: np.ndarray,
    blind_b_bgr: np.ndarray,
    source_index: int,
    position: int,
    total: int,
) -> np.ndarray:
    panels = np.hstack((_fit_panel(rgb_bgr), _fit_panel(blind_a_bgr), _fit_panel(blind_b_bgr)))
    headers = np.hstack(
        (
            _header(f"SOURCE RGB | source index {source_index} | {position + 1}/{total}"),
            _header("DEPTH A | fixed sequence-global display scale"),
            _header("DEPTH B | fixed sequence-global display scale"),
        )
    )
    return np.vstack((headers, panels))


def _review_fps(timestamps_ns: tuple[int, ...], positions: tuple[int, ...]) -> float:
    selected = np.asarray([timestamps_ns[index] for index in positions], dtype=np.int64)
    if len(selected) < 2:
        return 5.0
    delta = np.diff(selected).astype(np.float64)
    median_ns = float(np.median(delta))
    if median_ns <= 0.0:
        return 5.0
    return float(np.clip(1e9 / median_ns, 2.0, 15.0))


def _encode_review_video(frame_dir: Path, output: Path, *, fps: float) -> str:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise ReconstructionValidationError("I4.3G3C requires ffmpeg to encode the blinded temporal review video.")
    input_pattern = str((frame_dir / "frame_%04d.png").resolve())
    attempts = (
        ("libx264", ["-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p"]),
        ("mpeg4", ["-c:v", "mpeg4", "-q:v", "2", "-pix_fmt", "yuv420p"]),
    )
    errors: list[str] = []
    for codec, codec_args in attempts:
        command = [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-framerate",
            f"{fps:.8g}",
            "-start_number",
            "0",
            "-i",
            input_pattern,
            *codec_args,
            str(output.resolve()),
        ]
        completed = subprocess.run(command, capture_output=True, text=True, check=False)
        if completed.returncode == 0 and output.is_file() and output.stat().st_size > 0:
            return codec
        errors.append(f"{codec}:rc={completed.returncode}:{completed.stderr.strip()}")
    raise ReconstructionValidationError("I4.3G3C ffmpeg review encoding failed: " + " | ".join(errors))


def _workspace_rgb_paths(sources, positions: tuple[int, ...]) -> list[Path]:
    by_keyframe = {str(item.get("keyframe_id")): item for item in sources.workspace_images}
    paths: list[Path] = []
    for position in positions:
        keyframe_id = sources.rolling_depth.ordered_keyframe_ids[position]
        item = by_keyframe.get(keyframe_id)
        if item is None:
            raise ReconstructionValidationError(f"I4.3G3C workspace is missing matched keyframe {keyframe_id}.")
        relative = Path(str(item.get("staged_image_path", "")))
        path = (sources.workspace_root / relative).resolve()
        verify_sha256(path, str(item.get("staged_sha256", "")), label=f"matched RGB {keyframe_id}")
        paths.append(path)
    return paths


def _prepare(args: argparse.Namespace) -> int:
    out = args.output_dir.expanduser().resolve()
    if out.exists() and any(out.iterdir()):
        raise ReconstructionValidationError(f"I4.3G3C prepare output must be empty or absent: {out}")

    g1_path = args.g1_snapshot.expanduser().resolve()
    g1_sha = verify_sha256(g1_path, args.g1_sha256, label="G1 qualification snapshot")
    g1 = read_json_object(g1_path, label="I4.3G3C G1 qualification snapshot")
    verify_g1_temporal_review_requirement(g1, scenario_id=args.scenario)

    sources = load_g3b_matched_view_sources(
        args.production_snapshot,
        expected_sha256=args.production_sha256,
        scenario_id=args.scenario,
    )
    rolling_depth = load_depth_stack(
        sources.rolling_depth.depth_npz_path,
        expected_view_count=sources.view_count,
        label="RollingDepth",
    )
    vda_depth = load_depth_stack(
        sources.video_depth_anything.depth_npz_path,
        expected_view_count=sources.view_count,
        label="Video Depth Anything",
    )
    if rolling_depth.shape != vda_depth.shape:
        raise ReconstructionValidationError(
            f"I4.3G3C matched provider depth grids differ: RollingDepth={rolling_depth.shape} VDA={vda_depth.shape}"
        )

    rolling_summary = read_json_object(sources.rolling_depth.summary_path, label="I4.3G3C RollingDepth summary")
    rolling_multiplier = rolling_depth_display_multiplier(rolling_summary)
    ranges = {
        "rolling_depth": sequence_global_display_range(rolling_depth, multiplier=rolling_multiplier),
        "video_depth_anything": sequence_global_display_range(vda_depth, multiplier=1.0),
    }
    stacks = {
        "rolling_depth": (rolling_depth, rolling_multiplier),
        "video_depth_anything": (vda_depth, 1.0),
    }

    positions = centered_contiguous_window(sources.view_count, args.window_frames)
    rgb_paths = _workspace_rgb_paths(sources, positions)
    assignment = blind_assignment(sources.production_snapshot_sha256, scenario_id=args.scenario)
    commitment = assignment_commitment(
        assignment,
        seed_sha256=sources.production_snapshot_sha256,
        scenario_id=args.scenario,
    )

    out.mkdir(parents=True, exist_ok=False)
    frames_dir = out / "review_frames"
    frames_dir.mkdir()
    rendered_frames: list[Path] = []
    for local_index, (position, rgb_path) in enumerate(zip(positions, rgb_paths, strict=True)):
        rgb = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
        if rgb is None:
            raise ReconstructionValidationError(f"I4.3G3C could not decode matched RGB: {rgb_path}")
        provider_images: dict[str, np.ndarray] = {}
        for implementation, (stack, multiplier) in stacks.items():
            low, high = ranges[implementation]
            provider_images[implementation] = _depth_bgr(stack[position], low=low, high=high, multiplier=multiplier)
        frame = _compose_frame(
            rgb_bgr=rgb,
            blind_a_bgr=provider_images[assignment["A"]],
            blind_b_bgr=provider_images[assignment["B"]],
            source_index=sources.rolling_depth.ordered_source_sequence_indices[position],
            position=local_index,
            total=len(positions),
        )
        target = frames_dir / f"frame_{local_index:04d}.png"
        if not cv2.imwrite(str(target), frame):
            raise ReconstructionValidationError(f"I4.3G3C could not write review frame: {target}")
        rendered_frames.append(target)

    representative = [rendered_frames[0], rendered_frames[len(rendered_frames) // 2], rendered_frames[-1]]
    contact_images = [cv2.imread(str(path), cv2.IMREAD_COLOR) for path in representative]
    if any(image is None for image in contact_images):
        raise ReconstructionValidationError("I4.3G3C could not reload representative review frames.")
    contact = np.vstack(contact_images)  # type: ignore[arg-type]
    contact_path = out / "blinded_temporal_contact_sheet.png"
    if not cv2.imwrite(str(contact_path), contact):
        raise ReconstructionValidationError("I4.3G3C could not write blinded temporal contact sheet.")

    fps = _review_fps(sources.rolling_depth.ordered_timestamps_ns, positions)
    video_path = out / "blinded_temporal_review.mp4"
    codec = _encode_review_video(frames_dir, video_path, fps=fps)

    commitment_path = _write_json(
        out / "blind_assignment_commitment.json",
        {
            "schema_version": 1,
            "phase": I43G3C_PHASE,
            "scenario_id": args.scenario,
            "assignment_commitment_sha256": commitment,
            "mapping_in_public_review_package": False,
            "reveal_only_after_completed_manual_review": True,
        },
    )
    review_path = _write_json(
        out / "manual_review.json",
        build_manual_review_template(
            scenario_id=args.scenario,
            assignment_commitment_sha256=commitment,
            window_positions=positions,
        ),
    )

    summary = {
        "schema_version": 1,
        "phase": I43G3C_PHASE,
        "status": "ready_for_blinded_manual_review",
        "pass": True,
        "scenario_id": args.scenario,
        "comparison_mode": "matched_view_diagnostic",
        "g1_requirement": {"path": str(g1_path), "sha256": g1_sha, "manual_review_dimensions": list(verify_g1_temporal_review_requirement(g1, scenario_id=args.scenario)["manual_review_dimensions"])},
        "g3b_production_snapshot": {"path": str(sources.production_snapshot_path), "sha256": sources.production_snapshot_sha256},
        "source_identity": {"source_file_sha256": sources.source_file_sha256},
        "authoritative_sfm": {"path": str(sources.sfm_snapshot_path), "sha256": sources.sfm_snapshot_sha256, "frame_id": sources.sfm_frame_id},
        "production_statuses": {
            "rolling_depth": sources.rolling_depth.status,
            "video_depth_anything": sources.video_depth_anything.status,
        },
        "frozen_provider_inputs": {
            "rolling_depth": {
                "evidence": {"path": str(sources.rolling_depth.evidence_path), "sha256": sources.rolling_depth.evidence_sha256},
                "summary": {"path": str(sources.rolling_depth.summary_path), "sha256": sources.rolling_depth.summary_sha256},
                "native_depth": {"path": str(sources.rolling_depth.depth_npz_path), "sha256": sources.rolling_depth.depth_npz_sha256},
            },
            "video_depth_anything": {
                "evidence": {"path": str(sources.video_depth_anything.evidence_path), "sha256": sources.video_depth_anything.evidence_sha256},
                "summary": {"path": str(sources.video_depth_anything.summary_path), "sha256": sources.video_depth_anything.summary_sha256},
                "native_depth": {"path": str(sources.video_depth_anything.depth_npz_path), "sha256": sources.video_depth_anything.depth_npz_sha256},
            },
        },
        "matched_window": {
            "selection_policy": "deterministic_centered_contiguous_provider_output_window",
            "requested_frames": int(args.window_frames),
            "view_count": len(positions),
            "positions": list(positions),
            "source_sequence_indices": [sources.rolling_depth.ordered_source_sequence_indices[index] for index in positions],
            "timestamps_ns": [sources.rolling_depth.ordered_timestamps_ns[index] for index in positions],
            "common_contiguous_source_ordered_subsequence": True,
            "provider_sequences_reordered": False,
        },
        "display_policy": {
            "provider_native_frozen_depth_used": True,
            "rolling_depth_native_to_canonical_polarity_multiplier": rolling_multiplier,
            "rolling_depth_sequence_global_display_range": list(ranges["rolling_depth"]),
            "video_depth_anything_sequence_global_display_range": list(ranges["video_depth_anything"]),
            "display_range_quantiles": [0.02, 0.98],
            "display_ranges_computed_once_over_full_frozen_provider_sequence": True,
            "per_frame_auto_fit": False,
            "independent_temporal_frame_auto_fit": False,
            "cross_provider_absolute_color_value_comparison_permitted": False,
            "depth_color_semantics": "provider-internal farther-depth ordering only; display-only; no metric authority",
            "source_rgb_same_authoritative_sfm_workspace": True,
            "output_frame_size_px": [FRAME_WIDTH, FRAME_HEIGHT],
            "review_playback_fps": fps,
            "review_playback_is_approximate_timestamp_cadence": True,
            "video_codec": codec,
        },
        "blinding": {
            "mapping_written_to_review_package": False,
            "assignment_commitment_sha256": commitment,
            "commitment": _artifact(commitment_path),
            "reveal_only_after_completed_manual_review": True,
        },
        "review_artifacts": {
            "video": _artifact(video_path),
            "contact_sheet": _artifact(contact_path),
            "manual_review": {"path": str(review_path.resolve()), "initial_sha256": sha256_file(review_path), "editable": True},
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
            "same_authoritative_sfm_shared_by_both_providers": True,
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
    summary_path = _write_json(out / "i4_3g3c_prepare_summary.json", summary)
    print("I4.3G3C temporal matched-view: READY FOR BLINDED MANUAL REVIEW")
    print(f"scenario={args.scenario} views={len(positions)} production_statuses={summary['production_statuses']}")
    print("sfm_rerun=False provider_rerun=False per_frame_auto_fit=False mapping_written=False")
    print(f"review_video={video_path}")
    print(f"contact_sheet={contact_path}")
    print(f"manual_review={review_path}")
    print(f"prepare_summary={summary_path}")
    return 0


def _reveal(args: argparse.Namespace) -> int:
    summary_path = args.prepare_summary.expanduser().resolve()
    summary_sha = verify_sha256(summary_path, args.prepare_sha256, label="G3C prepare summary")
    summary = read_json_object(summary_path, label="I4.3G3C prepare summary")
    if summary.get("phase") != I43G3C_PHASE or summary.get("status") != "ready_for_blinded_manual_review" or summary.get("pass") is not True:
        raise ReconstructionValidationError("I4.3G3C reveal requires a ready passing prepare summary.")
    review_path = args.manual_review.expanduser().resolve()
    review = read_json_object(review_path, label="I4.3G3C completed manual review")
    if not review_is_complete(review):
        raise ReconstructionValidationError("I4.3G3C manual review is incomplete; fill all six ratings and set reviewer_confirmation=true.")
    if review.get("scenario_id") != summary.get("scenario_id"):
        raise ReconstructionValidationError("I4.3G3C manual review scenario does not match prepare summary.")
    statuses = summary.get("production_statuses")
    if not isinstance(statuses, Mapping):
        raise ReconstructionValidationError("I4.3G3C prepare summary lacks frozen production statuses.")
    prod = summary.get("g3b_production_snapshot")
    if not isinstance(prod, Mapping):
        raise ReconstructionValidationError("I4.3G3C prepare summary lacks G3B snapshot identity.")
    payload = build_controlled_reveal(
        review=review,
        production_snapshot_sha256=str(prod.get("sha256", "")),
        production_statuses=statuses,
        prepare_summary_sha256=summary_sha,
    )
    payload["evidence_bindings"] = {
        "prepare_summary": {"path": str(summary_path), "sha256": summary_sha},
        "manual_review": {"path": str(review_path), "sha256": sha256_file(review_path)},
        "g3b_production_snapshot": dict(prod),
    }
    output = args.output.expanduser().resolve()
    if output.exists():
        raise ReconstructionValidationError(f"I4.3G3C controlled reveal output already exists: {output}")
    _write_json(output, payload)
    print("I4.3G3C controlled blind reveal: PASS")
    print(f"scenario={payload['scenario_id']} production_statuses={payload['production_statuses']}")
    print("provider_selected=False provider_promoted=False production_rejection_overridden=False")
    print(f"controlled_reveal={output}")
    return 0


def _close(args: argparse.Namespace) -> int:
    refs = (
        ("g3_temporal_static_sweep", args.static_reveal, args.static_sha256),
        ("g3_temporal_motion", args.motion_reveal, args.motion_sha256),
        ("g3_temporal_low_texture", args.low_texture_reveal, args.low_texture_sha256),
    )
    payloads: list[Mapping[str, Any]] = []
    bindings: dict[str, Any] = {}
    for scenario_id, path_value, sha in refs:
        path = path_value.expanduser().resolve()
        actual = verify_sha256(path, sha, label=f"{scenario_id} controlled reveal")
        payload = read_json_object(path, label=f"I4.3G3C {scenario_id} controlled reveal")
        if payload.get("scenario_id") != scenario_id:
            raise ReconstructionValidationError(f"I4.3G3C controlled reveal scenario mismatch for {scenario_id}.")
        payloads.append(payload)
        bindings[scenario_id] = {"path": str(path), "sha256": actual}
    closure = build_i4_3g3c_final_closure(payloads)
    closure["evidence_bindings"] = bindings
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ReconstructionValidationError(f"I4.3G3C closure output must be empty or absent: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    closure_path = _write_json(output_dir / "i4_3g3c_final_closure.json", closure)
    print("I4.3G3C temporal-provider qualification: FORMALLY CLOSED")
    print("scenario_count=3 all_manual_reviews_complete=True i4_3g3_closed=True i4_3g4_authorized=True")
    print("provider_selected=False provider_promoted=False universal_provider_winner=False metric_scale=False")
    print(f"closure={closure_path}")
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    prepare = sub.add_parser("prepare", help="Create one blinded matched-view temporal review package from frozen G3B evidence.")
    prepare.add_argument("--scenario", choices=I43G3C_SCENARIOS, required=True)
    prepare.add_argument("--g1-snapshot", type=Path, required=True)
    prepare.add_argument("--g1-sha256", required=True)
    prepare.add_argument("--production-snapshot", type=Path, required=True)
    prepare.add_argument("--production-sha256", required=True)
    prepare.add_argument("--window-frames", type=int, default=I43G3C_DEFAULT_WINDOW_FRAMES)
    prepare.add_argument("--output-dir", type=Path, required=True)
    prepare.set_defaults(func=_prepare)

    reveal = sub.add_parser("reveal", help="Reveal A/B mapping only after the six-dimension blinded review is complete.")
    reveal.add_argument("--prepare-summary", type=Path, required=True)
    reveal.add_argument("--prepare-sha256", required=True)
    reveal.add_argument("--manual-review", type=Path, required=True)
    reveal.add_argument("--output", type=Path, required=True)
    reveal.set_defaults(func=_reveal)

    close = sub.add_parser("close", help="Formally close G3 after all three controlled scenario reveals exist.")
    close.add_argument("--static-reveal", type=Path, required=True)
    close.add_argument("--static-sha256", required=True)
    close.add_argument("--motion-reveal", type=Path, required=True)
    close.add_argument("--motion-sha256", required=True)
    close.add_argument("--low-texture-reveal", type=Path, required=True)
    close.add_argument("--low-texture-sha256", required=True)
    close.add_argument("--output-dir", type=Path, required=True)
    close.set_defaults(func=_close)
    return parser


def main() -> int:
    args = _parser().parse_args()
    try:
        return int(args.func(args))
    except Exception as exc:
        print(f"I4.3G3C ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
