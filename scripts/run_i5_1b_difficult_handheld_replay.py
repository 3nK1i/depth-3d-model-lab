#!/usr/bin/env python3
"""Run the fresh I5.1B difficult-handheld replay on the frozen I4 foundation.

The replay is deliberately like-for-like with the historical Horse campaign:
current I1 global baseline -> existing I4.1.6A localization -> existing I4.1.6B
DA3/dense A/B.  Controlled experiment bounds are recorded explicitly and do
not become production caps.  PASS/DEGRADED/REJECTED are all legitimate replay
quality outcomes when the evidence chain completes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping

import cv2

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.reconstruction.errors import ReconstructionValidationError  # noqa: E402
from core.reconstruction.i5_difficult_handheld_robustness import (  # noqa: E402
    build_i5_1b_difficult_handheld_replay,
    validate_i5_1a_for_replay,
)
from core.reconstruction.offline_dense_i4_decoupled_sampling_integration import (  # noqa: E402
    probe_video_metadata,
)

_DEFAULT_SFM_PYTHON = Path.home() / "anaconda3/envs/depth3d-pycolmap-probe/bin/python3.11"
_DEFAULT_DA3_PYTHON = Path.home() / "anaconda3/envs/depth3d-da3/bin/python3.11"
_DEFAULT_DA3_REPO = Path.home() / "Desktop/depth3d-da3-backends/depth-anything-3"
OUTPUT_NAME = "i5_1b_difficult_handheld_replay.json"


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--i5-1a-baseline", type=Path, required=True)
    p.add_argument("--i5-1a-sha256", required=True)
    p.add_argument("--video", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--sfm-python", type=Path, default=_DEFAULT_SFM_PYTHON)
    p.add_argument("--da3-python", type=Path, default=_DEFAULT_DA3_PYTHON)
    p.add_argument("--da3-repo", type=Path, default=_DEFAULT_DA3_REPO)
    p.add_argument("--profile", choices=("efficient", "balanced", "high_quality"), default="efficient")
    p.add_argument("--matching", choices=("exhaustive",), default="exhaustive")
    p.add_argument("--max-keyframes", type=int, default=40)
    p.add_argument("--max-dense-views", type=int, default=40)
    p.add_argument("--dense-candidate-cap", type=int, default=120)
    p.add_argument("--video-target-fps", type=float, default=2.0)
    p.add_argument("--local-target-fps", type=float, default=2.0)
    p.add_argument("--process-res", type=int, default=392)
    p.add_argument("--sfm-timeout-minutes", type=float, default=60.0)
    p.add_argument("--da3-timeout-seconds", type=float, default=600.0)
    p.add_argument("--da3-total-timeout-seconds", type=float, default=7200.0)
    args = p.parse_args()
    if args.max_keyframes < 10 or args.max_dense_views < 10:
        p.error("I5.1B controlled replay requires at least 10 global keyframes/dense views")
    if args.dense_candidate_cap < args.max_keyframes:
        p.error("--dense-candidate-cap must be >= --max-keyframes")
    if args.video_target_fps <= 0.0 or args.local_target_fps <= 0.0:
        p.error("replay target FPS values must be positive")
    if args.process_res <= 0 or args.sfm_timeout_minutes <= 0.0 or args.da3_timeout_seconds <= 0.0 or args.da3_total_timeout_seconds <= 0.0:
        p.error("replay execution limits must be positive")
    return p


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ReconstructionValidationError(f"I5.1B cannot read {label} {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ReconstructionValidationError(f"I5.1B {label} root must be an object: {path}")
    return value


def _write_json(path: Path, payload: Mapping[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    return path


def _verify_sha(path: Path, expected: str, *, label: str) -> str:
    expected_text = str(expected).strip().lower()
    if len(expected_text) != 64 or any(ch not in "0123456789abcdef" for ch in expected_text):
        raise ReconstructionValidationError(f"I5.1B {label} expected SHA256 is malformed.")
    actual = _sha256(path)
    if actual != expected_text:
        raise ReconstructionValidationError(f"I5.1B {label} SHA256 mismatch: expected={expected_text} actual={actual}")
    return actual


def _source_identity(video: Path) -> dict[str, Any]:
    total_frames, fps, duration = probe_video_metadata(video)
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        capture.release()
        raise ReconstructionValidationError(f"I5.1B cannot open video: {video}")
    try:
        ok, frame = capture.read()
        if not ok or frame is None:
            raise ReconstructionValidationError(f"I5.1B cannot decode first video frame: {video}")
        height, width = frame.shape[:2]
    finally:
        capture.release()
    return {
        "path": str(video.resolve()),
        "sha256": _sha256(video),
        "size_bytes": int(video.stat().st_size),
        "total_frames": int(total_frames),
        "fps": float(fps),
        "duration_seconds": float(duration),
        "width": int(width),
        "height": int(height),
    }


def build_i1_command(args: argparse.Namespace, *, video: Path, output: Path) -> list[str]:
    return [
        sys.executable,
        str(ROOT / "scripts/run_i1_offline_dense_campaign.py"),
        "--video", str(video),
        "--video-target-fps", str(args.video_target_fps),
        "--output-dir", str(output / "01_i1_baseline"),
        "--sfm-python", str(args.sfm_python.expanduser().resolve()),
        "--da3-python", str(args.da3_python.expanduser().resolve()),
        "--da3-repo", str(args.da3_repo.expanduser().resolve()),
        "--profile", str(args.profile),
        "--matching", str(args.matching),
        "--max-keyframes", str(args.max_keyframes),
        "--max-dense-views", str(args.max_dense_views),
        "--process-res", str(args.process_res),
        "--sfm-timeout-minutes", str(args.sfm_timeout_minutes),
        "--da3-timeout-seconds", str(args.da3_timeout_seconds),
    ]


def build_localization_command(args: argparse.Namespace, *, video: Path, output: Path) -> list[str]:
    return [
        sys.executable,
        str(ROOT / "scripts/run_i4_1_6_decoupled_sampling_integration.py"),
        "--i4-dir", str(output),
        "--video", str(video),
        "--output-dir", str(output / "02_decoupled_localization"),
        "--plans", "uniform_localized_dense_pool", "global_anchors_plus_local_neighbors",
        "--dense-candidate-cap", str(args.dense_candidate_cap),
        "--local-target-fps", str(args.local_target_fps),
    ]


def build_dense_ab_command(args: argparse.Namespace, *, video: Path, output: Path) -> list[str]:
    return [
        sys.executable,
        str(ROOT / "scripts/run_i4_1_6b_decoupled_dense_ab.py"),
        "--i4-dir", str(output),
        "--i4-1-6a-dir", str(output / "02_decoupled_localization"),
        "--video", str(video),
        "--output-dir", str(output / "03_real_dense_ab"),
        "--da3-python", str(args.da3_python.expanduser().resolve()),
        "--da3-repo", str(args.da3_repo.expanduser().resolve()),
        "--process-res", str(args.process_res),
        "--da3-total-timeout-seconds", str(args.da3_total_timeout_seconds),
        "--dense-candidate-cap", str(args.dense_candidate_cap),
        "--local-target-fps", str(args.local_target_fps),
    ]


def _run(command: list[str], *, log: Path) -> int:
    log.parent.mkdir(parents=True, exist_ok=True)
    print("\nI5.1B command: " + " ".join(command), flush=True)
    with log.open("w", encoding="utf-8") as handle:
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            handle.write(line)
            handle.flush()
        return int(process.wait())


def run(args: argparse.Namespace) -> Path:
    baseline_path = args.i5_1a_baseline.expanduser().resolve()
    video = args.video.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    if not baseline_path.is_file():
        raise ReconstructionValidationError(f"I5.1B I5.1A baseline is missing: {baseline_path}")
    if not video.is_file():
        raise ReconstructionValidationError(f"I5.1B video is missing: {video}")
    if output.exists() and any(output.iterdir()):
        raise ReconstructionValidationError(f"I5.1B output directory must be empty or absent: {output}")
    output.mkdir(parents=True, exist_ok=True)

    baseline_sha = _verify_sha(baseline_path, args.i5_1a_sha256, label="I5.1A baseline")
    baseline = _read_json(baseline_path, label="I5.1A baseline")
    validate_i5_1a_for_replay(baseline)
    source = _source_identity(video)
    _write_json(output / "00_source_identity.json", source)

    print("I5.1B predecessor I5.1A: PASS", flush=True)
    print(
        f"fresh_source_sha256={source['sha256']} frames={source['total_frames']} "
        f"fps={source['fps']:.6g} duration={source['duration_seconds']:.3f}s "
        f"resolution={source['width']}x{source['height']}",
        flush=True,
    )
    print(
        "controlled_replay_bounds_only=True permanent_global_view_cap=None "
        f"max_keyframes={args.max_keyframes} max_dense_views={args.max_dense_views} "
        f"dense_candidate_cap={args.dense_candidate_cap}",
        flush=True,
    )

    stage_codes: dict[str, int | None] = {"i1": None, "localization": None, "dense_ab": None}
    localization: dict[str, Any] | None = None
    dense: dict[str, Any] | None = None

    stage_codes["i1"] = _run(build_i1_command(args, video=video, output=output), log=output / "logs/01_i1_baseline.log")
    i1_path = output / "01_i1_baseline/i1_campaign_summary.json"
    if not i1_path.is_file():
        raise ReconstructionValidationError("I5.1B I1 replay did not emit i1_campaign_summary.json.")
    i1 = _read_json(i1_path, label="I1 replay summary")

    sfm_report = output / "01_i1_baseline/01_sfm/real_sfm_report.json"
    if sfm_report.is_file():
        stage_codes["localization"] = _run(
            build_localization_command(args, video=video, output=output),
            log=output / "logs/02_decoupled_localization.log",
        )
        loc_path = output / "02_decoupled_localization/i4_1_6a_summary.json"
        if not loc_path.is_file():
            raise ReconstructionValidationError("I5.1B localization stage did not emit i4_1_6a_summary.json.")
        localization = _read_json(loc_path, label="I4.1.6A localization summary")

        loc_decision = localization.get("decision", {})
        selected_path = i1.get("selected_path")
        if bool(isinstance(loc_decision, Mapping) and loc_decision.get("dense_ab_execution_authorized")) and isinstance(selected_path, Mapping):
            stage_codes["dense_ab"] = _run(
                build_dense_ab_command(args, video=video, output=output),
                log=output / "logs/03_real_dense_ab.log",
            )
            dense_path = output / "03_real_dense_ab/i4_1_6b_summary.json"
            if not dense_path.is_file():
                raise ReconstructionValidationError("I5.1B dense A/B stage did not emit i4_1_6b_summary.json.")
            dense = _read_json(dense_path, label="I4.1.6B dense A/B summary")

    result = build_i5_1b_difficult_handheld_replay(
        i5_1a_result=baseline,
        i5_1a_sha256=baseline_sha,
        source_identity=source,
        i1_summary=i1,
        localization_summary=localization,
        dense_ab_summary=dense,
        stage_return_codes=stage_codes,
        replay_parameters={
            "profile": args.profile,
            "matching": args.matching,
            "max_keyframes": args.max_keyframes,
            "max_dense_views": args.max_dense_views,
            "dense_candidate_cap": args.dense_candidate_cap,
            "video_target_fps": args.video_target_fps,
            "local_target_fps": args.local_target_fps,
            "process_res": args.process_res,
        },
    )
    target = _write_json(output / OUTPUT_NAME, result)
    quality = result["quality"]
    comparison = result["historical_comparison"]
    print("\nI5.1B Current Frozen-Foundation Difficult-Handheld Replay: EVIDENCE_COMPLETE", flush=True)
    print(
        f"quality_status={quality['status']} global_selected={quality['fresh_global_selected_qualified_views']} "
        f"best_selected={quality['fresh_best_selected_qualified_views']} dense_gate={quality['fresh_dense_gate_reached']}",
        flush=True,
    )
    print(
        f"historical_selected={comparison['historical_selected_qualified_views']} "
        f"selected_gain={comparison['selected_view_gain_over_historical']} "
        "historical_content_hash_equivalence_claimed=False",
        flush=True,
    )
    print("pycolmap_authority=True threshold_relaxation=False auto=False hybrid=False metric_scale=False metrology=False", flush=True)
    print("I5.1C difficult-handheld robustness qualification and closure: AUTHORIZED", flush=True)
    print(f"evidence={target}", flush=True)
    return target


def main() -> int:
    try:
        run(_parser().parse_args())
    except (ReconstructionValidationError, OSError, ValueError, TypeError, KeyError) as exc:
        print(f"I5.1B ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
