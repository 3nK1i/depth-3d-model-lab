#!/usr/bin/env python3
"""Phase 9.5H single-held-out-view NeRF quality characterization.

The South Building closure input currently contains only three source frames
(two train + one held-out).  That is intentionally below the established Phase
8.7E3 production-style sanity policy (>=8 train + >=3 held-out), so this script
records real PSNR/SSIM/LPIPS evidence for the single held-out view without
claiming production quality certification.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.reconstruction_nerf_interactive import (
    NerfInteractiveRenderRequest,
    build_interactive_nerf_presentation,
    verify_interactive_nerf_integrity,
)
from core.reconstruction.nerf_interactive_runtime import (
    current_process_nerf_heavy_modules,
    default_phase9_nerf_runtime_spec,
)
from core.reconstruction.persistent_scene import read_persistent_reconstruction_scene
from scripts.run_phase9_5g_real_nerf_resource_check import (
    _child_env,
    _graceful_stop,
    _read_protocol,
    _wait_released,
)

PREFIX = "DEPTH3D_NERF_JSON "
DEFAULT_METRICS_PYTHON = Path.home() / "anaconda3/envs/eia-gaussian-metrics/bin/python"


def _json_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"Expected JSON object: {path}")
    return payload


def _fit_resolution(width: int, height: int, *, max_dimension: int = 640) -> tuple[int, int]:
    width = int(width)
    height = int(height)
    if width <= 0 or height <= 0:
        raise RuntimeError("Held-out source dimensions must be positive")
    scale = min(1.0, float(max_dimension) / float(max(width, height)))
    out_w = max(160, int(round(width * scale)))
    out_h = max(160, int(round(height * scale)))
    if max(out_w, out_h) > 1280:
        shrink = 1280.0 / float(max(out_w, out_h))
        out_w = max(160, int(round(out_w * shrink)))
        out_h = max(160, int(round(out_h * shrink)))
    return out_w, out_h


def _run_metrics(metrics_python: Path, manifest: Path, result: Path, *, timeout_s: float) -> dict[str, Any]:
    worker = ROOT / "scripts/gaussian_benchmark_workers/metrics_worker.py"
    completed = subprocess.run(
        [str(metrics_python), str(worker), "--render-manifest", str(manifest), "--result", str(result), "--device", "cpu"],
        cwd=str(ROOT),
        env={**os.environ, "PYTHONNOUSERSITE": "1"},
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=float(timeout_s),
        check=False,
    )
    (result.parent / "metrics_console.log").write_text(completed.stdout or "", encoding="utf-8")
    if completed.returncode != 0:
        tail = "\n".join((completed.stdout or "").splitlines()[-30:])
        raise RuntimeError(f"Held-out metrics worker failed ({completed.returncode}):\n{tail}")
    return _json_object(result)


def _contact_sheet(reference: Path, rendered: Path, metrics: dict[str, float], target: Path) -> None:
    with Image.open(reference) as image:
        ref = image.convert("RGB")
    with Image.open(rendered) as image:
        pred = image.convert("RGB")
    label_h = 44
    canvas = Image.new("RGB", (ref.width * 2, ref.height + label_h), "white")
    canvas.paste(ref, (0, label_h))
    canvas.paste(pred, (ref.width, label_h))
    draw = ImageDraw.Draw(canvas)
    draw.text((6, 6), "HELD-OUT REFERENCE", fill="black")
    draw.text(
        (ref.width + 6, 6),
        f"NERFACTO  PSNR {metrics['psnr_db']:.2f}  SSIM {metrics['ssim']:.3f}  LPIPS {metrics['lpips']:.3f}",
        fill="black",
    )
    canvas.save(target)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-manifest", required=True)
    parser.add_argument("--output-dir", default="/tmp/phase9_5h_heldout_quality")
    parser.add_argument("--metrics-python", type=Path, default=DEFAULT_METRICS_PYTHON)
    parser.add_argument("--timeout-s", type=float, default=180.0)
    args = parser.parse_args()

    out = Path(args.output_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    summary_path = out / "phase9_5h_heldout_quality_summary.json"
    heavy_before = list(current_process_nerf_heavy_modules())
    proc: subprocess.Popen[str] | None = None
    worker_pid = 0
    graceful = False
    released = False
    try:
        scene_manifest = Path(args.scene_manifest).expanduser().resolve()
        scene = read_persistent_reconstruction_scene(scene_manifest)
        presentation = build_interactive_nerf_presentation(scene)
        if presentation is None or not presentation.ready:
            raise RuntimeError("Persistent scene has no ready Phase 9.5 NeRF representation")
        verify_interactive_nerf_integrity(scene, presentation)

        transforms_path = Path(presentation.transforms_path).resolve()
        transforms = _json_object(transforms_path)
        frames = transforms.get("frames")
        val_files = transforms.get("val_filenames")
        train_files = transforms.get("train_filenames")
        if not isinstance(frames, list) or not frames:
            raise RuntimeError("Persisted transforms contain no frames")
        if not isinstance(val_files, list) or len(val_files) != 1:
            raise RuntimeError("Phase 9.5H South Building closure expects exactly one held-out validation frame")
        if not isinstance(train_files, list) or len(train_files) != 2:
            raise RuntimeError("Phase 9.5H South Building closure expects exactly two training frames")
        val_file = str(val_files[0])
        frame = next((item for item in frames if isinstance(item, dict) and str(item.get("file_path")) == val_file), None)
        if frame is None:
            raise RuntimeError("Held-out validation frame is missing from persisted transforms")

        source = (Path(presentation.dataset_root).resolve() / val_file).resolve()
        if not source.is_file():
            raise RuntimeError(f"Held-out reference image is missing: {source}")
        src_w = int(frame["w"])
        src_h = int(frame["h"])
        width, height = _fit_resolution(src_w, src_h)
        sx = float(width) / float(src_w)
        sy = float(height) / float(src_h)
        reference = out / "heldout_reference.png"
        with Image.open(source) as image:
            image.convert("RGB").resize((width, height), Image.Resampling.LANCZOS).save(reference)

        request = NerfInteractiveRenderRequest(
            request_id=1,
            camera_to_world_gl=tuple(tuple(float(v) for v in row) for row in frame["transform_matrix"]),
            width=width,
            height=height,
            fx=float(frame["fl_x"]) * sx,
            fy=float(frame["fl_y"]) * sy,
            cx=float(frame["cx"]) * sx,
            cy=float(frame["cy"]) * sy,
            scene_id=presentation.scene_id,
            artifact_id=presentation.artifact_id,
        )

        runtime = default_phase9_nerf_runtime_spec()
        worker = ROOT / "scripts/nerf_workers/nerfstudio_interactive_worker.py"
        session = out / "session"
        session.mkdir(parents=True, exist_ok=True)
        console = out / "nerfstudio_interactive_console.log"
        with console.open("w", encoding="utf-8") as log:
            proc = subprocess.Popen(
                [runtime.python_executable, str(worker), "--load-spec", presentation.load_spec_path, "--session-dir", str(session)],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=log,
                text=True,
                bufsize=1,
                env=_child_env(),
                cwd=str(ROOT),
                start_new_session=True,
            )
            ready = _read_protocol(proc, timeout_s=float(args.timeout_s), expected="ready")
            worker_pid = int(ready.get("worker_pid", proc.pid) or proc.pid)
            assert proc.stdin is not None
            proc.stdin.write(json.dumps(request.as_protocol_payload(), separators=(",", ":")) + "\n")
            proc.stdin.flush()
            frame_result = _read_protocol(proc, timeout_s=float(args.timeout_s), expected="frame")
            rendered = Path(str(frame_result["frame_path"])).resolve()
            stopped, graceful = _graceful_stop(proc, timeout_s=float(args.timeout_s))
            _ = stopped
        released, _release_ms = _wait_released(worker_pid)

        metrics_python = args.metrics_python.expanduser().resolve()
        if not metrics_python.is_file():
            raise RuntimeError(f"Shared isolated image-quality metrics interpreter is missing: {metrics_python}")
        quality_fingerprint = hashlib.sha256(
            f"{presentation.dataset_fingerprint_sha256}|{val_file}|{presentation.artifact_id}".encode("utf-8")
        ).hexdigest()
        render_manifest = out / "heldout_render_manifest.json"
        render_manifest.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "backend_kind": "nerfstudio_nerfacto_external",
                    "benchmark_fingerprint_sha256": quality_fingerprint,
                    "dataset_fingerprint_sha256": presentation.dataset_fingerprint_sha256,
                    "validation_frame_ids": [str(frame.get("eia_frame_id", val_file))],
                    "renders": [
                        {
                            "frame_id": str(frame.get("eia_frame_id", val_file)),
                            "reference_image_path": str(reference),
                            "rendered_image_path": str(rendered),
                            "width": width,
                            "height": height,
                        }
                    ],
                    "canonical_geometry_authoritative": False,
                    "certified_metrology": False,
                },
                indent=2,
                sort_keys=True,
            ) + "\n",
            encoding="utf-8",
        )
        metrics_payload = _run_metrics(
            metrics_python,
            render_manifest,
            out / "quality_metrics.json",
            timeout_s=float(args.timeout_s),
        )
        aggregate = dict(metrics_payload.get("aggregate", {}))
        metrics = {
            "psnr_db": float(aggregate.get("psnr_db", float("nan"))),
            "ssim": float(aggregate.get("ssim", float("nan"))),
            "lpips": float(aggregate.get("lpips", float("nan"))),
        }
        finite = all(math.isfinite(value) for value in metrics.values())
        contact = out / "heldout_contact_sheet.png"
        _contact_sheet(reference, rendered, metrics, contact)
        heavy_after = list(current_process_nerf_heavy_modules())
        passed = (
            finite
            and metrics["psnr_db"] >= 0.0
            and 0.0 <= metrics["ssim"] <= 1.0
            and metrics["lpips"] >= 0.0
            and graceful
            and released
            and not heavy_before
            and not heavy_after
            and bool(scene.canonical_geometry_authoritative)
        )
        summary = {
            "phase": "9.5H",
            "status": "PASS" if passed else "FAIL",
            "scope": "single_view_heldout_observational_quality",
            "scene_manifest": str(scene_manifest),
            "scene_id": scene.scene_id,
            "dataset_fingerprint_sha256": presentation.dataset_fingerprint_sha256,
            "training_frame_count": len(train_files),
            "validation_frame_count": len(val_files),
            "heldout_file": val_file,
            "render_width": width,
            "render_height": height,
            "observed_render_ms": float(frame_result.get("render_ms", 0.0) or 0.0),
            "metrics": metrics,
            "metrics_evaluator": metrics_payload.get("evaluator", {}),
            "quality_characterized": True,
            "visual_quality_evaluated": True,
            "quality_certified": False,
            "production_quality_certified": False,
            "phase8_e3_sanity_policy_applicable": False,
            "limitation_reason": "insufficient_views_for_phase8_e3_sanity_policy",
            "phase8_e3_policy_requirement": ">=8 training views and >=3 held-out views",
            "reference_path": str(reference),
            "rendered_path": str(rendered),
            "contact_sheet_path": str(contact),
            "worker_pid": worker_pid,
            "worker_shutdown_graceful": graceful,
            "worker_released": released,
            "canonical_geometry_authoritative": True,
            "picking_source": "canonical_geometry_only",
            "nerf_scene_role": "appearance_only",
            "nerf_canonical_geometry_authoritative": False,
            "nerf_metric_scale_authoritative": False,
            "nerf_measurement_authoritative": False,
            "nerf_qc_authoritative": False,
            "certified_metrology": False,
            "production_heavy_modules_before": heavy_before,
            "production_heavy_modules_after": heavy_after,
        }
    except Exception as exc:
        if proc is not None and proc.poll() is None:
            try:
                _graceful_stop(proc, timeout_s=10.0)
            except Exception:
                pass
        summary = {
            "phase": "9.5H",
            "status": "FAIL",
            "scope": "single_view_heldout_observational_quality",
            "error": f"{type(exc).__name__}: {exc}",
            "quality_characterized": False,
            "quality_certified": False,
            "production_quality_certified": False,
            "canonical_geometry_authoritative": True,
            "nerf_scene_role": "appearance_only",
            "certified_metrology": False,
        }

    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))
    print(f"Phase 9.5H held-out quality summary: {summary_path}")
    return 0 if summary.get("status") == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
