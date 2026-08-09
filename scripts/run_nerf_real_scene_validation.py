#!/usr/bin/env python3
"""Run Phase 8.7E3 real-scene held-out Nerfacto quality validation."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.reconstruction import (  # noqa: E402
    NERFSTUDIO_E2_CUDA,
    NERFSTUDIO_E2_RTX2060_COMPUTE_CAPABILITY,
    NERFSTUDIO_E2_TORCH,
    NERFSTUDIO_METHOD,
    NERFSTUDIO_RELEASE_REVISION_PREFIX,
    NERFSTUDIO_RELEASE_TAG,
    TINY_CUDA_NN_E2_REVISION,
    NerfBackendKind,
    NerfHeldOutQualityEvaluation,
    NerfOptimizationProfile,
    NerfQualitySanityPolicy,
    NerfRealSceneValidationSpec,
    NerfSceneArtifact,
    neural_scene_dataset_fingerprint,
    probe_nerf_environment,
    read_nerf_held_out_render_manifest,
    read_neural_scene_dataset_manifest,
    write_nerf_held_out_quality_evaluation,
    write_nerf_real_scene_validation_spec,
)

DEFAULT_OUTPUT = Path("/tmp/phase8_7e3_nerf_real_scene")
RTX2060_MIN_CUDA_VISIBLE_VRAM_MIB = 5500.0


def _default_env_python(name: str) -> Path:
    return Path.home() / "anaconda3" / "envs" / name / "bin" / "python"


def _git_revision(path: Path) -> str:
    completed = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
        timeout=10,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"could not read Git revision for {path}: {completed.stderr.strip()}")
    return completed.stdout.strip().lower()


def _version_base(value: str) -> str:
    return str(value).split("+")[0]


@contextmanager
def _isolated_environment(python: Path):
    prefix = python.resolve().parent.parent
    keys = ("PATH", "LD_LIBRARY_PATH", "LIBRARY_PATH", "CUDA_HOME", "CUDACXX", "CONDA_PREFIX", "CONDA_DEFAULT_ENV", "PYTHONNOUSERSITE")
    previous = {key: os.environ.get(key) for key in keys}

    def prepend(key: str, paths: list[Path]) -> None:
        values = [str(path) for path in paths if path.exists()]
        old = os.environ.get(key, "")
        if old:
            values.append(old)
        if values:
            os.environ[key] = os.pathsep.join(values)

    os.environ["CONDA_PREFIX"] = str(prefix)
    os.environ["CONDA_DEFAULT_ENV"] = prefix.name
    os.environ["PYTHONNOUSERSITE"] = "1"
    prepend("PATH", [prefix / "bin"])
    nvcc = prefix / "bin" / "nvcc"
    if nvcc.is_file():
        os.environ["CUDA_HOME"] = str(prefix)
        os.environ["CUDACXX"] = str(nvcc)
    libs = [prefix / "lib", prefix / "lib64", prefix / "targets" / "x86_64-linux" / "lib"]
    prepend("LD_LIBRARY_PATH", libs)
    prepend("LIBRARY_PATH", libs)
    try:
        yield prefix
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _load_split(path: Path, dataset_fingerprint: str) -> tuple[tuple[str, ...], tuple[str, ...], dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("phase") != "8.7E3":
        raise RuntimeError("split manifest is not a Phase 8.7E3 preparation manifest")
    if str(payload.get("dataset_fingerprint_sha256", "")) != dataset_fingerprint:
        raise RuntimeError("split manifest dataset fingerprint mismatch")
    training = tuple(str(value) for value in payload.get("training_frame_ids", ()))
    validation = tuple(str(value) for value in payload.get("validation_frame_ids", ()))
    if not training or not validation:
        raise RuntimeError("split manifest does not contain train/held-out views")
    return training, validation, payload


def _preflight(*, python: Path, nerfstudio_root: Path, tcnn_root: Path) -> dict:
    probe = probe_nerf_environment(python, timeout_s=60.0)
    reasons: list[str] = []
    if not probe.interpreter_is_isolated:
        reasons.append("interpreter_not_isolated")
    if not probe.backend_ready:
        reasons.append("backend_environment_not_ready")
    if probe.nerfstudio_version != NERFSTUDIO_RELEASE_TAG.removeprefix("v"):
        reasons.append("nerfstudio_version_mismatch")
    if _version_base(probe.torch_version) != NERFSTUDIO_E2_TORCH:
        reasons.append("torch_version_mismatch")
    if not probe.torch_cuda_version.startswith(NERFSTUDIO_E2_CUDA):
        reasons.append("torch_cuda_version_mismatch")
    if not _git_revision(nerfstudio_root).startswith(NERFSTUDIO_RELEASE_REVISION_PREFIX):
        reasons.append("nerfstudio_revision_mismatch")
    if _git_revision(tcnn_root) != TINY_CUDA_NN_E2_REVISION:
        reasons.append("tiny_cuda_nn_revision_mismatch")
    if not probe.cuda_device_names or "RTX 2060" not in probe.cuda_device_names[0]:
        reasons.append("rtx2060_gpu_name_mismatch")
    if not probe.cuda_compute_capabilities or probe.cuda_compute_capabilities[0] != NERFSTUDIO_E2_RTX2060_COMPUTE_CAPABILITY:
        reasons.append("rtx2060_compute_capability_mismatch")
    if not probe.cuda_total_vram_mib or probe.cuda_total_vram_mib[0] < RTX2060_MIN_CUDA_VISIBLE_VRAM_MIB:
        reasons.append("rtx2060_cuda_visible_vram_below_profile")
    return {"probe": probe.as_dict(), "reasons": reasons}


def _run_worker(
    *,
    python: Path,
    request: dict,
    output: Path,
    nerfstudio_root: Path,
    tcnn_root: Path,
    timeout_s: float,
) -> dict:
    request_path = output / "worker_request.json"
    result_path = output / "worker_result.json"
    request_path.write_text(json.dumps(request, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    worker = ROOT / "scripts" / "nerf_workers" / "nerfstudio_e3_worker.py"
    old_ns = os.environ.get("EIA_NERFSTUDIO_ROOT")
    old_tcnn = os.environ.get("EIA_TCNN_ROOT")
    os.environ["EIA_NERFSTUDIO_ROOT"] = str(nerfstudio_root)
    os.environ["EIA_TCNN_ROOT"] = str(tcnn_root)
    try:
        completed = subprocess.run(
            [str(python), str(worker), "--request", str(request_path), "--result", str(result_path)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=float(timeout_s),
            check=False,
        )
    finally:
        if old_ns is None:
            os.environ.pop("EIA_NERFSTUDIO_ROOT", None)
        else:
            os.environ["EIA_NERFSTUDIO_ROOT"] = old_ns
        if old_tcnn is None:
            os.environ.pop("EIA_TCNN_ROOT", None)
        else:
            os.environ["EIA_TCNN_ROOT"] = old_tcnn
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip() or "unknown E3 worker failure"
        raise RuntimeError(f"E3 isolated Nerfstudio worker failed: {detail}")
    if not result_path.is_file():
        raise RuntimeError("E3 worker did not produce result JSON")
    return json.loads(result_path.read_text(encoding="utf-8"))


def _run_metrics(*, metrics_python: Path, render_manifest: Path, result: Path, device: str, timeout_s: float) -> dict:
    worker = ROOT / "scripts" / "gaussian_benchmark_workers" / "metrics_worker.py"
    completed = subprocess.run(
        [str(metrics_python), str(worker), "--render-manifest", str(render_manifest), "--result", str(result), "--device", str(device)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=float(timeout_s),
        check=False,
    )
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip() or "unknown metric worker failure"
        raise RuntimeError(f"shared held-out metrics worker failed: {detail}")
    if not result.is_file():
        raise RuntimeError("shared held-out metrics worker did not produce JSON")
    payload = json.loads(result.read_text(encoding="utf-8"))
    payload["validation_fingerprint_sha256"] = payload.get("benchmark_fingerprint_sha256", "")
    return payload


def _contact_sheet(render_manifest, quality: NerfHeldOutQualityEvaluation, path: Path) -> Path:
    metrics = {value.frame_id: value for value in quality.views}
    rows: list[Image.Image] = []
    target_width = 360
    for item in render_manifest.renders:
        with Image.open(item.reference_image_path) as ref_img:
            ref = ref_img.convert("RGB")
        with Image.open(item.rendered_image_path) as pred_img:
            pred = pred_img.convert("RGB")
        scale = target_width / float(ref.width)
        size = (target_width, max(64, int(round(ref.height * scale))))
        ref = ref.resize(size)
        pred = pred.resize(size)
        label_h = 42
        row = Image.new("RGB", (target_width * 2, size[1] + label_h), "white")
        row.paste(ref, (0, label_h))
        row.paste(pred, (target_width, label_h))
        draw = ImageDraw.Draw(row)
        m = metrics[item.frame_id]
        draw.text((6, 4), f"REF {item.frame_id}", fill="black")
        draw.text((target_width + 6, 4), f"NERFACTO  PSNR {m.psnr_db:.2f}  SSIM {m.ssim:.3f}  LPIPS {m.lpips:.3f}", fill="black")
        rows.append(row)
    sheet = Image.new("RGB", (target_width * 2, sum(row.height for row in rows)), "white")
    y = 0
    for row in rows:
        sheet.paste(row, (0, y))
        y += row.height
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path)
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-manifest", required=True, type=Path)
    parser.add_argument("--split-manifest", required=True, type=Path)
    parser.add_argument("--source-label", default="nerfstudio-poster-real-scene")
    parser.add_argument("--profile", choices=[value.value for value in NerfOptimizationProfile], default="efficient")
    parser.add_argument("--training-steps", type=int, default=1000)
    parser.add_argument("--nerf-python", type=Path, default=_default_env_python("eia-nerf-nerfstudio"))
    parser.add_argument("--metrics-python", type=Path, default=_default_env_python("eia-gaussian-metrics"))
    parser.add_argument("--nerfstudio-root", type=Path, default=Path.home() / "Desktop/eia-nerf-backends/nerfstudio")
    parser.add_argument("--tiny-cuda-nn-root", type=Path, default=Path.home() / "Desktop/eia-nerf-backends/tiny-cuda-nn")
    parser.add_argument("--metrics-device", choices=("cpu", "cuda", "auto"), default="cpu")
    parser.add_argument("--timeout-s", type=float, default=3600.0)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    output = args.output_dir.expanduser().resolve()
    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)
    summary_path = output / "summary.json"

    summary = {
        "phase": "8.7E3",
        "pass": False,
        "backend_kind": NerfBackendKind.NERFSTUDIO_NERFACTO_EXTERNAL.value,
        "method_name": NERFSTUDIO_METHOD,
        "validation_scope": "real_scene_heldout_quality",
        "performance_hardening_deferred_to": "8.7E4",
        "scene_role": "appearance_only",
        "canonical_geometry_authoritative": False,
        "certified_metrology": False,
        "reconstruction_ui_enabled": False,
    }
    try:
        dataset_path = args.dataset_manifest.expanduser().resolve()
        split_path = args.split_manifest.expanduser().resolve()
        dataset = read_neural_scene_dataset_manifest(dataset_path)
        dataset_fingerprint = neural_scene_dataset_fingerprint(dataset)
        training_ids, validation_ids, split_payload = _load_split(split_path, dataset_fingerprint)
        spec = NerfRealSceneValidationSpec(
            validation_id="phase8-7e3-real-scene",
            source_label=args.source_label,
            dataset=dataset,
            training_frame_ids=training_ids,
            validation_frame_ids=validation_ids,
            optimization_profile=NerfOptimizationProfile(args.profile),
            training_steps=args.training_steps,
        )
        spec_path = write_nerf_real_scene_validation_spec(spec, output / "validation_spec.json")
        summary["spec_path"] = str(spec_path)
        summary["split_preparation"] = split_payload.get("preparation", {})

        python = args.nerf_python.expanduser().resolve()
        nerfstudio_root = args.nerfstudio_root.expanduser().resolve()
        tcnn_root = args.tiny_cuda_nn_root.expanduser().resolve()
        with _isolated_environment(python):
            preflight = _preflight(python=python, nerfstudio_root=nerfstudio_root, tcnn_root=tcnn_root)
            summary["environment_probe"] = preflight["probe"]
            summary["preflight_reasons"] = preflight["reasons"]
            if preflight["reasons"]:
                raise RuntimeError("E3 preflight failed: " + ",".join(preflight["reasons"]))
            request = {
                "phase": "8.7E3",
                "request_id": "phase8-7e3-real-nerfacto",
                "backend_kind": NerfBackendKind.NERFSTUDIO_NERFACTO_EXTERNAL.value,
                "method_name": NERFSTUDIO_METHOD,
                "profile": spec.optimization_profile.value,
                "training_steps": spec.training_steps,
                "dataset_manifest_path": str(dataset_path),
                "dataset_fingerprint_sha256": spec.dataset_fingerprint_sha256,
                "validation_fingerprint_sha256": spec.validation_fingerprint_sha256,
                "training_frame_ids": list(spec.training_frame_ids),
                "validation_frame_ids": list(spec.validation_frame_ids),
                "output_dir": str(output / "worker-output"),
            }
            worker_result = _run_worker(
                python=python,
                request=request,
                output=output,
                nerfstudio_root=nerfstudio_root,
                tcnn_root=tcnn_root,
                timeout_s=args.timeout_s,
            )

        artifact = NerfSceneArtifact.from_payload(worker_result["artifact"])
        if artifact.dataset_fingerprint_sha256 != spec.dataset_fingerprint_sha256:
            raise RuntimeError("E3 worker artifact dataset fingerprint mismatch")
        render_manifest_path = Path(worker_result["render_manifest_path"]).resolve()
        render_manifest = read_nerf_held_out_render_manifest(render_manifest_path)
        if render_manifest.validation_fingerprint_sha256 != spec.validation_fingerprint_sha256:
            raise RuntimeError("E3 held-out render validation fingerprint mismatch")

        metrics_python = args.metrics_python.expanduser().resolve()
        if not metrics_python.is_file():
            raise RuntimeError(f"shared D3 metrics interpreter is missing: {metrics_python}")
        raw_metrics_path = output / "quality_raw.json"
        metrics_payload = _run_metrics(
            metrics_python=metrics_python,
            render_manifest=render_manifest_path,
            result=raw_metrics_path,
            device=args.metrics_device,
            timeout_s=args.timeout_s,
        )
        quality = NerfHeldOutQualityEvaluation.from_metrics_worker_payload(metrics_payload)
        if quality.validation_fingerprint_sha256 != spec.validation_fingerprint_sha256:
            raise RuntimeError("E3 metric fingerprint mismatch")
        if quality.dataset_fingerprint_sha256 != spec.dataset_fingerprint_sha256:
            raise RuntimeError("E3 metric dataset fingerprint mismatch")
        quality_path = write_nerf_held_out_quality_evaluation(quality, output / "quality_evaluation.json")
        quality_pass = quality.passes(spec.quality_policy)
        contact_sheet = _contact_sheet(render_manifest, quality, output / "heldout_contact_sheet.png")
        if not quality_pass:
            raise RuntimeError(
                "E3 nondegenerate quality sanity gate failed: "
                f"psnr={quality.aggregate.psnr_db:.4f},ssim={quality.aggregate.ssim:.6f},lpips={quality.aggregate.lpips:.6f}"
            )

        runtime_evidence_path = Path(worker_result["runtime_evidence_path"]).resolve()
        runtime_evidence = json.loads(runtime_evidence_path.read_text(encoding="utf-8"))
        summary.update(
            {
                "pass": True,
                "real_scene_validation_passed": True,
                "quality_evidence_complete": True,
                "quality_sanity_pass": True,
                "validation_fingerprint_sha256": spec.validation_fingerprint_sha256,
                "dataset_fingerprint_sha256": spec.dataset_fingerprint_sha256,
                "training_frame_count": len(spec.training_frame_ids),
                "validation_frame_count": len(spec.validation_frame_ids),
                "training_steps": spec.training_steps,
                "quality_policy": spec.quality_policy.as_dict(),
                "quality": quality.as_dict(),
                "artifact": artifact.as_dict(),
                "runtime_evidence": runtime_evidence,
                "render_manifest_path": str(render_manifest_path),
                "quality_evaluation_path": str(quality_path),
                "contact_sheet_path": str(contact_sheet),
                "shared_metrics_worker": "phase8_7d3_backend_neutral_psnr_ssim_lpips",
                "shared_metrics_environment": str(metrics_python),
            }
        )
    except Exception as exc:
        summary.update({"pass": False, "real_scene_validation_passed": False, "failure": f"{type(exc).__name__}: {exc}"})
        summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print("Phase 8.7E3 real-scene Nerfacto validation: FAIL")
        print(summary["failure"])
        print(f"summary={summary_path}")
        return 1

    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    quality = summary["quality"]["aggregate"]
    print("Phase 8.7E3 real-scene Nerfacto validation: PASS")
    print(
        f"frames=train:{summary['training_frame_count']} heldout:{summary['validation_frame_count']} "
        f"steps={summary['training_steps']}"
    )
    print(f"PSNR={quality['psnr_db']:.4f} SSIM={quality['ssim']:.6f} LPIPS={quality['lpips']:.6f}")
    print("quality_evidence_complete=True quality_sanity_pass=True")
    print("scene_role=appearance_only canonical_geometry_authoritative=False certified_metrology=False")
    print("performance_hardening_deferred_to=8.7E4 reconstruction_ui_enabled=False")
    print(f"contact_sheet={summary['contact_sheet_path']}")
    print(f"summary={summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
