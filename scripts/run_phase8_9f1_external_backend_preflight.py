#!/usr/bin/env python3
"""Phase 8.9F1 read-only external-backend + dataset preflight.

This probe never trains or runs reconstruction inference.  It inspects the
production repository, isolated runtime probes, external checkout revisions and
known compatibility markers, dataset availability, disk/GPU headroom, and the
existing source contracts needed by the later 8.9F real-scene campaign.
"""
from __future__ import annotations

import argparse
import ast
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.reconstruction import (  # noqa: E402
    GRAPHDECO_QUALIFIED_REVISION,
    GSPLAT_QUALIFIED_VERSION,
    GaussianBackendKind,
    NERFSTUDIO_RELEASE_REVISION_PREFIX,
    TINY_CUDA_NN_E2_REVISION,
    graphdeco_checkout_revision,
    probe_da3_environment,
    probe_gaussian_environment,
    probe_gaussian_runtime_evidence,
    probe_nerf_environment,
)

PHASE = "8.9F1"
EXPECTED_HEAD = "9debb17936abd1f8ac3b64fbbdb1d0dcdf61fe7e"
EXPECTED_BRANCH = "phase8-reconstruction"
EXPECTED_PYCOLMAP_VERSION = "4.1.1"
EXPECTED_DA3_REVISION = "3d835ec1a5802d64a8b8b15f817a1ab54809bfe4"
EXPECTED_GSPLAT_REVISION = "937e29912570c372bed6747a5c9bf85fed877bae"
EXPECTED_NERFSTUDIO_REVISION = "6b60855003011b2ca23c2fe3f8e2ca6314c69924"

F1_REPO_FILES = {
    "docs/phase_history/PHASE8_9F1_VALIDATION.md",
    "scripts/run_phase8_9f1_external_backend_preflight.py",
    "scripts/run_phase8_9f1_tests.sh",
    "tests/integration/test_phase8_9f1_external_backend_contract.py",
    "tests/unit/test_phase8_9f1_external_backend_preflight.py",
}

HEAVY_MODULES = (
    "pycolmap",
    "depth_anything_3",
    "gsplat",
    "nerfstudio",
    "tinycudann",
    "diff_gaussian_rasterization",
    "simple_knn",
)

DEFAULT_PATHS = {
    "pycolmap_python": Path.home() / "anaconda3/envs/depth3d-pycolmap-probe/bin/python",
    "da3_python": Path.home() / "anaconda3/envs/depth3d-da3/bin/python",
    "da3_root": Path.home() / "Desktop/depth3d-da3-backends/depth-anything-3",
    "gsplat_python": Path.home() / "anaconda3/envs/depth3d-gaussian-gsplat/bin/python",
    "gsplat_root": Path.home() / "Desktop/depth3d-gaussian-backends/gsplat",
    "graphdeco_python": Path.home() / "anaconda3/envs/depth3d-gaussian-graphdeco/bin/python",
    "graphdeco_root": Path.home() / "Desktop/depth3d-gaussian-backends/gaussian-splatting",
    "metrics_python": Path.home() / "anaconda3/envs/depth3d-gaussian-metrics/bin/python",
    "nerf_python": Path.home() / "anaconda3/envs/depth3d-nerf-nerfstudio/bin/python",
    "nerfstudio_root": Path.home() / "Desktop/depth3d-nerf-backends/nerfstudio",
    "tcnn_root": Path.home() / "Desktop/depth3d-nerf-backends/tiny-cuda-nn",
    "south_building": Path.home() / "Desktop/phase8_3c_south_building/south-building/images",
}


def _run(command: list[str], *, timeout: float = 30.0, cwd: Path | None = None) -> dict[str, Any]:
    try:
        completed = subprocess.run(
            command,
            cwd=None if cwd is None else str(cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"ok": False, "returncode": None, "stdout": "", "stderr": f"{type(exc).__name__}: {exc}"}
    return {
        "ok": completed.returncode == 0,
        "returncode": completed.returncode,
        "stdout": completed.stdout.strip(),
        "stderr": completed.stderr.strip(),
    }


def _git_revision(path: Path) -> str:
    if not (path / ".git").exists():
        return ""
    result = _run(["git", "-C", str(path), "rev-parse", "HEAD"], timeout=10.0)
    return result["stdout"].splitlines()[-1].strip().lower() if result["ok"] and result["stdout"] else ""


def _git_status(path: Path) -> list[str]:
    if not (path / ".git").exists():
        return []
    result = _run(["git", "-C", str(path), "status", "--short", "--untracked-files=all"], timeout=15.0)
    return [line.rstrip() for line in result["stdout"].splitlines()] if result["ok"] else [f"<git-status-error> {result['stderr']}"]


def _repository_state(repo: Path, expected_head: str) -> dict[str, Any]:
    branch = _run(["git", "branch", "--show-current"], cwd=repo)["stdout"].strip()
    head = _run(["git", "rev-parse", "HEAD"], cwd=repo)["stdout"].strip()
    origin = _run(["git", "rev-parse", "origin/phase8-reconstruction"], cwd=repo)["stdout"].strip()
    status = _git_status(repo)
    unexpected = []
    for line in status:
        path = line[3:].strip() if len(line) >= 4 else line.strip()
        if path not in F1_REPO_FILES:
            unexpected.append(line)
    return {
        "branch": branch,
        "head": head,
        "origin_head": origin,
        "expected_head": expected_head,
        "head_matches_expected": not expected_head or head == expected_head,
        "head_matches_origin": bool(head and head == origin),
        "unexpected_worktree_changes": unexpected,
        "f1_overlay_files_present": sorted(line[3:].strip() for line in status if len(line) >= 4 and line[3:].strip() in F1_REPO_FILES),
        "pass": bool(
            branch == EXPECTED_BRANCH
            and (not expected_head or head == expected_head)
            and head == origin
            and not unexpected
        ),
    }


def _direct_heavy_imports(repo: Path) -> list[dict[str, str]]:
    findings: list[dict[str, str]] = []
    core = repo / "core" / "reconstruction"
    for path in sorted(core.rglob("*.py")):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except (OSError, SyntaxError):
            continue
        for node in ast.walk(tree):
            names: list[str] = []
            if isinstance(node, ast.Import):
                names.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                names.append(node.module)
            for name in names:
                top = name.split(".", 1)[0]
                if top in HEAVY_MODULES:
                    findings.append({"file": str(path.relative_to(repo)), "module": name})
    return findings


def _production_environment() -> dict[str, Any]:
    import torch

    importable = {name: importlib.util.find_spec(name) is not None for name in HEAVY_MODULES}
    gpu: dict[str, Any] = {
        "cuda_available": bool(torch.cuda.is_available()),
        "torch_version": str(torch.__version__),
        "torch_cuda_version": str(torch.version.cuda or ""),
        "device_count": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
        "devices": [],
    }
    if torch.cuda.is_available():
        for index in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(index)
            gpu["devices"].append({
                "index": index,
                "name": str(torch.cuda.get_device_name(index)),
                "compute_capability": list(torch.cuda.get_device_capability(index)),
                "total_vram_mib": float(props.total_memory / (1024 ** 2)),
            })
    return {
        "python_executable": str(Path(sys.executable).resolve()),
        "conda_prefix": os.environ.get("CONDA_PREFIX", ""),
        "conda_default_env": os.environ.get("CONDA_DEFAULT_ENV", ""),
        "heavy_module_importability": importable,
        "heavy_runtime_leak_into_depth": any(importable.values()),
        "gpu": gpu,
    }


def _disk(path: Path) -> dict[str, Any]:
    target = path if path.exists() else path.parent
    usage = shutil.disk_usage(target)
    return {
        "path": str(target.resolve()),
        "total_gib": usage.total / (1024 ** 3),
        "used_gib": usage.used / (1024 ** 3),
        "free_gib": usage.free / (1024 ** 3),
    }


def _dataset_inventory(path: Path) -> dict[str, Any]:
    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    files = sorted(p for p in path.iterdir() if p.is_file() and p.suffix.lower() in extensions) if path.is_dir() else []
    by_ext: dict[str, int] = {}
    for item in files:
        by_ext[item.suffix.lower()] = by_ext.get(item.suffix.lower(), 0) + 1
    return {
        "path": str(path),
        "exists": path.is_dir(),
        "image_count": len(files),
        "extensions": by_ext,
        "first_images": [str(item) for item in files[:5]],
        "sfm_ready": len(files) >= 8,
        "da3_ready": len(files) >= 2,
    }


def _find_nerfstudio_sources() -> list[str]:
    roots = [Path.home() / "Desktop" / "datasets", Path.home() / "Desktop"]
    found: list[str] = []
    for root in roots:
        if not root.is_dir():
            continue
        # Keep this bounded: common datasets root is recursive; broader Desktop is only two levels.
        if root.name == "datasets":
            candidates = root.rglob("transforms.json")
        else:
            candidates = (p for p in root.glob("*/*/transforms.json"))
        for path in candidates:
            try:
                found.append(str(path.parent.resolve()))
            except OSError:
                continue
            if len(found) >= 20:
                break
        if len(found) >= 20:
            break
    return sorted(set(found))


def _python_json(python: Path, code: str) -> dict[str, Any]:
    if not python.is_file():
        return {"available": False, "error": "python_executable_missing", "python_executable": str(python)}
    result = _run([str(python), "-c", code], timeout=60.0)
    if not result["ok"]:
        return {"available": False, "error": result["stderr"] or result["stdout"] or "probe_failed", "python_executable": str(python)}
    try:
        payload = json.loads(result["stdout"].splitlines()[-1])
    except (IndexError, json.JSONDecodeError) as exc:
        return {"available": False, "error": f"invalid_probe_json:{exc}", "python_executable": str(python)}
    payload["python_executable"] = str(python)
    return payload


def _pycolmap(paths: dict[str, Path]) -> dict[str, Any]:
    code = (
        "import json,platform,pycolmap; "
        "print(json.dumps({'available':True,'python_version':platform.python_version(),"
        "'pycolmap_version':str(getattr(pycolmap,'__version__',''))},sort_keys=True))"
    )
    payload = _python_json(paths["pycolmap_python"], code)
    payload["qualified_version_match"] = payload.get("pycolmap_version") == EXPECTED_PYCOLMAP_VERSION
    payload["ready"] = bool(payload.get("available") and payload["qualified_version_match"])
    return payload


def _checkout(path: Path, *, expected_revision: str = "", allow_status_prefixes: tuple[str, ...] = ()) -> dict[str, Any]:
    revision = _git_revision(path)
    status = _git_status(path)
    # ``_run`` trims leading whitespace from the first stdout line. Git's short
    # porcelain format uses that leading column to distinguish index/worktree
    # state, so compare allowed entries after removing only leading whitespace
    # from both sides. This keeps deliberate compatibility patches accepted
    # without broadly accepting unrelated checkout drift.
    normalized_allowed = tuple(prefix.lstrip() for prefix in allow_status_prefixes)
    unexpected = [
        line
        for line in status
        if not any(line.lstrip().startswith(prefix) for prefix in normalized_allowed)
    ]
    return {
        "path": str(path),
        "exists": path.is_dir(),
        "revision": revision,
        "expected_revision": expected_revision,
        "revision_match": not expected_revision or revision == expected_revision,
        "git_status": status,
        "unexpected_status": unexpected,
    }


def _external_backends(paths: dict[str, Path]) -> dict[str, Any]:
    pycolmap = _pycolmap(paths)

    da3_probe = probe_da3_environment(paths["da3_python"], paths["da3_root"])
    da3_checkout = _checkout(
        paths["da3_root"],
        expected_revision=EXPECTED_DA3_REVISION,
        allow_status_prefixes=(" M src/depth_anything_3/api.py", "?? .depth3d_da3_runtime_lock.json"),
    )
    da3_ready = bool(
        da3_probe.available
        and da3_checkout["revision_match"]
        and not da3_checkout["unexpected_status"]
    )

    gsplat_probe = probe_gaussian_environment(paths["gsplat_python"], backend_kind=GaussianBackendKind.GSPLAT_EXTERNAL)
    gsplat_runtime = probe_gaussian_runtime_evidence(paths["gsplat_python"])
    gsplat_checkout = _checkout(paths["gsplat_root"], expected_revision=EXPECTED_GSPLAT_REVISION)
    gsplat_ready = bool(
        gsplat_probe.backend_ready
        and gsplat_probe.interpreter_is_isolated
        and gsplat_probe.gsplat_version.split("+", 1)[0] == GSPLAT_QUALIFIED_VERSION
        and not gsplat_runtime.error
        and gsplat_checkout["revision_match"]
        and not gsplat_checkout["unexpected_status"]
    )

    graphdeco_probe = probe_gaussian_environment(
        paths["graphdeco_python"],
        backend_kind=GaussianBackendKind.GRAPHDECO_REFERENCE_EXTERNAL,
        graphdeco_root=paths["graphdeco_root"],
    )
    graphdeco_runtime = probe_gaussian_runtime_evidence(paths["graphdeco_python"])
    graphdeco_checkout = _checkout(
        paths["graphdeco_root"],
        expected_revision=GRAPHDECO_QUALIFIED_REVISION,
        allow_status_prefixes=(" M scene/__init__.py", " m submodules/diff-gaussian-rasterization", " ? submodules/simple-knn", " M submodules/diff-gaussian-rasterization", "?? submodules/simple-knn"),
    )
    split_marker = paths["graphdeco_root"] / "scene" / "__init__.py"
    cstdint_marker = paths["graphdeco_root"] / "submodules" / "diff-gaussian-rasterization" / "cuda_rasterizer" / "rasterizer_impl.h"
    split_patch = split_marker.is_file() and "llffhold=0" in split_marker.read_text(encoding="utf-8", errors="replace")
    cstdint_patch = cstdint_marker.is_file() and "#include <cstdint>" in cstdint_marker.read_text(encoding="utf-8", errors="replace")
    graphdeco_ready = bool(
        graphdeco_probe.backend_ready
        and graphdeco_probe.interpreter_is_isolated
        and not graphdeco_runtime.error
        and graphdeco_checkout["revision_match"]
        and split_patch
        and cstdint_patch
        and not graphdeco_checkout["unexpected_status"]
    )

    nerf_probe = probe_nerf_environment(paths["nerf_python"], timeout_s=60.0)
    nerfstudio_checkout = _checkout(paths["nerfstudio_root"], expected_revision=EXPECTED_NERFSTUDIO_REVISION)
    tcnn_checkout = _checkout(paths["tcnn_root"], expected_revision=TINY_CUDA_NN_E2_REVISION)
    nerf_ready = bool(
        nerf_probe.backend_ready
        and nerf_probe.interpreter_is_isolated
        and nerfstudio_checkout["revision"].startswith(NERFSTUDIO_RELEASE_REVISION_PREFIX)
        and tcnn_checkout["revision_match"]
        and not nerfstudio_checkout["unexpected_status"]
        and not tcnn_checkout["unexpected_status"]
    )

    metrics_code = (
        "import json,platform; "
        "mods={}; "
        "import importlib.util as u; "
        "[mods.__setitem__(n,u.find_spec(n) is not None) for n in ('torch','lpips')]; "
        "print(json.dumps({'available':True,'python_version':platform.python_version(),'modules':mods},sort_keys=True))"
    )
    metrics = _python_json(paths["metrics_python"], metrics_code)

    return {
        "pycolmap": pycolmap,
        "da3": {"probe": da3_probe.as_dict(), "checkout": da3_checkout, "ready": da3_ready},
        "gaussian": {
            "gsplat": {"probe": gsplat_probe.as_dict(), "runtime": gsplat_runtime.as_dict(), "checkout": gsplat_checkout, "ready": gsplat_ready},
            "graphdeco": {
                "probe": graphdeco_probe.as_dict(),
                "runtime": graphdeco_runtime.as_dict(),
                "checkout": graphdeco_checkout,
                "explicit_test_split_patch": split_patch,
                "cstdint_patch": cstdint_patch,
                "ready": graphdeco_ready,
                "usage_scope": "research_evaluation_only",
            },
            "metrics": metrics,
            "primary_release_candidate": "gsplat",
            "primary_ready": gsplat_ready,
        },
        "nerf": {
            "probe": nerf_probe.as_dict(),
            "nerfstudio_checkout": nerfstudio_checkout,
            "tiny_cuda_nn_checkout": tcnn_checkout,
            "ready": nerf_ready,
        },
    }


def _source_contracts(repo: Path) -> dict[str, Any]:
    da3_provider = (repo / "core/reconstruction/da3_provider.py").read_text(encoding="utf-8")
    da3_worker = (repo / "scripts/da3_workers/da3_worker.py").read_text(encoding="utf-8")
    canonical_bridge = (repo / "scripts/run_da3_canonical_geometry_integration.py").read_text(encoding="utf-8")
    f4 = (repo / "scripts/run_reconstruction_f4_real_backends.py").read_text(encoding="utf-8")
    appearance = (repo / "core/reconstruction/pipeline_appearance.py").read_text(encoding="utf-8")
    persistent = (repo / "core/reconstruction/persistent_scene.py").read_text(encoding="utf-8")
    switch = (repo / "app/reconstruction_representation_switching.py").read_text(encoding="utf-8")
    return {
        "da3_accepts_external_sfm_poses": "external_sfm" in da3_provider or "sfm_pose" in da3_provider,
        "da3_worker_generates_pose_output": "prediction.extrinsics" in da3_worker,
        "da3_supporting_geometry_authority": "SUPPORTING_GEOMETRY" in da3_provider,
        "da3_canonical_bridge_exists": "canonical_geometry_authoritative\": True" in canonical_bridge or "canonical_geometry_authoritative': True" in canonical_bridge,
        "appearance_dataset_materialized_from_ingestion_and_trajectory": "materialize_appearance_dataset" in appearance and "PipelineCameraTrajectoryEvidence" in appearance,
        "current_real_f4_script_uses_deterministic_fixture": "make_f3_fixture" in f4,
        "persistent_scene_tracks_file_references": "verify_persistent_scene_file_references" in persistent and "file_references" in persistent,
        "ui_switching_is_persisted_preview_only": "does not render neural representations" in switch and "canonical geometry remains authoritative" in switch,
    }


def _existing_evidence() -> dict[str, Any]:
    candidates = {
        "da3_real": Path("/tmp/phase8_7f2_da3_real/worker-output/result_manifest.json"),
        "da3_canonical": Path("/tmp/phase8_7f3_da3_canonical/summary.json"),
        "gaussian_qualification": Path("/tmp/phase8_7d2_gaussian_qualification/summary.json"),
        "nerf_real": Path("/tmp/phase8_7e3_nerf_real_scene/summary.json"),
        "f4_real": Path("/tmp/phase8_7f4_real_backends/summary.json"),
        "f5_persistent": Path("/tmp/phase8_7f5_real_persistence/summary.json"),
    }
    return {name: {"path": str(path), "exists": path.is_file()} for name, path in candidates.items()}


def _campaign_plan(contracts: dict[str, Any], dataset: dict[str, Any], backends: dict[str, Any]) -> dict[str, Any]:
    literal_pose_feed = bool(contracts["da3_accepts_external_sfm_poses"])
    return {
        "primary_dataset": dataset["path"],
        "dataset_image_count": dataset["image_count"],
        "literal_sfm_pose_to_da3_feed_supported": literal_pose_feed,
        "truthful_chain_interpretation": (
            "SfM and DA3 are separately real pose/geometry evidence paths; DA3 currently infers its own supporting poses. "
            "The later validation bridge may bind real DA3 evidence into canonical geometry and then feed the existing appearance executors, "
            "but must not claim that SfM poses were injected into DA3."
        ),
        "f2_geometry_candidate": {
            "sfm": "rerun/consume real South Building PyCOLMAP evidence",
            "da3": "DA3-SMALL, process_res=392, bounded real image subset on RTX 2060",
            "canonical": "existing run_da3_canonical_geometry_integration contract",
        },
        "f3_gaussian_candidate": {
            "backend": "gsplat",
            "reason": "qualified permissive external backend and safest Phase-8 release candidate",
            "ready": bool(backends.get("gaussian", {}).get("primary_ready", False)),
        },
        "f4_nerf_candidate": {
            "backend": "nerfstudio_nerfacto_external",
            "profile": "efficient",
            "bounded_training_steps": 16,
            "ready": bool(backends.get("nerf", {}).get("ready", False)),
        },
        "validation_only_bridge_required": bool(contracts["current_real_f4_script_uses_deterministic_fixture"]),
        "production_architecture_change_required": False,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("/tmp/phase8_9f1_preflight"))
    parser.add_argument("--expected-head", default=EXPECTED_HEAD)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_PATHS["south_building"])
    parser.add_argument("--static-only", action="store_true", help="Skip external environments/GPU; used for dependency-light validation.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    out = args.output_dir.expanduser().resolve()
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)

    repo_state = _repository_state(ROOT, str(args.expected_head).strip())
    contracts = _source_contracts(ROOT)
    imports = _direct_heavy_imports(ROOT)
    dataset = _dataset_inventory(args.dataset.expanduser().resolve())
    nerfstudio_sources = _find_nerfstudio_sources()

    if args.static_only:
        production = {"skipped": True}
        backends: dict[str, Any] = {"skipped": True}
        disks = {"repo": _disk(ROOT), "tmp": _disk(Path("/tmp"))}
        backend_ready = True
    else:
        production = _production_environment()
        backends = _external_backends(DEFAULT_PATHS)
        disks = {
            "repo": _disk(ROOT),
            "desktop": _disk(Path.home() / "Desktop"),
            "tmp": _disk(Path("/tmp")),
        }
        backend_ready = bool(
            backends["pycolmap"]["ready"]
            and backends["da3"]["ready"]
            and backends["gaussian"]["primary_ready"]
            and backends["nerf"]["ready"]
        )

    plan = _campaign_plan(contracts, dataset, backends if not args.static_only else {})
    blockers: list[str] = []
    warnings: list[str] = []
    if not repo_state["pass"] and not args.static_only:
        blockers.append("repository_baseline_not_exact")
    if imports:
        blockers.append("direct_heavy_import_in_core_reconstruction")
    if not dataset["sfm_ready"] or not dataset["da3_ready"]:
        blockers.append("primary_dataset_not_ready")
    if not args.static_only:
        if production.get("heavy_runtime_leak_into_depth"):
            blockers.append("heavy_runtime_importable_in_production_depth")
        if not backend_ready:
            blockers.append("one_or_more_required_external_backends_not_ready")
        if not backends["gaussian"]["graphdeco"]["ready"]:
            warnings.append("graphdeco_reference_backend_not_ready_or_checkout_state_differs; gsplat remains primary")
    if not contracts["da3_accepts_external_sfm_poses"]:
        warnings.append("DA3 does not accept SfM poses; do not claim literal SfM-pose-to-DA3 injection")
    if contracts["current_real_f4_script_uses_deterministic_fixture"]:
        warnings.append("existing F4 real-backend runner uses deterministic F3 fixture; later real-chain validation needs a validation-only bridge")
    if not nerfstudio_sources:
        warnings.append("no Nerfstudio transforms.json source discovered; South Building can still be used through pipeline-materialized appearance datasets")

    payload = {
        "phase": PHASE,
        "pass": not blockers,
        "mode": "static_only" if args.static_only else "full_read_only_preflight",
        "production_code_changed_by_8_9f1": False,
        "external_checkouts_modified_by_8_9f1": False,
        "heavy_training_or_inference_started": False,
        "external_import_probes_executed": not args.static_only,
        "repository": repo_state,
        "production_environment": production,
        "direct_heavy_imports_in_core_reconstruction": imports,
        "disk": disks,
        "primary_dataset": dataset,
        "nerfstudio_source_candidates": nerfstudio_sources,
        "external_backends": backends,
        "source_contracts": contracts,
        "existing_tmp_evidence": _existing_evidence(),
        "recommended_campaign": plan,
        "blockers": blockers,
        "warnings": warnings,
        "authority": {
            "da3": "supporting_relative_geometry_until_canonicalized",
            "canonical_geometry": "authoritative",
            "gaussian": "appearance_only",
            "nerf": "appearance_only",
            "certified_metrology": False,
        },
    }
    summary = out / "summary.json"
    summary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(f"Phase {PHASE} external backend + dataset preflight: {'PASS' if payload['pass'] else 'BLOCKED'}")
    print(f"mode={payload['mode']} dataset_images={dataset['image_count']} blockers={len(blockers)} warnings={len(warnings)}")
    if not args.static_only:
        print(
            "ready="
            f"pycolmap:{backends['pycolmap']['ready']} "
            f"da3:{backends['da3']['ready']} "
            f"gsplat:{backends['gaussian']['gsplat']['ready']} "
            f"graphdeco:{backends['gaussian']['graphdeco']['ready']} "
            f"nerf:{backends['nerf']['ready']}"
        )
    print(f"summary={summary}")
    return 0 if payload["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
