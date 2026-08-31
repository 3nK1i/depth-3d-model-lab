#!/usr/bin/env python3
"""I4.3G4B real single-image production-native qualification runner.

The runner is scenario-scoped. It consumes the closed G4A preflight, reuses the
exact frozen G2 authoritative PyCOLMAP SfM JSON without rerunning SfM, builds an
independent I4.3F adaptive plan for each of the six authorized single-image
providers, and executes those providers sequentially.

Matched-view diagnostics and manual review are deliberately excluded; they are
I4.3G4C work.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
import os
from pathlib import Path
import re
from statistics import median
import shutil
import subprocess
import sys
import threading
import time
from typing import Any, Callable, Mapping

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.hardware.gpu_info import detect_hardware  # noqa: E402
from core.hardware.profiles import ComputeProfile, resolve_profile  # noqa: E402
from core.models.catalog import (  # noqa: E402
    DEPTH_ANYTHING_V2_MODEL_ID,
    DEPTH_PRO_MODEL_ID,
    MARIGOLD_MODEL_ID,
    METRIC3D_V2_MODEL_ID,
    MOGE2_MODEL_ID,
    UNIDEPTH_V2_MODEL_ID,
)
from core.models.registry import resolve_model_runtime_config  # noqa: E402
from core.models.runtime_config import ModelRuntimeContext  # noqa: E402
from core.reconstruction.adaptive_production_planning import (  # noqa: E402
    finalize_adaptive_provider_execution_result,
    prepare_adaptive_provider_execution,
)
from core.reconstruction.adaptive_view_selection import AdaptiveSelectionResourceSnapshot  # noqa: E402
from core.reconstruction.dense_provider_architecture import DenseProviderRunStatus  # noqa: E402
from core.reconstruction.depth_anything_v2_dense_provider import (  # noqa: E402
    DepthAnythingV2ProductionProviderAdapter,
    DepthAnythingV2ProviderRunContext,
    depth_anything_v2_dense_provider_descriptor,
    probe_depth_anything_v2_environment,
)
from core.reconstruction.depth_pro_dense_provider import (  # noqa: E402
    DepthProProductionProviderAdapter,
    DepthProProviderRunContext,
    depth_pro_dense_provider_descriptor,
    probe_depth_pro_environment,
)
from core.reconstruction.errors import ReconstructionValidationError  # noqa: E402
from core.reconstruction.marigold_dense_provider import (  # noqa: E402
    MarigoldProductionProviderAdapter,
    MarigoldProviderRunContext,
    marigold_dense_provider_descriptor,
    probe_marigold_environment,
)
from core.reconstruction.metric3d_v2_dense_provider import (  # noqa: E402
    Metric3DV2ProductionProviderAdapter,
    Metric3DV2ProviderRunContext,
    metric3d_v2_dense_provider_descriptor,
    probe_metric3d_v2_environment,
)
from core.reconstruction.moge2_dense_provider import (  # noqa: E402
    MoGe2ProductionProviderAdapter,
    MoGe2ProviderRunContext,
    moge_2_dense_provider_descriptor,
    probe_moge_2_environment,
)
from core.reconstruction.mvs_da3_qualification import build_source_identity  # noqa: E402
from core.reconstruction.sfm_contracts import SfmStageStatistics, SfmWorkspaceManifest  # noqa: E402
from core.reconstruction.sfm_pycolmap import PYCOLMAP_BACKEND_ID  # noqa: E402
from core.reconstruction.sfm_result_serialization import offline_sfm_result_from_dict  # noqa: E402
from core.reconstruction.single_image_provider_production_qualification import (  # noqa: E402
    I43G4B_PROVIDER_IMPLEMENTATIONS,
    build_g4b_scenario_snapshot,
    recognized_quality_rejection,
    resolve_g4b_scenario_binding,
    sha256_file,
    validate_current_runtime_against_g4a,
    validate_i4_3g4b_predecessor,
    validate_single_image_production_plan,
    validate_single_image_provider_run_evidence,
)
from core.reconstruction.unidepth_v2_dense_provider import (  # noqa: E402
    UniDepthV2ProductionProviderAdapter,
    UniDepthV2ProviderRunContext,
    probe_unidepth_v2_environment,
    unidepth_v2_dense_provider_descriptor,
)

_MIB = 1024**2
_SUMMARY_FILENAMES = {
    "metric3d_v2": "i4_3e1_metric3d_v2_execution_summary.json",
    "unidepth_v2": "i4_3e2_unidepth_v2_execution_summary.json",
    "moge_2": "i4_3e3_moge_2_execution_summary.json",
    "depth_pro": "i4_3e4_depth_pro_execution_summary.json",
    "depth_anything_v2": "i4_3e5_depth_anything_v2_execution_summary.json",
    "marigold": "i4_3e6_marigold_execution_summary.json",
}
_MODEL_IDS = {
    "metric3d_v2": METRIC3D_V2_MODEL_ID,
    "unidepth_v2": UNIDEPTH_V2_MODEL_ID,
    "moge_2": MOGE2_MODEL_ID,
    "depth_pro": DEPTH_PRO_MODEL_ID,
    "depth_anything_v2": DEPTH_ANYTHING_V2_MODEL_ID,
    "marigold": MARIGOLD_MODEL_ID,
}
_WORKER_SCRIPTS = {
    "metric3d_v2": "reconstruction_single_image_depth_provider_worker.py",
    "unidepth_v2": "reconstruction_unidepth_v2_provider_worker.py",
    "moge_2": "reconstruction_moge2_provider_worker.py",
    "depth_pro": "reconstruction_depth_pro_provider_worker.py",
    "depth_anything_v2": "reconstruction_depth_anything_v2_provider_worker.py",
    "marigold": "reconstruction_marigold_provider_worker.py",
}
_WORKER_RESULT_MANIFESTS = {
    "metric3d_v2": "single_image_depth_result_manifest.json",
    "unidepth_v2": "unidepth_v2_result_manifest.json",
    "moge_2": "moge2_result_manifest.json",
    "depth_pro": "depth_pro_result_manifest.json",
    "depth_anything_v2": "depth_anything_v2_result_manifest.json",
    "marigold": "marigold_result_manifest.json",
}
_REJECTION_COUNTS = re.compile(r"accepted=(?P<accepted>\d+)\s+required=(?P<required>\d+)")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("preflight", "production-native"), default="preflight")
    parser.add_argument("--scenario", required=True)
    parser.add_argument("--i4-3g4a-snapshot", type=Path, required=True)
    parser.add_argument("--i4-3g4a-sha256", required=True)
    parser.add_argument("--project-root", type=Path, default=ROOT)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--metric3d-checkpoint", type=Path)
    parser.add_argument("--unidepth-checkpoint-dir", type=Path)
    parser.add_argument("--unidepth-variant", choices=("auto", "small", "base", "large"), default="auto")
    parser.add_argument("--moge-variant", choices=("auto", "small", "base", "large"), default="auto")
    parser.add_argument("--depth-pro-checkpoint", type=Path)
    parser.add_argument("--dav2-source-dir", type=Path)
    parser.add_argument("--dav2-checkpoint", type=Path)
    parser.add_argument("--dav2-variant", choices=("auto", "small", "base", "large"), default="auto")
    parser.add_argument("--marigold-checkpoint-dir", type=Path)
    parser.add_argument("--marigold-profile", choices=("auto", "efficient", "balanced", "ultra"), default="auto")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--attempt", type=int, default=1)
    args = parser.parse_args()
    if args.attempt < 1:
        parser.error("--attempt must be >= 1")
    return parser


def _read_json(path: Path, *, label: str) -> dict[str, Any]:
    source = Path(path).expanduser().resolve()
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ReconstructionValidationError(f"Unable to read {label}: {source}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ReconstructionValidationError(f"{label} must contain a JSON object: {source}")
    return payload


def _write_json(path: Path, payload: Mapping[str, Any]) -> Path:
    target = Path(path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(dict(payload), indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    return target


def _artifact(path: Path) -> dict[str, Any]:
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise ReconstructionValidationError(f"I4.3G4B expected artifact is missing: {source}")
    return {"path": str(source), "sha256": sha256_file(source), "size_bytes": int(source.stat().st_size)}


def _hash_artifacts(paths: list[Path]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw in paths:
        source = Path(raw).expanduser().resolve()
        key = str(source)
        if key in seen:
            continue
        seen.add(key)
        result.append(_artifact(source))
    return result


def _verify_sha(path: Path, expected: str, *, label: str) -> str:
    source = Path(path).expanduser().resolve()
    expected_value = str(expected).strip().lower()
    if len(expected_value) != 64 or any(ch not in "0123456789abcdef" for ch in expected_value):
        raise ReconstructionValidationError(f"I4.3G4B invalid expected SHA256 for {label}: {expected!r}")
    actual = sha256_file(source)
    if actual != expected_value:
        raise ReconstructionValidationError(
            f"I4.3G4B {label} SHA256 mismatch: expected={expected_value} actual={actual} path={source}"
        )
    return actual


def _verify_g4a_predecessor_artifacts(g4a: Mapping[str, Any]) -> dict[str, Any]:
    predecessor = g4a.get("predecessor_evidence")
    if not isinstance(predecessor, Mapping):
        raise ReconstructionValidationError("I4.3G4B requires G4A predecessor_evidence.")
    verified: dict[str, Any] = {}
    for key in ("i4_3g1_snapshot", "i4_3g3c_final_closure"):
        raw = predecessor.get(key)
        if not isinstance(raw, Mapping):
            raise ReconstructionValidationError(f"I4.3G4B requires G4A predecessor artifact: {key}")
        path = Path(str(raw.get("path", ""))).expanduser().resolve()
        sha = str(raw.get("sha256", "")).strip().lower()
        _verify_sha(path, sha, label=f"G4A-bound {key}")
        verified[key] = {"path": str(path), "sha256": sha}
    return verified


def _available_ram_bytes() -> int | None:
    try:
        import psutil  # type: ignore
        return int(psutil.virtual_memory().available)
    except Exception:
        pass
    try:
        pages = int(os.sysconf("SC_AVPHYS_PAGES"))
        page_size = int(os.sysconf("SC_PAGE_SIZE"))
        return pages * page_size if pages > 0 and page_size > 0 else None
    except (AttributeError, OSError, ValueError):
        return None


def _adaptive_resources() -> tuple[Any, Any, AdaptiveSelectionResourceSnapshot]:
    hardware = detect_hardware(0)
    resolved = resolve_profile(ComputeProfile.AUTO, hardware)
    resources = AdaptiveSelectionResourceSnapshot.from_hardware(
        hardware,
        available_ram_bytes=_available_ram_bytes(),
    )
    return hardware, resolved, resources


class _NvidiaMemorySampler:
    """Best-available device-wide peak used-memory sampler for G1 evidence."""

    def __init__(self, *, gpu_index: int = 0, interval_s: float = 0.20) -> None:
        self.gpu_index = int(gpu_index)
        self.interval_s = float(interval_s)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._samples: list[float] = []
        self._error = ""

    def _query(self) -> float | None:
        tool = shutil.which("nvidia-smi")
        if not tool:
            self._error = "nvidia_smi_missing"
            return None
        try:
            completed = subprocess.run(
                [tool, "--query-gpu=memory.used", "--format=csv,noheader,nounits", "-i", str(self.gpu_index)],
                text=True,
                capture_output=True,
                check=False,
                timeout=3.0,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            self._error = f"nvidia_smi_query_failed:{type(exc).__name__}:{exc}"
            return None
        if completed.returncode != 0:
            self._error = f"nvidia_smi_query_failed:{completed.stderr.strip() or completed.returncode}"
            return None
        try:
            value = float(completed.stdout.strip().splitlines()[0])
        except (ValueError, IndexError):
            self._error = "nvidia_smi_memory_parse_failed"
            return None
        return value if value >= 0.0 else None

    def _loop(self) -> None:
        while not self._stop.is_set():
            value = self._query()
            if value is not None:
                self._samples.append(value)
            self._stop.wait(self.interval_s)

    def start(self) -> None:
        baseline = self._query()
        if baseline is not None:
            self._samples.append(baseline)
        self._thread = threading.Thread(target=self._loop, name="i43g4b-nvidia-memory", daemon=True)
        self._thread.start()

    def stop(self) -> dict[str, Any]:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(1.0, self.interval_s * 5.0))
        final = self._query()
        if final is not None:
            self._samples.append(final)
        if not self._samples:
            raise ReconstructionValidationError(
                f"I4.3G4B could not measure required peak GPU memory: {self._error or 'no_samples'}"
            )
        return {
            "measurement_scope": "device_wide_nvidia_smi_peak_used_memory",
            "gpu_index": self.gpu_index,
            "sample_interval_s": self.interval_s,
            "sample_count": len(self._samples),
            "baseline_used_mib": float(self._samples[0]),
            "peak_used_mib": float(max(self._samples)),
            "final_used_mib": float(self._samples[-1]),
            "query_error": self._error,
        }


def _measured_call(call: Callable[[], Any]) -> tuple[Any | None, Exception | None, float, dict[str, Any]]:
    sampler = _NvidiaMemorySampler()
    sampler.start()
    started = time.perf_counter()
    result: Any | None = None
    error: Exception | None = None
    try:
        result = call()
    except Exception as exc:  # noqa: BLE001 - classification occurs after measurement is frozen
        error = exc
    finally:
        wall = time.perf_counter() - started
        resources = sampler.stop()
    return result, error, wall, resources


def _live_runtime_probes(args: argparse.Namespace) -> dict[str, Any]:
    project = args.project_root.expanduser().resolve()
    python = args.python.expanduser().resolve()
    return {
        "metric3d_v2": probe_metric3d_v2_environment(
            project_root=project,
            python_executable=python,
            checkpoint_path=args.metric3d_checkpoint.expanduser().resolve() if args.metric3d_checkpoint else None,
        ),
        "unidepth_v2": probe_unidepth_v2_environment(
            project_root=project,
            python_executable=python,
            checkpoint_path=args.unidepth_checkpoint_dir.expanduser().resolve() if args.unidepth_checkpoint_dir else None,
            variant=args.unidepth_variant,
        ),
        "moge_2": probe_moge_2_environment(
            project_root=project,
            python_executable=python,
            variant=args.moge_variant,
        ),
        "depth_pro": probe_depth_pro_environment(
            project_root=project,
            python_executable=python,
            checkpoint_path=args.depth_pro_checkpoint.expanduser().resolve() if args.depth_pro_checkpoint else None,
        ),
        "depth_anything_v2": probe_depth_anything_v2_environment(
            project_root=project,
            python_executable=python,
            variant=args.dav2_variant,
            source_directory=args.dav2_source_dir.expanduser().resolve() if args.dav2_source_dir else None,
            checkpoint_path=args.dav2_checkpoint.expanduser().resolve() if args.dav2_checkpoint else None,
        ),
        "marigold": probe_marigold_environment(
            project_root=project,
            python_executable=python,
            profile=args.marigold_profile,
            checkpoint_dir=args.marigold_checkpoint_dir.expanduser().resolve() if args.marigold_checkpoint_dir else None,
        ),
    }


def _storage_state(g4a: Mapping[str, Any]) -> dict[str, Any]:
    storage = g4a.get("storage")
    if not isinstance(storage, Mapping):
        raise ReconstructionValidationError("I4.3G4B G4A storage section is missing.")
    path = Path(str(storage.get("storage_path", ""))).expanduser().resolve()
    minimum = int(storage.get("minimum_free_disk_bytes", 0))
    if not path.exists() or minimum <= 0:
        raise ReconstructionValidationError("I4.3G4B G4A storage binding is invalid.")
    usage = shutil.disk_usage(path)
    ready = int(usage.free) >= minimum
    return {
        "storage_path": str(path),
        "free_disk_bytes": int(usage.free),
        "free_disk_gib": int(usage.free) / 1024**3,
        "minimum_free_disk_bytes": minimum,
        "minimum_free_disk_gib": minimum / 1024**3,
        "storage_ready": ready,
        "threshold_reused_from_g4a": True,
    }


def _require_storage_ready(g4a: Mapping[str, Any], *, stage: str) -> dict[str, Any]:
    state = _storage_state(g4a)
    if state["storage_ready"] is not True:
        raise ReconstructionValidationError(
            "I4.3G4B storage safety floor failed "
            f"{stage}: free_gib={state['free_disk_gib']:.2f} "
            f"minimum_gib={state['minimum_free_disk_gib']:.2f} path={state['storage_path']}"
        )
    return state


def _runtime_preflight(
    *,
    g4a: Mapping[str, Any],
    binding,
    sfm_result: Any,
    current_probes: Mapping[str, Any],
    hardware: Any,
    resolved_profile: Any,
    resources: AdaptiveSelectionResourceSnapshot,
    storage_state: Mapping[str, Any],
) -> dict[str, Any]:
    frozen_runtime = g4a.get("runtime")
    if not isinstance(frozen_runtime, Mapping):
        raise ReconstructionValidationError("I4.3G4B G4A runtime section is missing.")
    current_payload: dict[str, Any] = {}
    production_bindings: dict[str, Any] = {}
    for implementation in I43G4B_PROVIDER_IMPLEMENTATIONS:
        probe = current_probes[implementation]
        payload = probe.as_dict()
        validate_current_runtime_against_g4a(
            implementation=implementation,
            frozen_runtime=frozen_runtime[implementation],
            current_runtime=payload,
        )
        current_payload[implementation] = payload

        descriptor = _descriptor(implementation, probe)
        prep = prepare_adaptive_provider_execution(
            sfm_result,
            provider=descriptor,
            reconstruction_frame_id=binding.authoritative_sfm_frame_id,
            resources=resources,
        )
        plan_payload = prep.plan.as_dict()
        validate_single_image_production_plan(plan_payload, implementation=implementation)
        runtime_config, device, precision = _runtime_config(
            implementation, probe, hardware, resolved_profile
        )
        worker_script = (ROOT / "scripts" / _WORKER_SCRIPTS[implementation]).resolve()
        if not worker_script.is_file():
            raise ReconstructionValidationError(
                f"I4.3G4B worker script is missing for {implementation}: {worker_script}"
            )
        production_bindings[implementation] = {
            "provider_id": descriptor.provider_id,
            "implementation": implementation,
            "selected_view_count": int(prep.plan.selected_view_count),
            "source_view_count": int(prep.plan.source_view_count),
            "selected_keyframe_ids": list(prep.plan.selected_keyframe_ids),
            "plan_reason_codes": list(prep.plan.reason_codes),
            "device": device,
            "precision": precision,
            "runtime_profile": str(runtime_config.get("profile", "production_native_i4_3f_adaptive")),
            "worker_script": str(worker_script),
            "worker_script_sha256": sha256_file(worker_script),
            "provider_inference_started": False,
        }
    return {
        "schema_version": 1,
        "phase": "I4.3G4B",
        "status": "preflight_pass",
        "pass": True,
        "scenario": binding.as_dict(),
        "runtime": current_payload,
        "production_bindings": production_bindings,
        "hardware": {
            "backend": str(getattr(getattr(hardware, "backend", None), "value", getattr(hardware, "backend", ""))),
            "device": str(getattr(hardware, "device", "")),
            "device_name": str(getattr(hardware, "name", "")),
            "total_memory_gib": getattr(hardware, "total_memory_gib", None),
            "free_memory_gib": getattr(hardware, "free_memory_gib", None),
            "resolved_profile": str(getattr(getattr(resolved_profile, "effective", None), "value", "")),
            "resolved_precision": str(getattr(getattr(resolved_profile, "precision", None), "value", "")),
            "profile_downgraded": bool(getattr(resolved_profile, "downgraded", False)),
            "profile_warning": getattr(resolved_profile, "warning", None),
        },
        "storage": dict(storage_state),
        "scope": {
            "authoritative_sfm_reused_from_g2": True,
            "authoritative_sfm_rerun": False,
            "provider_inference_started": False,
            "matched_view_diagnostic_started": False,
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


def _verify_frozen_sfm(binding) -> Any:
    _verify_sha(
        binding.authoritative_sfm_snapshot_path,
        binding.authoritative_sfm_snapshot_sha256,
        label=f"{binding.scenario_id} frozen authoritative SfM",
    )
    payload = _read_json(binding.authoritative_sfm_snapshot_path, label="frozen authoritative SfM JSON")
    result = offline_sfm_result_from_dict(payload)
    if result.backend_id != PYCOLMAP_BACKEND_ID:
        raise ReconstructionValidationError("I4.3G4B frozen authoritative SfM backend is not PyCOLMAP.")
    if result.statistics.registered_keyframes != binding.registered_view_count:
        raise ReconstructionValidationError("I4.3G4B frozen SfM registered-view count drifted from G4A.")
    if result.statistics.landmarks != binding.landmark_count:
        raise ReconstructionValidationError("I4.3G4B frozen SfM landmark count drifted from G4A.")
    root = Path(result.workspace.workspace_root).expanduser().resolve()
    ordered = tuple(sorted(result.workspace.images, key=lambda item: item.order))
    # The workspace is the materialized SfM *input* set. PyCOLMAP may register
    # only a subset of those keyframes, so comparing workspace length with the
    # registered-camera count is invalid. The core SfM contract instead binds
    # workspace length to input_keyframes and registered_keyframes to cameras.
    if len(ordered) != result.statistics.input_keyframes:
        raise ReconstructionValidationError("I4.3G4B frozen SfM workspace/input-keyframe count is inconsistent.")
    registered_ids = {camera.keyframe_id for camera in result.cameras}
    workspace_ids = {item.keyframe_id for item in ordered}
    if len(registered_ids) != binding.registered_view_count or not registered_ids.issubset(workspace_ids):
        raise ReconstructionValidationError("I4.3G4B frozen SfM registered-camera/workspace binding is inconsistent.")
    for item in ordered:
        path = (root / item.staged_image_path).resolve()
        if not path.is_file():
            raise ReconstructionValidationError(f"I4.3G4B frozen SfM staged image is missing: {path}")
        if sha256_file(path) != item.staged_sha256:
            raise ReconstructionValidationError(f"I4.3G4B frozen SfM staged image checksum drifted: {item.keyframe_id}")
    return result


def _registered_view_projection(sfm_result: Any) -> Any:
    """Project frozen SfM to its registered camera views for dense-provider input.

    This is an immutable provider-input projection only. Camera transforms and
    landmark coordinates stay in the exact authoritative PyCOLMAP frame; no SfM
    is rerun and the frozen source snapshot remains untouched.
    """

    registered = {camera.keyframe_id for camera in sfm_result.cameras}
    ordered_workspace = tuple(sorted(sfm_result.workspace.images, key=lambda item: item.order))
    selected_workspace = tuple(item for item in ordered_workspace if item.keyframe_id in registered)
    if len(selected_workspace) != sfm_result.statistics.registered_keyframes:
        raise ReconstructionValidationError("I4.3G4B could not project every registered SfM camera to one workspace image.")
    if len(selected_workspace) == len(ordered_workspace):
        return sfm_result

    projected_images = tuple(replace(item, order=index) for index, item in enumerate(selected_workspace))
    projected_workspace = SfmWorkspaceManifest(
        session_id=sfm_result.workspace.session_id,
        source=sfm_result.workspace.source,
        workspace_root=sfm_result.workspace.workspace_root,
        images_directory=sfm_result.workspace.images_directory,
        database_path=sfm_result.workspace.database_path,
        sparse_directory=sfm_result.workspace.sparse_directory,
        checkpoints_directory=sfm_result.workspace.checkpoints_directory,
        reports_directory=sfm_result.workspace.reports_directory,
        logs_directory=sfm_result.workspace.logs_directory,
        images=projected_images,
    )
    projected_landmarks = []
    for landmark in sfm_result.landmarks:
        observations = tuple(obs for obs in landmark.observations if obs.keyframe_id in registered)
        if len(observations) < 2:
            continue
        projected_landmarks.append(replace(landmark, observations=observations))
    observation_count = sum(len(item.observations) for item in projected_landmarks)
    camera_ids = {camera.camera_id for camera in sfm_result.cameras}
    projected_camera_models = tuple(model for model in sfm_result.camera_models if model.camera_id in camera_ids)
    metadata = dict(sfm_result.metadata)
    metadata["i4_3g4b_registered_view_projection"] = {
        "projection_only": True,
        "authoritative_sfm_modified": False,
        "original_input_keyframes": sfm_result.statistics.input_keyframes,
        "registered_keyframes": sfm_result.statistics.registered_keyframes,
        "camera_pose_authority_owner": "pycolmap_sfm",
    }
    return replace(
        sfm_result,
        workspace=projected_workspace,
        camera_models=projected_camera_models,
        landmarks=tuple(projected_landmarks),
        statistics=SfmStageStatistics(
            input_keyframes=len(projected_images),
            registered_keyframes=len(sfm_result.cameras),
            landmarks=len(projected_landmarks),
            observations=observation_count,
            elapsed_ns=sfm_result.statistics.elapsed_ns,
            mean_reprojection_error_px=sfm_result.statistics.mean_reprojection_error_px,
        ),
        metadata=metadata,
    )


def _runtime_config(implementation: str, probe: Any, hardware: Any, resolved: Any) -> tuple[dict[str, object], str, str]:
    overrides: dict[str, object] = {}
    if implementation == "metric3d_v2":
        overrides.update({"source_dir": probe.source_directory, "checkpoint": probe.checkpoint_path})
    elif implementation == "unidepth_v2":
        folder = {"small": "vits14", "base": "vitb14", "large": "vitl14"}[probe.variant]
        overrides.update({
            "checkpoint_dir": probe.checkpoint_path,
            "repository": probe.repository,
            "variant": folder,
        })
    elif implementation == "moge_2":
        overrides.update({"pretrained": probe.checkpoint_repository})
    elif implementation == "depth_pro":
        overrides.update({"checkpoint": probe.checkpoint_path})
    elif implementation == "depth_anything_v2":
        encoder = {"small": "vits", "base": "vitb", "large": "vitl"}[probe.variant]
        overrides.update({
            "source_dir": probe.source_directory,
            "checkpoint": probe.checkpoint_path,
            "encoder": encoder,
        })
    elif implementation == "marigold":
        overrides.update({
            "repository": probe.checkpoint_repository,
            "checkpoint_dir": probe.checkpoint_path,
            "profile": probe.profile,
        })
    else:  # pragma: no cover - frozen implementation tuple protects this branch
        raise ReconstructionValidationError(f"Unsupported I4.3G4B provider: {implementation}")
    config = resolve_model_runtime_config(
        _MODEL_IDS[implementation],
        ModelRuntimeContext(
            hardware_info=hardware,
            resolved_profile=resolved,
            overrides=overrides,
        ),
    )
    device = str(config.get("device", resolved.device))
    precision = str(config.get("precision", resolved.precision.value)).lower()
    if not device.startswith("cuda"):
        raise ReconstructionValidationError(f"I4.3G4B production qualification requires CUDA runtime: {implementation} device={device}")
    if precision not in {"fp32", "fp16"}:
        raise ReconstructionValidationError(f"I4.3G4B unsupported provider precision: {implementation} precision={precision}")
    # The profile builder must resolve to the same model/checkpoint identity that G4A froze.
    if implementation in {"metric3d_v2", "depth_pro", "depth_anything_v2"}:
        configured = Path(str(config.get("checkpoint", ""))).expanduser()
        if not configured.is_absolute():
            configured = (ROOT / configured).resolve()
        if configured != Path(probe.checkpoint_path).expanduser().resolve():
            raise ReconstructionValidationError(
                f"I4.3G4B hardware-resolved checkpoint drift for {implementation}: probe={probe.checkpoint_path} config={configured}"
            )
    if implementation == "unidepth_v2":
        configured = Path(str(config.get("checkpoint_dir", ""))).expanduser()
        if not configured.is_absolute():
            configured = (ROOT / configured).resolve()
        if configured != Path(probe.checkpoint_path).expanduser().resolve() or str(config.get("repository")) != probe.repository:
            raise ReconstructionValidationError("I4.3G4B UniDepth runtime config drifted from the G4A-bound checkpoint identity.")
    if implementation == "moge_2" and str(config.get("pretrained")) != probe.checkpoint_repository:
        raise ReconstructionValidationError("I4.3G4B MoGe-2 runtime config drifted from the G4A-bound checkpoint identity.")
    if implementation == "marigold":
        configured = Path(str(config.get("checkpoint_dir", ""))).expanduser()
        if not configured.is_absolute():
            configured = (ROOT / configured).resolve()
        if configured != Path(probe.checkpoint_path).expanduser().resolve() or str(config.get("repository")) != probe.checkpoint_repository:
            raise ReconstructionValidationError("I4.3G4B Marigold runtime config drifted from the G4A-bound checkpoint identity.")
    return config, device, precision


def _descriptor(implementation: str, probe: Any):
    return {
        "metric3d_v2": metric3d_v2_dense_provider_descriptor,
        "unidepth_v2": unidepth_v2_dense_provider_descriptor,
        "moge_2": moge_2_dense_provider_descriptor,
        "depth_pro": depth_pro_dense_provider_descriptor,
        "depth_anything_v2": depth_anything_v2_dense_provider_descriptor,
        "marigold": marigold_dense_provider_descriptor,
    }[implementation](probe)


def _adapter(implementation: str, *, sfm_result: Any, frame_id: str, output_dir: Path, probe: Any, runtime_config: Mapping[str, object], device: str, precision: str):
    common = {
        "sfm_result": sfm_result,
        "reconstruction_frame_id": frame_id,
        "output_dir": output_dir,
        "python_executable": Path(probe.python_executable),
        "worker_script": ROOT / "scripts" / _WORKER_SCRIPTS[implementation],
        "runtime_config": runtime_config,
        "device": device,
        "precision": precision,
    }
    if implementation == "metric3d_v2":
        context = Metric3DV2ProviderRunContext(**common)
        return Metric3DV2ProductionProviderAdapter(context=context, environment=probe)
    if implementation == "unidepth_v2":
        context = UniDepthV2ProviderRunContext(**common)
        return UniDepthV2ProductionProviderAdapter(context=context, environment=probe)
    if implementation == "moge_2":
        context = MoGe2ProviderRunContext(**common)
        return MoGe2ProductionProviderAdapter(context=context, environment=probe)
    if implementation == "depth_pro":
        context = DepthProProviderRunContext(**common)
        return DepthProProductionProviderAdapter(context=context, environment=probe)
    if implementation == "depth_anything_v2":
        context = DepthAnythingV2ProviderRunContext(**common)
        return DepthAnythingV2ProductionProviderAdapter(context=context, environment=probe)
    if implementation == "marigold":
        context = MarigoldProviderRunContext(**common)
        return MarigoldProductionProviderAdapter(context=context, environment=probe)
    raise ReconstructionValidationError(f"Unsupported I4.3G4B provider: {implementation}")


def _canonical_point_count(path: Path) -> int:
    source = Path(path).expanduser().resolve()
    try:
        with np.load(source, allow_pickle=False) as archive:
            points = np.asarray(archive["points_reconstruction"])
    except Exception as exc:  # noqa: BLE001
        raise ReconstructionValidationError(f"I4.3G4B could not inspect canonical point artifact {source}: {exc}") from exc
    if points.ndim != 2 or points.shape[1] != 3:
        raise ReconstructionValidationError("I4.3G4B canonical point artifact has invalid point shape.")
    return int(points.shape[0])


def _worker_artifacts(provider_out: Path, *, implementation: str) -> tuple[list[Path], int]:
    if implementation not in _WORKER_RESULT_MANIFESTS:
        raise ReconstructionValidationError(f"Unsupported I4.3G4B worker evidence provider: {implementation}")
    worker_manifest = provider_out / "provider" / "worker_output" / _WORKER_RESULT_MANIFESTS[implementation]
    input_manifest = provider_out / "provider" / "input_manifest.json"
    payload = _read_json(worker_manifest, label="I4.3G4B single-image worker result")
    records = payload.get("frame_records")
    if not isinstance(records, list) or not records:
        raise ReconstructionValidationError("I4.3G4B quality rejection lacks complete worker frame evidence.")
    paths = [input_manifest, worker_manifest]
    base = worker_manifest.parent
    for record in records:
        if not isinstance(record, Mapping):
            raise ReconstructionValidationError("I4.3G4B worker frame record is malformed.")
        raw = str(record.get("evidence_npz_path", "")).strip()
        if not raw:
            raise ReconstructionValidationError("I4.3G4B worker frame record lacks native evidence path.")
        path = Path(raw).expanduser()
        if not path.is_absolute():
            path = (base / path).resolve()
        paths.append(path)
    return paths, len(records)


def _alignment_metrics(summary: Mapping[str, Any], *, selected: int) -> tuple[int, dict[str, Any], dict[str, str]]:
    alignment = summary.get("alignment")
    if not isinstance(alignment, Mapping):
        raise ReconstructionValidationError("I4.3G4B provider summary is missing alignment evidence.")
    accepted = int(alignment.get("accepted_views", 0))
    if accepted < 0 or accepted > selected:
        raise ReconstructionValidationError("I4.3G4B provider alignment accepted-view count is invalid.")
    records = alignment.get("records")
    if not isinstance(records, list) or len(records) != selected:
        raise ReconstructionValidationError("I4.3G4B provider summary must retain one alignment record per selected view.")
    accepted_records = [item for item in records if isinstance(item, Mapping) and item.get("status") == "accepted"]
    candidates = sum(
        int(item.get("candidate_observations", item.get("input_correspondences", 0)))
        for item in records
        if isinstance(item, Mapping)
    )
    valid_corr = sum(int(item.get("valid_correspondences", 0)) for item in records if isinstance(item, Mapping))
    medians = [
        float(item["median_relative_residual"])
        for item in accepted_records
        if item.get("median_relative_residual") is not None
    ]
    optional: dict[str, Any] = {"registered_view_surface_coverage": accepted / float(selected)}
    reasons: dict[str, str] = {}
    if candidates > 0:
        optional["sparse_correspondence_coverage"] = min(1.0, valid_corr / float(candidates))
    else:
        reasons["sparse_correspondence_coverage"] = "provider_alignment_records_report_no_eligible_sparse_correspondence_candidates"
    if medians:
        optional["alignment_median_relative_residual"] = float(median(medians))
    else:
        reasons["alignment_median_relative_residual"] = "provider_alignment_records_have_no_accepted_residual_values"
    # Do not relabel p90 as p95. Persist only an actual p95 field if a provider adds one later.
    actual_p95 = [
        float(item["p95_relative_residual"])
        for item in accepted_records
        if item.get("p95_relative_residual") is not None
    ]
    if actual_p95:
        optional["alignment_p95_relative_residual"] = float(median(actual_p95))
    else:
        reasons["alignment_p95_relative_residual"] = "frozen_E_series_adapters_do_not_persist_per_view_p95_no_posthoc_interpolation"
    return accepted, optional, reasons


def _provider_evidence(
    *,
    implementation: str,
    binding: Any,
    attempt: int,
    plan_path: Path,
    plan: Any,
    descriptor: Any,
    runtime_config: Mapping[str, object],
    wall_time_s: float,
    resource_evidence: Mapping[str, Any],
    processed_view_count: int,
    aligned_view_fraction: float,
    artifacts: list[dict[str, Any]],
    status: str,
    optional_metrics: Mapping[str, Any] | None = None,
    optional_reasons: Mapping[str, str] | None = None,
    canonical_point_count: int | None = None,
    execution_completed: bool = True,
    rejection: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    metrics: dict[str, Any] = {
        "execution_completed": bool(execution_completed),
        "source_input_identity_preserved": True,
        "authoritative_sfm_preserved": True,
        "adaptive_plan_bound": True,
        "provider_provenance_complete": True,
        "forbidden_authority_claims_absent": True,
        "wall_time_s": float(wall_time_s),
        "peak_vram_mib": float(resource_evidence["peak_used_mib"]),
        "selected_view_count": int(plan.selected_view_count),
        "processed_view_count": int(processed_view_count),
        "aligned_view_fraction": float(aligned_view_fraction),
    }
    if canonical_point_count is not None:
        metrics["canonical_point_count"] = int(canonical_point_count)
    metrics.update(dict(optional_metrics or {}))
    reasons = dict(optional_reasons or {})
    all_optional = (
        "peak_host_memory_mib",
        "sparse_correspondence_coverage",
        "alignment_median_relative_residual",
        "alignment_p95_relative_residual",
        "inter_view_consistency_ratio",
        "inter_view_median_relative_residual",
        "registered_view_surface_coverage",
        "canonical_point_count",
        "detached_component_fraction",
        "unsupported_geometry_fraction",
    )
    for metric in all_optional:
        if metric not in metrics and metric not in reasons:
            reasons[metric] = "not_available_from_existing_E_series_production_adapter_without_new_G4_quality_policy"
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "scenario_id": binding.scenario_id,
        "provider_implementation": implementation,
        "comparison_mode": "production_native",
        "attempt": int(attempt),
        "status": status,
        "source_identity": {
            "source_directory": str(binding.source_directory),
            "image_count": binding.source_image_count,
            "asset_set_sha256": binding.source_asset_set_sha256,
            "manifest_path": str(binding.source_manifest_path),
            "manifest_sha256": binding.source_manifest_sha256,
        },
        "authoritative_sfm_identity": {
            "snapshot_path": str(binding.authoritative_sfm_snapshot_path),
            "snapshot_sha256": binding.authoritative_sfm_snapshot_sha256,
            "frame_id": binding.authoritative_sfm_frame_id,
            "registered_view_count": binding.registered_view_count,
            "landmark_count": binding.landmark_count,
            "reused_without_rerun": True,
        },
        "adaptive_plan_identity": {
            "plan_path": str(plan_path.resolve()),
            "plan_sha256": sha256_file(plan_path),
            "selected_view_ids": list(plan.selected_keyframe_ids),
        },
        "provider_provenance": {
            "provider_id": descriptor.provider_id,
            "implementation": implementation,
            "backend_version": descriptor.backend_version,
            "adapter_version": descriptor.adapter_version,
            "effective_profile": str(runtime_config.get("profile", "production_native_i4_3f_adaptive")),
            "checkpoint_or_model_identity": dict(descriptor.metadata),
            "runtime_config": dict(runtime_config),
            "plan_reason_codes": list(plan.reason_codes),
        },
        "runtime_resources": {
            "wall_time_s": float(wall_time_s),
            "peak_vram_mib": float(resource_evidence["peak_used_mib"]),
            "peak_host_memory_mib": None,
            **dict(resource_evidence),
        },
        "metrics": metrics,
        "optional_metric_reasons": reasons,
        "artifacts": artifacts,
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
    if rejection is not None:
        manifest["rejection"] = dict(rejection)
    validate_single_image_provider_run_evidence(manifest)
    return manifest


def _verify_source_and_sfm_unchanged(binding: Any, *, stage: str) -> None:
    current = build_source_identity(binding.source_directory, minimum_images=3)
    if current.image_count != binding.source_image_count or current.asset_set_sha256 != binding.source_asset_set_sha256:
        raise ReconstructionValidationError(f"I4.3G4B source identity changed {stage}.")
    _verify_sha(
        binding.authoritative_sfm_snapshot_path,
        binding.authoritative_sfm_snapshot_sha256,
        label=f"frozen authoritative SfM {stage}",
    )


def _execute_provider(
    *,
    implementation: str,
    binding: Any,
    sfm_result: Any,
    resources: AdaptiveSelectionResourceSnapshot,
    hardware: Any,
    resolved_profile: Any,
    probe: Any,
    output_dir: Path,
    attempt: int,
) -> dict[str, Any]:
    descriptor = _descriptor(implementation, probe)
    prep = prepare_adaptive_provider_execution(
        sfm_result,
        provider=descriptor,
        reconstruction_frame_id=binding.authoritative_sfm_frame_id,
        resources=resources,
    )
    validate_single_image_production_plan(prep.plan.as_dict(), implementation=implementation)
    plan_path = _write_json(output_dir / "adaptive_plans" / f"{implementation}.json", prep.plan.as_dict())

    _verify_source_and_sfm_unchanged(binding, stage=f"immediately before {implementation}")
    runtime_config, device, precision = _runtime_config(implementation, probe, hardware, resolved_profile)
    provider_out = output_dir / "providers" / implementation
    adapter = _adapter(
        implementation,
        sfm_result=prep.provider_sfm_result,
        frame_id=binding.authoritative_sfm_frame_id,
        output_dir=provider_out,
        probe=probe,
        runtime_config=runtime_config,
        device=device,
        precision=precision,
    )
    local_result, error, wall, monitor = _measured_call(
        lambda: adapter.execute(prep.selection_request, prep.selection_decision)
    )

    if error is not None:
        message = str(error)
        if not isinstance(error, ReconstructionValidationError) or not recognized_quality_rejection(
            implementation=implementation,
            error_message=message,
        ):
            raise error
        match = _REJECTION_COUNTS.search(message)
        if match is None:
            raise ReconstructionValidationError(
                f"I4.3G4B recognized {implementation} rejection lacks accepted/required counts: {message}"
            )
        accepted = int(match.group("accepted"))
        required = int(match.group("required"))
        native_paths, processed = _worker_artifacts(provider_out, implementation=implementation)
        selected = prep.plan.selected_view_count
        if processed != selected or accepted < 0 or accepted >= required or accepted > selected:
            raise ReconstructionValidationError(
                f"I4.3G4B rejected {implementation} worker/accounting mismatch: selected={selected} processed={processed} accepted={accepted} required={required}"
            )
        evidence = _provider_evidence(
            implementation=implementation,
            binding=binding,
            attempt=attempt,
            plan_path=plan_path,
            plan=prep.plan,
            descriptor=descriptor,
            runtime_config=runtime_config,
            wall_time_s=wall,
            resource_evidence=monitor,
            processed_view_count=processed,
            aligned_view_fraction=accepted / float(selected),
            artifacts=_hash_artifacts(native_paths),
            status="rejected",
            optional_metrics={"registered_view_surface_coverage": accepted / float(selected)},
            optional_reasons={
                "peak_host_memory_mib": "not_measured_by_existing_E_series_worker_or_G4B_device_sampler",
                "sparse_correspondence_coverage": "E_series_adapter_does_not_persist_alignment_records_when_terminal_minimum_accepted_views_gate_rejects",
                "alignment_median_relative_residual": "E_series_adapter_does_not_persist_alignment_records_when_terminal_minimum_accepted_views_gate_rejects",
                "alignment_p95_relative_residual": "E_series_adapter_does_not_persist_alignment_records_when_terminal_minimum_accepted_views_gate_rejects",
            },
            canonical_point_count=None,
            execution_completed=False,
            rejection={
                "terminal": True,
                "recognized_existing_quality_gate": True,
                "reason": message,
                "accepted_views": accepted,
                "required_accepted_views": required,
                "canonical_geometry_generated": False,
                "threshold_changed": False,
                "provider_execution_reran": False,
                "scenario_execution_continues_to_other_provider": True,
            },
        )
        evidence_path = _write_json(output_dir / "evidence" / f"{implementation}_production_native.json", evidence)
        _verify_source_and_sfm_unchanged(binding, stage=f"after rejected {implementation}")
        return {
            "evidence": _artifact(evidence_path),
            "status": "rejected",
            "result": None,
            "resource_evidence": monitor,
        }

    if local_result is None:
        raise ReconstructionValidationError(f"I4.3G4B {implementation} returned no result and no error.")
    result = finalize_adaptive_provider_execution_result(local_result, prep)
    if result.status not in {DenseProviderRunStatus.COMPLETED, DenseProviderRunStatus.DEGRADED}:
        raise ReconstructionValidationError(
            f"I4.3G4B {implementation} returned non-terminal production status: {result.status.value} {result.failure_reason}"
        )
    summary_path = provider_out / _SUMMARY_FILENAMES[implementation]
    summary = _read_json(summary_path, label=f"I4.3G4B {implementation} E-series execution summary")
    selected = prep.plan.selected_view_count
    accepted, optional, optional_reasons = _alignment_metrics(summary, selected=selected)
    canonical_path = Path(result.canonical_geometry_ref).expanduser().resolve()
    point_count = _canonical_point_count(canonical_path)
    status = "degraded" if result.status is DenseProviderRunStatus.DEGRADED else "pass"
    artifact_paths = [summary_path, canonical_path, *[Path(value) for value in result.native_artifact_refs]]
    # Include per-view native evidence as cryptographically addressed evidence, not just the worker manifest.
    native_paths, worker_count = _worker_artifacts(provider_out, implementation=implementation)
    if worker_count != selected:
        raise ReconstructionValidationError(f"I4.3G4B {implementation} worker processed-view count drifted from adaptive plan.")
    artifact_paths.extend(native_paths)
    optional_reasons.update({
        "peak_host_memory_mib": "not_measured_by_existing_E_series_worker_or_G4B_device_sampler",
        "inter_view_consistency_ratio": "reserved_for_controlled_G4C_matched_view_diagnostic_not_inferred_from_production_run",
        "inter_view_median_relative_residual": "reserved_for_controlled_G4C_matched_view_diagnostic_not_inferred_from_production_run",
        "detached_component_fraction": "no_frozen_G4B_component_diagnostic_policy_authorized",
        "unsupported_geometry_fraction": "no_frozen_G4B_unsupported_geometry_diagnostic_policy_authorized",
    })
    evidence = _provider_evidence(
        implementation=implementation,
        binding=binding,
        attempt=attempt,
        plan_path=plan_path,
        plan=prep.plan,
        descriptor=descriptor,
        runtime_config=runtime_config,
        wall_time_s=wall,
        resource_evidence=monitor,
        processed_view_count=worker_count,
        aligned_view_fraction=accepted / float(selected),
        artifacts=_hash_artifacts(artifact_paths),
        status=status,
        optional_metrics=optional,
        optional_reasons=optional_reasons,
        canonical_point_count=point_count,
    )
    evidence_path = _write_json(output_dir / "evidence" / f"{implementation}_production_native.json", evidence)
    _verify_source_and_sfm_unchanged(binding, stage=f"after {implementation}")
    return {
        "evidence": _artifact(evidence_path),
        "status": status,
        "result": result.as_dict(),
        "resource_evidence": monitor,
    }


def main() -> int:
    args = _parser().parse_args()
    out = args.output_dir.expanduser().resolve()
    try:
        if out.exists() and any(out.iterdir()):
            raise ReconstructionValidationError(f"I4.3G4B output directory must be empty or absent: {out}")
        out.mkdir(parents=True, exist_ok=True)

        g4a_path = args.i4_3g4a_snapshot.expanduser().resolve()
        g4a_sha = _verify_sha(g4a_path, args.i4_3g4a_sha256, label="I4.3G4A snapshot")
        g4a = _read_json(g4a_path, label="I4.3G4A snapshot")
        validate_i4_3g4b_predecessor(g4a)
        predecessor_chain = _verify_g4a_predecessor_artifacts(g4a)
        binding = resolve_g4b_scenario_binding(
            i4_3g4a_snapshot=g4a,
            scenario_id=args.scenario,
            verify_files=True,
        )
        sfm_result = _verify_frozen_sfm(binding)
        provider_base_sfm = _registered_view_projection(sfm_result)
        storage_state = _require_storage_ready(g4a, stage="before scenario preflight")
        hardware, resolved_profile, resources = _adaptive_resources()
        probes = _live_runtime_probes(args)
        preflight = _runtime_preflight(
            g4a=g4a,
            binding=binding,
            sfm_result=provider_base_sfm,
            current_probes=probes,
            hardware=hardware,
            resolved_profile=resolved_profile,
            resources=resources,
            storage_state=storage_state,
        )
        preflight["frozen_sfm_input"] = {
            "materialized_input_keyframes": int(sfm_result.statistics.input_keyframes),
            "registered_keyframes": int(sfm_result.statistics.registered_keyframes),
            "registered_view_projection_used": provider_base_sfm is not sfm_result,
            "provider_base_input_keyframes": int(provider_base_sfm.statistics.input_keyframes),
            "camera_pose_authority_owner": "pycolmap_sfm",
            "authoritative_sfm_modified": False,
        }
        preflight["predecessors"] = {
            "i4_3g4a": {"path": str(g4a_path), "sha256": g4a_sha},
            **predecessor_chain,
            "g2b_reused_context": {
                "path": str(binding.g2b_production_snapshot_path),
                "sha256": binding.g2b_production_snapshot_sha256,
            },
            "authoritative_sfm": {
                "path": str(binding.authoritative_sfm_snapshot_path),
                "sha256": binding.authoritative_sfm_snapshot_sha256,
            },
        }
        preflight_path = _write_json(out / "i4_3g4b_runtime_preflight_snapshot.json", preflight)
        print("I4.3G4B runtime/source/frozen-SfM preflight: PASS")
        print(
            f"scenario={binding.scenario_id} source_images={binding.source_image_count} "
            f"registered_views={binding.registered_view_count} landmarks={binding.landmark_count}"
        )
        print(
            "runtimes_ready=6/6 plans_ready=6/6 frozen_sfm_reused=True sfm_rerun=False "
            f"hardware_profile={preflight['hardware']['resolved_profile']} precision={preflight['hardware']['resolved_precision']} "
            f"storage_free_gib={preflight['storage']['free_disk_gib']:.2f}"
        )
        print("provider_inference_started=False matched_view_started=False")
        print(f"preflight_snapshot={preflight_path}")
        if args.mode == "preflight":
            return 0

        provider_results: dict[str, Any] = {}
        provider_evidence_payloads: dict[str, Mapping[str, Any]] = {}
        provider_evidence_artifacts: dict[str, Mapping[str, Any]] = {}
        for index, implementation in enumerate(I43G4B_PROVIDER_IMPLEMENTATIONS, start=1):
            _require_storage_ready(g4a, stage=f"before provider {implementation}")
            print(f"[{index}/6] I4.3G4B executing {implementation} ...", flush=True)
            result = _execute_provider(
                implementation=implementation,
                binding=binding,
                sfm_result=provider_base_sfm,
                resources=resources,
                hardware=hardware,
                resolved_profile=resolved_profile,
                probe=probes[implementation],
                output_dir=out,
                attempt=int(args.attempt),
            )
            provider_results[implementation] = result
            evidence_artifact = result["evidence"]
            evidence_payload = _read_json(Path(evidence_artifact["path"]), label=f"{implementation} G4B evidence")
            validate_single_image_provider_run_evidence(evidence_payload)
            provider_evidence_payloads[implementation] = evidence_payload
            provider_evidence_artifacts[implementation] = evidence_artifact
            print(
                f"[{index}/6] {implementation}: {evidence_payload['status']} "
                f"selected={evidence_payload['metrics']['selected_view_count']} "
                f"processed={evidence_payload['metrics']['processed_view_count']} "
                f"aligned_fraction={evidence_payload['metrics']['aligned_view_fraction']:.4f}",
                flush=True,
            )

        _verify_source_and_sfm_unchanged(binding, stage="after all six providers")
        closure = build_g4b_scenario_snapshot(
            binding=binding,
            g4a_snapshot_path=g4a_path,
            g4a_snapshot_sha256=g4a_sha,
            provider_evidence=provider_evidence_payloads,
            provider_artifacts=provider_evidence_artifacts,
        )
        closure["preflight"] = _artifact(preflight_path)
        closure["provider_execution"] = provider_results
        closure["authoritative_sfm"] = {
            "path": str(binding.authoritative_sfm_snapshot_path),
            "sha256": binding.authoritative_sfm_snapshot_sha256,
            "reused_without_rerun": True,
            "registered_view_count": binding.registered_view_count,
            "landmark_count": binding.landmark_count,
            "same_frozen_sfm_shared_by_all_six_providers": True,
            "materialized_input_keyframes": int(sfm_result.statistics.input_keyframes),
            "registered_view_projection_used": provider_base_sfm is not sfm_result,
            "provider_base_input_keyframes": int(provider_base_sfm.statistics.input_keyframes),
        }
        closure_path = _write_json(out / "i4_3g4b_production_native_snapshot.json", closure)
        statuses = closure["production_statuses"]
        print(
            "I4.3G4B production-native scenario: PASS"
            + (" (provider quality rejection recorded)" if closure["accounting"]["provider_quality_rejection_present"] else "")
        )
        print(" ".join(f"{name}={statuses[name]}" for name in I43G4B_PROVIDER_IMPLEMENTATIONS))
        print("same_authoritative_sfm_shared=True provider_selected=False auto=False hybrid=False metric_scale=False metrology=False")
        print(f"production_snapshot={closure_path}")
        return 0
    except Exception as exc:  # noqa: BLE001 - CLI boundary persists deterministic failure evidence
        failure = {
            "schema_version": 1,
            "phase": "I4.3G4B",
            "status": "failed",
            "pass": False,
            "scenario_id": str(getattr(args, "scenario", "")),
            "attempt": int(getattr(args, "attempt", 1)),
            "error_type": type(exc).__name__,
            "error": str(exc),
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
        try:
            _write_json(out / "i4_3g4b_failure.json", failure)
        except Exception:
            pass
        print(f"I4.3G4B ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
