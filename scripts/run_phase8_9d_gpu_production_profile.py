#!/usr/bin/env python3
"""Phase 8.9D whole-application GPU production profiling.

Real measurements are collected only from the GPU physically present on the
machine.  Synthetic RTX 2060/4090/5090 entries are readiness contracts for the
existing hardware-adaptive policies and MUST NOT be presented as measurements.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.hardware import AcceleratorBackend, ComputeProfile, HardwareInfo, detect_hardware, resolve_profile  # noqa: E402
from core.reconstruction import resolve_reconstruction_profile  # noqa: E402

_GIB = 1024**3
_HEAVY_MODULE_PREFIXES = (
    "pycolmap",
    "depth_anything_3",
    "gsplat",
    "nerfstudio",
    "gaussian_renderer",
    "stella_vslam",
)


def _synthetic_hardware(target: str) -> HardwareInfo:
    presets: dict[str, tuple[str, float, float, tuple[int, int]]] = {
        "rtx2060": ("NVIDIA GeForce RTX 2060", 6.0, 5.2, (7, 5)),
        "rtx4090": ("NVIDIA GeForce RTX 4090", 24.0, 22.0, (8, 9)),
        "rtx5090": ("NVIDIA GeForce RTX 5090", 32.0, 30.0, (12, 0)),
    }
    try:
        name, total, free, capability = presets[target]
    except KeyError as exc:
        raise ValueError(f"Unknown readiness target: {target}") from exc
    return HardwareInfo(
        backend=AcceleratorBackend.CUDA,
        device="cuda:0",
        device_index=0,
        name=name,
        total_memory_bytes=int(total * _GIB),
        free_memory_bytes=int(free * _GIB),
        compute_capability=capability,
        supports_fp16=True,
        supports_bf16=capability[0] >= 8,
        torch_available=True,
        torch_cuda_available=True,
        torch_version="readiness-only",
        torch_cuda_version="readiness-only",
        compiled_architectures=(f"sm_{capability[0]}{capability[1]}",),
        torch_build_compatible=True,
    )


def _policy_report(hardware: HardwareInfo, *, evidence: str) -> dict[str, Any]:
    compute = resolve_profile(ComputeProfile.AUTO, hardware)
    reconstruction = resolve_reconstruction_profile(hardware)
    return {
        "evidence": evidence,
        "measured": evidence == "real_hardware_measurement",
        "hardware": {
            "name": hardware.name,
            "device": hardware.device,
            "total_vram_gib": hardware.total_memory_gib,
            "free_vram_gib": hardware.free_memory_gib,
            "compute_capability": list(hardware.compute_capability) if hardware.compute_capability else None,
            "supports_fp16": hardware.supports_fp16,
            "supports_bf16": hardware.supports_bf16,
            "torch_build_compatible": hardware.torch_build_compatible,
        },
        "automatic_compute_profile": {
            "effective": compute.effective.value,
            "device": compute.device,
            "precision": compute.precision.value,
            "model_repo": compute.spec.model_repo,
            "point_budget": compute.spec.point_budget,
            "live_point_budget": compute.spec.live_point_budget,
            "downgraded": compute.downgraded,
            "memory_pressure": compute.memory_pressure,
        },
        "automatic_reconstruction_profile": {
            "effective": reconstruction.effective.value,
            "device": reconstruction.memory_policy.device,
            "gpu_budget_gib": reconstruction.memory_policy.gpu_budget_bytes / _GIB,
            "gpu_reserve_gib": reconstruction.memory_policy.gpu_reserve_bytes / _GIB,
            "max_active_keyframes": reconstruction.memory_policy.max_active_keyframes,
            "downgraded": reconstruction.downgraded,
            "memory_pressure": reconstruction.memory_pressure,
        },
    }


def readiness_matrix() -> dict[str, Any]:
    return {
        target: _policy_report(_synthetic_hardware(target), evidence="policy_readiness_only_not_measured")
        for target in ("rtx2060", "rtx4090", "rtx5090")
    }


def _run_nvidia_smi(fields: str) -> list[list[str]]:
    command = [
        "nvidia-smi",
        f"--query-gpu={fields}",
        "--format=csv,noheader,nounits",
    ]
    completed = subprocess.run(command, capture_output=True, text=True, timeout=5, check=False)
    if completed.returncode != 0:
        return []
    rows: list[list[str]] = []
    for line in completed.stdout.splitlines():
        if line.strip():
            rows.append([part.strip() for part in line.split(",")])
    return rows


def _gpu_snapshot(device_index: int | None) -> dict[str, Any]:
    fields = "index,name,uuid,driver_version,memory.total,memory.free,memory.used,utilization.gpu,temperature.gpu,power.draw"
    rows = _run_nvidia_smi(fields)
    if not rows:
        return {"available": False}
    wanted = 0 if device_index is None else int(device_index)
    row = next((item for item in rows if item and item[0] == str(wanted)), rows[0])
    keys = (
        "index",
        "name",
        "uuid",
        "driver_version",
        "memory_total_mib",
        "memory_free_mib",
        "memory_used_mib",
        "utilization_gpu_percent",
        "temperature_c",
        "power_draw_w",
    )
    return {"available": True, **dict(zip(keys, row, strict=False))}


def _process_gpu_memory_mib(pid: int) -> float:
    command = [
        "nvidia-smi",
        "--query-compute-apps=pid,used_gpu_memory",
        "--format=csv,noheader,nounits",
    ]
    completed = subprocess.run(command, capture_output=True, text=True, timeout=5, check=False)
    if completed.returncode != 0:
        return 0.0
    total = 0.0
    for line in completed.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 2 or parts[0] != str(pid):
            continue
        try:
            total += float(parts[1])
        except ValueError:
            continue
    return total


def _rss_mib() -> float:
    try:
        for line in Path("/proc/self/status").read_text(encoding="utf-8").splitlines():
            if line.startswith("VmRSS:"):
                return float(line.split()[1]) / 1024.0
    except OSError:
        pass
    return 0.0


def _loaded_heavy_modules() -> list[str]:
    loaded: set[str] = set()
    for name in sys.modules:
        if any(name == prefix or name.startswith(prefix + ".") for prefix in _HEAVY_MODULE_PREFIXES):
            loaded.add(name.split(".", 1)[0])
    return sorted(loaded)


def _probe_application_shell(*, duration_seconds: float, output_dir: Path) -> dict[str, Any]:
    # Keep the profiling probe from mutating the operator's persisted Qt layout.
    qsettings_dir = output_dir / "qsettings"
    qsettings_dir.mkdir(parents=True, exist_ok=True)
    os.environ["XDG_CONFIG_HOME"] = str(qsettings_dir)

    from PyQt6.QtWidgets import QApplication

    from app.main_window import MainWindow
    from app.settings import APP_NAME as SETTINGS_APP_NAME, ORG_NAME
    from app.theme import apply_theme
    from core.version import __version__

    QApplication.setOrganizationName(ORG_NAME)
    QApplication.setApplicationName(SETTINGS_APP_NAME)
    QApplication.setApplicationVersion(__version__)
    existing = QApplication.instance()
    owns_app = existing is None
    app = existing or QApplication(["phase8_9d_gpu_profile"])
    apply_theme(app)

    before_gpu_mib = _process_gpu_memory_mib(os.getpid())
    before_rss_mib = _rss_mib()
    startup_start = time.perf_counter()
    window = MainWindow()
    window.resize(1366, 768)
    window.show()
    app.processEvents()
    startup_seconds = time.perf_counter() - startup_start

    deadline = time.monotonic() + max(0.5, duration_seconds)
    rss_samples: list[float] = []
    gpu_samples: list[float] = []
    while time.monotonic() < deadline:
        app.processEvents()
        rss_samples.append(_rss_mib())
        gpu_samples.append(_process_gpu_memory_mib(os.getpid()))
        time.sleep(0.1)

    shutdown_start = time.perf_counter()
    window.close()
    app.processEvents()
    shutdown_seconds = time.perf_counter() - shutdown_start
    after_gpu_mib = _process_gpu_memory_mib(os.getpid())
    after_rss_mib = _rss_mib()
    if owns_app:
        app.quit()

    return {
        "executed": True,
        "actual_main_window": True,
        "window_size": [1366, 768],
        "duration_seconds": duration_seconds,
        "startup_seconds": startup_seconds,
        "shutdown_seconds": shutdown_seconds,
        "rss_before_mib": before_rss_mib,
        "rss_peak_mib": max(rss_samples or [before_rss_mib]),
        "rss_after_close_mib": after_rss_mib,
        "process_gpu_memory_before_mib": before_gpu_mib,
        "process_gpu_memory_peak_mib": max(gpu_samples or [before_gpu_mib]),
        "process_gpu_memory_after_close_mib": after_gpu_mib,
        "heavy_modules_loaded": _loaded_heavy_modules(),
        "heavy_runtime_leak_detected": bool(_loaded_heavy_modules()),
    }


def run_real_profile(*, output_dir: Path, with_ui: bool, duration_seconds: float) -> dict[str, Any]:
    detect_hardware.cache_clear()
    hardware = detect_hardware()
    if not hardware.has_cuda_device or not hardware.cuda_ready:
        raise RuntimeError("8.9D real profiling requires a CUDA-ready GPU in the production environment.")

    before = _gpu_snapshot(hardware.device_index)
    result: dict[str, Any] = {
        "phase": "8.9D",
        "pass": False,
        "measurement_scope": "physically_present_gpu_only",
        "real_hardware": _policy_report(hardware, evidence="real_hardware_measurement"),
        "readiness_matrix": readiness_matrix(),
        "gpu_snapshot_before": before,
        "application_shell": {"executed": False},
        "claims": {
            "rtx4090_measured": "4090" in hardware.name,
            "rtx5090_measured": "5090" in hardware.name,
            "readiness_matrix_is_measurement": False,
        },
        "errors": [],
    }
    try:
        if with_ui:
            result["application_shell"] = _probe_application_shell(
                duration_seconds=duration_seconds,
                output_dir=output_dir,
            )
        result["gpu_snapshot_after"] = _gpu_snapshot(hardware.device_index)
        shell = result["application_shell"]
        shell_ok = (not with_ui) or (
            isinstance(shell, dict)
            and shell.get("executed") is True
            and shell.get("actual_main_window") is True
            and shell.get("heavy_runtime_leak_detected") is False
        )
        result["pass"] = bool(shell_ok)
    except Exception as exc:  # noqa: BLE001 - release profiler records exact failure
        result["errors"].append(f"{type(exc).__name__}: {exc}")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--with-ui", action="store_true", help="Instantiate and close the actual MainWindow at 1366x768.")
    parser.add_argument("--duration-seconds", type=float, default=3.0)
    parser.add_argument(
        "--policy-readiness-only",
        action="store_true",
        help="Emit deterministic 2060/4090/5090 policy readiness without claiming measurements.",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / "summary.json"

    if args.policy_readiness_only:
        summary = {
            "phase": "8.9D",
            "pass": True,
            "measurement_scope": "none_policy_readiness_only",
            "readiness_matrix": readiness_matrix(),
            "claims": {
                "rtx2060_measured": False,
                "rtx4090_measured": False,
                "rtx5090_measured": False,
            },
        }
    else:
        summary = run_real_profile(
            output_dir=args.output_dir,
            with_ui=args.with_ui,
            duration_seconds=max(0.5, float(args.duration_seconds)),
        )

    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Phase 8.9D GPU production profile: {'PASS' if summary.get('pass') else 'FAIL'}")
    print(f"summary={summary_path}")
    return 0 if summary.get("pass") else 1


if __name__ == "__main__":
    raise SystemExit(main())
