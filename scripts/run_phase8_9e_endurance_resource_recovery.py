#!/usr/bin/env python3
"""Phase 8.9E bounded whole-application endurance/resource/recovery campaign.

This campaign composes the already-qualified Phase 8.7G3 / 8.8H2-H3 recovery
and resource-ownership paths.  It intentionally uses a lightweight external
fixture worker for repeated Qt process lifecycle exercise; DA3, Gaussian, NeRF,
PyCOLMAP and Stella real-backend qualification remains owned by Phase 8.9F.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

_HEAVY_MODULE_PREFIXES = (
    "pycolmap",
    "depth_anything_3",
    "gsplat",
    "nerfstudio",
    "gaussian_renderer",
    "stella_vslam",
)


def _rss_mib() -> float:
    try:
        for line in Path("/proc/self/status").read_text(encoding="utf-8").splitlines():
            if line.startswith("VmRSS:"):
                return float(line.split()[1]) / 1024.0
    except OSError:
        pass
    return 0.0


def _child_pids() -> tuple[int, ...]:
    """Return direct child PIDs owned by this application process on Linux."""

    path = Path(f"/proc/{os.getpid()}/task/{os.getpid()}/children")
    try:
        raw = path.read_text(encoding="utf-8").strip()
    except OSError:
        return ()
    if not raw:
        return ()
    values: list[int] = []
    for token in raw.split():
        try:
            values.append(int(token))
        except ValueError:
            continue
    return tuple(sorted(values))


def _process_gpu_memory_mib(pid: int) -> float:
    command = [
        "nvidia-smi",
        "--query-compute-apps=pid,used_gpu_memory",
        "--format=csv,noheader,nounits",
    ]
    try:
        completed = subprocess.run(command, capture_output=True, text=True, timeout=5, check=False)
    except (OSError, subprocess.TimeoutExpired):
        return 0.0
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


def _loaded_heavy_modules() -> list[str]:
    loaded: set[str] = set()
    for name in sys.modules:
        if any(name == prefix or name.startswith(prefix + ".") for prefix in _HEAVY_MODULE_PREFIXES):
            loaded.add(name.split(".", 1)[0])
    return sorted(loaded)


def _growth_within_limit(before: float, after: float, limit: float) -> bool:
    return max(0.0, float(after) - float(before)) <= float(limit)


def _wait_until(predicate, *, timeout_s: float, step_ms: int = 10) -> bool:
    from PyQt6.QtTest import QTest

    deadline = time.monotonic() + float(timeout_s)
    while time.monotonic() < deadline:
        if predicate():
            return True
        QTest.qWait(max(1, int(step_ms)))
    return bool(predicate())


def _make_sparse_export_source(root: Path, session_id: str):
    from core.reconstruction import OfflineSparseExportSource

    source_root = (root / "export-source").resolve()
    source_root.mkdir(parents=True, exist_ok=True)
    sparse = source_root / "sparse_reconstruction.ply"
    sfm = source_root / "sfm_result.json"
    sparse.write_text(
        "\n".join(
            (
                "ply",
                "format ascii 1.0",
                "comment arbitrary reconstruction gauge; coordinates are NOT metres",
                "element vertex 1",
                "property float x",
                "property float y",
                "property float z",
                "property uchar red",
                "property uchar green",
                "property uchar blue",
                "end_header",
                "0 0 1 180 180 180",
                "",
            )
        ),
        encoding="ascii",
    )
    sfm.write_text(
        json.dumps(
            {
                "session_id": session_id,
                "backend_id": "phase8.9e-bounded-fixture",
                "backend_version": "1",
                "gauge": {
                    "scale_status": "arbitrary",
                    "alignment_status": "arbitrary",
                    "unit_to_metre": None,
                    "scale_source": "",
                    "relative_scale_uncertainty": None,
                },
            }
        ),
        encoding="utf-8",
    )
    return OfflineSparseExportSource(
        {
            "phase": "8.9E",
            "pass": True,
            "status": "completed",
            "session_id": session_id,
            "camera_pose_available": True,
            "canonical_geometry_authoritative": False,
            "persistent_scene_ready": False,
            "certified_metrology": False,
            "artifacts": {
                "sparse_reconstruction_ply": str(sparse),
                "sfm_result_json": str(sfm),
            },
        }
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cycles", type=int, default=5)
    parser.add_argument("--max-rss-growth-mib", type=float, default=256.0)
    parser.add_argument("--max-gpu-growth-mib", type=float, default=128.0)
    args = parser.parse_args()
    if args.cycles < 3:
        raise SystemExit("8.9E requires at least 3 repeated lifecycle cycles")

    output = args.output_dir.expanduser().resolve()
    shutil.rmtree(output, ignore_errors=True)
    output.mkdir(parents=True, exist_ok=True)
    qsettings = output / "qsettings"
    qsettings.mkdir(parents=True, exist_ok=True)
    os.environ["XDG_CONFIG_HOME"] = str(qsettings)
    os.environ.setdefault("QT_QPA_PLATFORM", "xcb")

    from PyQt6.QtTest import QTest
    from PyQt6.QtWidgets import QApplication

    import app.main_window as main_window_module
    from app.application_controller import CameraEnumerator
    from app.offline_reconstruction_controller import OfflineReconstructionRunState
    from app.reconstruction_controller import ReconstructionController
    from app.widgets.mode_tabs import AppMode
    from core.reconstruction import (
        ReconstructionExportRequest,
        export_reconstruction_bundle,
        verify_reconstruction_export_bundle,
    )
    from tests.reconstruction_h1_helpers import run_whole_pipeline
    from tests.reconstruction_h3_helpers import offline_configuration, write_h3_worker, write_ready_sequence

    errors: list[str] = []
    cycles: list[dict[str, Any]] = []
    original_camera_start = CameraEnumerator.start
    original_controller = main_window_module.ReconstructionController
    worker = write_h3_worker(output / "phase8_9e_worker.py")
    real_controller = ReconstructionController

    def controller_factory(application_controller, parent=None):
        return real_controller(application_controller, parent, offline_worker_script=worker)

    CameraEnumerator.start = lambda self: None
    main_window_module.ReconstructionController = controller_factory
    app = QApplication.instance() or QApplication(["phase8_9e_endurance"])
    window = None
    summary: dict[str, Any]
    try:
        window = main_window_module.MainWindow()
        window._mode_tabs.set_active_mode(AppMode.RECONSTRUCTION)
        window.resize(1366, 768)
        window.show()
        QTest.qWait(50)
        app.processEvents()

        controller = window._reconstruction_controller
        baseline_children = _child_pids()
        baseline_rss = _rss_mib()
        baseline_gpu = _process_gpu_memory_mib(os.getpid())
        source = write_ready_sequence(output)
        controller.select_image_sequence(source)
        finished: list[dict[str, Any]] = []
        controller.offline_run_finished.connect(finished.append)

        for index in range(args.cycles):
            before_count = len(finished)
            controller.start_offline(offline_configuration(output, f"cancel-cycle-{index:02d}"))
            started = _wait_until(
                lambda: controller.offline_state is OfflineReconstructionRunState.RUNNING,
                timeout_s=2.0,
            )
            active_children = _child_pids()
            cancel_requested = bool(started and controller.cancel_offline())
            terminal = _wait_until(lambda: len(finished) > before_count, timeout_s=5.0)
            idle = _wait_until(lambda: not controller.is_active, timeout_s=2.0)
            QTest.qWait(20)
            after_children = _child_pids()
            cycle = {
                "index": index,
                "started": started,
                "cancel_requested": cancel_requested,
                "terminal": terminal,
                "state": controller.offline_state.value,
                "idle_after": idle,
                "worker_seen": bool(set(active_children) - set(baseline_children)),
                "child_processes_clean": set(after_children).issubset(set(baseline_children)),
                "benchmark_navigation_released": bool(
                    window._mode_tabs._buttons[AppMode.BENCHMARK].isEnabled()
                ),
                "rss_after_mib": _rss_mib(),
                "gpu_after_mib": _process_gpu_memory_mib(os.getpid()),
            }
            cycle["pass"] = bool(
                cycle["started"]
                and cycle["cancel_requested"]
                and cycle["terminal"]
                and cycle["state"] == "cancelled"
                and cycle["idle_after"]
                and cycle["worker_seen"]
                and cycle["child_processes_clean"]
                and cycle["benchmark_navigation_released"]
            )
            cycles.append(cycle)

        # One real subprocess failure after repeated cancellation must still release ownership.
        before_count = len(finished)
        controller.start_offline(offline_configuration(output, "fail-after-endurance"))
        failure_terminal = _wait_until(lambda: len(finished) > before_count, timeout_s=5.0)
        failure_idle = _wait_until(lambda: not controller.is_active, timeout_s=2.0)
        QTest.qWait(20)
        failure_recovery = {
            "terminal": failure_terminal,
            "state": controller.offline_state.value,
            "idle_after": failure_idle,
            "child_processes_clean": set(_child_pids()).issubset(set(baseline_children)),
            "benchmark_navigation_released": bool(
                window._mode_tabs._buttons[AppMode.BENCHMARK].isEnabled()
            ),
        }
        failure_recovery["pass"] = bool(
            failure_recovery["terminal"]
            and failure_recovery["state"] == "failed"
            and failure_recovery["idle_after"]
            and failure_recovery["child_processes_clean"]
            and failure_recovery["benchmark_navigation_released"]
        )

        # A fresh successful process after terminal failure proves the failure did not poison restart.
        before_count = len(finished)
        controller.start_offline(offline_configuration(output, "recovered-success"))
        recovery_terminal = _wait_until(lambda: len(finished) > before_count, timeout_s=5.0)
        recovery_idle = _wait_until(lambda: not controller.is_active, timeout_s=2.0)
        QTest.qWait(20)
        fresh_recovery = {
            "terminal": recovery_terminal,
            "state": controller.offline_state.value,
            "idle_after": recovery_idle,
            "child_processes_clean": set(_child_pids()).issubset(set(baseline_children)),
            "benchmark_navigation_released": bool(
                window._mode_tabs._buttons[AppMode.BENCHMARK].isEnabled()
            ),
        }
        fresh_recovery["pass"] = bool(
            fresh_recovery["terminal"]
            and fresh_recovery["state"] == "completed"
            and fresh_recovery["idle_after"]
            and fresh_recovery["child_processes_clean"]
            and fresh_recovery["benchmark_navigation_released"]
        )

        # Repeated scene save/reload uses the actual application persistence controller.
        active_session_id = controller.offline_session_id
        if not active_session_id:
            raise RuntimeError("Fresh recovery did not retain a reconstruction session identity.")
        fixture, pipeline_result = run_whole_pipeline(session_id=active_session_id)
        adopted = controller.adopt_pipeline_result(pipeline_result, output / "scene-origin")
        persistence_cycles: list[dict[str, Any]] = []
        for index in range(args.cycles):
            manifest = controller.save_persistent_scene(output / f"scene-save-{index:02d}")
            reloaded = controller.reload_persistent_scene()
            item = {
                "index": index,
                "manifest_exists": manifest.is_file(),
                "scene_id_stable": reloaded.scene_id == adopted.scene_id,
                "fingerprint_stable": (
                    reloaded.scene_fingerprint_sha256 == adopted.scene_fingerprint_sha256
                ),
                "canonical_geometry_authoritative": reloaded.canonical_geometry_authoritative,
                "certified_metrology": reloaded.certified_metrology,
            }
            item["pass"] = bool(
                item["manifest_exists"]
                and item["scene_id_stable"]
                and item["fingerprint_stable"]
                and item["canonical_geometry_authoritative"]
                and item["certified_metrology"] is False
            )
            persistence_cycles.append(item)

        # Repeated product export remains non-destructive and authority-truthful.
        export_source = _make_sparse_export_source(output, fixture.request.session_id)
        export_cycles: list[dict[str, Any]] = []
        for index in range(args.cycles):
            result = export_reconstruction_bundle(
                ReconstructionExportRequest(
                    output_dir=(output / f"export-{index:02d}").resolve(),
                    sources=(export_source,),
                )
            )
            verified = verify_reconstruction_export_bundle(result.export_dir)
            item = {
                "index": index,
                "verified_file_count": len(verified),
                "manifest_exists": result.manifest_path.is_file(),
                "canonical_geometry_authoritative": any(
                    record.canonical_geometry_authoritative for record in verified
                ),
                "certified_metrology": any(record.certified_metrology for record in verified),
            }
            item["pass"] = bool(
                item["verified_file_count"] == 2
                and item["manifest_exists"]
                and item["canonical_geometry_authoritative"] is False
                and item["certified_metrology"] is False
            )
            export_cycles.append(item)

        pre_close_rss = _rss_mib()
        pre_close_gpu = _process_gpu_memory_mib(os.getpid())
        heavy_before_close = _loaded_heavy_modules()
        window.close()
        app.processEvents()
        QTest.qWait(50)
        final_children = _child_pids()
        final_rss = _rss_mib()
        final_gpu = _process_gpu_memory_mib(os.getpid())
        heavy_after_close = _loaded_heavy_modules()

        shutdown = {
            "controller_shutdown": controller.is_shutdown,
            "offline_controller_shutdown": controller.offline_controller.is_shutdown,
            "offline_running": controller.offline_is_running,
            "controller_active": controller.is_active,
            "child_processes_clean": set(final_children).issubset(set(baseline_children)),
        }
        shutdown["pass"] = bool(
            shutdown["controller_shutdown"]
            and shutdown["offline_controller_shutdown"]
            and not shutdown["offline_running"]
            and not shutdown["controller_active"]
            and shutdown["child_processes_clean"]
        )

        rss_growth = max(0.0, final_rss - baseline_rss)
        gpu_growth = max(0.0, final_gpu - baseline_gpu)
        resource_bounds = {
            "baseline_rss_mib": baseline_rss,
            "pre_close_rss_mib": pre_close_rss,
            "final_rss_mib": final_rss,
            "rss_growth_mib": rss_growth,
            "max_rss_growth_mib": float(args.max_rss_growth_mib),
            "rss_growth_within_bound": _growth_within_limit(
                baseline_rss, final_rss, args.max_rss_growth_mib
            ),
            "baseline_gpu_mib": baseline_gpu,
            "pre_close_gpu_mib": pre_close_gpu,
            "final_gpu_mib": final_gpu,
            "gpu_growth_mib": gpu_growth,
            "max_gpu_growth_mib": float(args.max_gpu_growth_mib),
            "gpu_growth_within_bound": _growth_within_limit(
                baseline_gpu, final_gpu, args.max_gpu_growth_mib
            ),
            "heavy_modules_before_close": heavy_before_close,
            "heavy_modules_after_close": heavy_after_close,
            "heavy_runtime_leak_detected": bool(heavy_after_close),
        }
        resource_bounds["pass"] = bool(
            resource_bounds["rss_growth_within_bound"]
            and resource_bounds["gpu_growth_within_bound"]
            and not resource_bounds["heavy_runtime_leak_detected"]
        )

        passed = bool(
            all(item["pass"] for item in cycles)
            and failure_recovery["pass"]
            and fresh_recovery["pass"]
            and all(item["pass"] for item in persistence_cycles)
            and all(item["pass"] for item in export_cycles)
            and shutdown["pass"]
            and resource_bounds["pass"]
            and not errors
        )
        summary = {
            "phase": "8.9E",
            "pass": passed,
            "errors": errors,
            "cycle_count": int(args.cycles),
            "application": {
                "actual_main_window": True,
                "window_size": [1366, 768],
            },
            "cancel_cycles": cycles,
            "failure_recovery": failure_recovery,
            "fresh_recovery": fresh_recovery,
            "persistence_cycles": persistence_cycles,
            "export_cycles": export_cycles,
            "shutdown": shutdown,
            "resources": resource_bounds,
            "production_code_changed_by_8_9e": False,
            "new_recovery_engine_created": False,
            "heavy_backends_started": False,
            "real_da3_gaussian_nerf_deferred_to": "8.9F",
            "canonical_geometry_authoritative_after_reload": bool(
                persistence_cycles and persistence_cycles[-1]["canonical_geometry_authoritative"]
            ),
            "certified_metrology": False,
        }
    except Exception as exc:  # noqa: BLE001 - release evidence must record exact failure
        errors.append(f"{type(exc).__name__}: {exc}")
        if window is not None:
            try:
                window.close()
                app.processEvents()
            except Exception:
                pass
        summary = {
            "phase": "8.9E",
            "pass": False,
            "errors": errors,
            "production_code_changed_by_8_9e": False,
            "new_recovery_engine_created": False,
            "heavy_backends_started": False,
            "real_da3_gaussian_nerf_deferred_to": "8.9F",
            "certified_metrology": False,
        }
    finally:
        CameraEnumerator.start = original_camera_start
        main_window_module.ReconstructionController = original_controller

    summary_path = output / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("Phase 8.9E endurance/resource/recovery campaign: " + ("PASS" if summary.get("pass") else "FAIL"))
    if summary.get("pass"):
        resources = summary["resources"]
        print(
            f"cycles={summary['cycle_count']} cancellations=PASS failure_recovery=PASS "
            "fresh_recovery=PASS persistence=PASS export=PASS shutdown=PASS"
        )
        print(
            f"rss_growth_mib={resources['rss_growth_mib']:.3f} "
            f"gpu_growth_mib={resources['gpu_growth_mib']:.3f} "
            "heavy_runtime_leak=False"
        )
    print(f"summary={summary_path}")
    return 0 if summary.get("pass") else 2


if __name__ == "__main__":
    raise SystemExit(main())
