#!/usr/bin/env python3
"""Phase 9.1I combined calibration/metric authority closure smoke.

This runner intentionally reuses the qualified Phase 9.1D-H non-UI smokes
instead of duplicating production logic.  It verifies the end-state authority
invariants needed to close Phase 9.1 while keeping real physical-camera and
real persistent-scene field validation explicitly deferred.
"""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]

SMOKES = (
    ("automatic_calibration", "scripts/run_phase9_1d_automatic_calibration_smoke.py"),
    ("metric_qualification", "scripts/run_phase9_1e_metric_qualification_smoke.py"),
    ("measurement_calibration", "scripts/run_phase9_1g_measurement_calibration_smoke.py"),
    ("authority_persistence", "scripts/run_phase9_1h_persistence_smoke.py"),
)


def _run_json_smoke(relative_path: str) -> dict[str, Any]:
    path = ROOT / relative_path
    proc = subprocess.run(
        [sys.executable, str(path)],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        timeout=120,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"Closure smoke failed: {relative_path}\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
        )
    try:
        payload = json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"Closure smoke did not emit JSON: {relative_path}\n{proc.stdout}"
        ) from exc
    if payload.get("status") != "PASS":
        raise RuntimeError(f"Closure smoke did not pass: {relative_path}: {payload}")
    return payload


def main() -> int:
    results = {name: _run_json_smoke(path) for name, path in SMOKES}

    automatic = results["automatic_calibration"]
    metric = results["metric_qualification"]
    measurement = results["measurement_calibration"]
    persistence = results["authority_persistence"]

    checks = {
        "checkerboard_available": bool(automatic["capabilities"]["checkerboard"]),
        "charuco_available": bool(automatic["capabilities"]["charuco"]),
        "automatic_calibration_does_not_create_metric_scale": (
            automatic["metric_status_without_scale_reference"] == "arbitrary"
            and automatic["metric_coordinates_allowed_without_scale_reference"] is False
        ),
        "validated_reference_target_can_qualify_metric_scale": (
            metric["reference_target"]["qualification_state"] == "qualified"
            and metric["reference_target"]["calibrated_metric_scale"] is True
        ),
        "known_size_benchmark_is_supporting_only": (
            metric["known_size_benchmark"]["qualification_state"] == "supporting_only"
            and metric["known_size_benchmark"]["metric_coordinates_allowed"] is False
        ),
        "future_rgbd_fails_closed": (
            metric["future_rgbd"]["qualification_state"] == "unsupported"
            and metric["future_rgbd"]["metric_coordinates_allowed"] is False
        ),
        "measure_and_inspect_receives_calibrated_intrinsics": (
            measurement["native_depth_intrinsics_source"] == "calibrated"
            and measurement["native_depth_calibration_status"] == "calibrated"
        ),
        "camera_calibration_preserves_depth_scale_semantics": (
            measurement["depth_scale_preserved"] == "relative"
            and measurement["metric_coordinates_from_calibration_alone"] is False
        ),
        "phase8_scene_schema_is_preserved": persistence["scene_schema_version_preserved"] == 1,
        "legacy_scene_without_phase9_sidecar_is_supported": bool(
            persistence["legacy_scene_without_sidecar"]
        ),
        "persisted_authority_is_self_contained": (
            persistence["external_profile_file_required"] is False
        ),
        "certified_metrology_remains_false": all(
            result.get("certified_metrology") is False for result in results.values()
        ),
    }
    failed = sorted(name for name, passed in checks.items() if not passed)
    if failed:
        raise RuntimeError(f"Phase 9.1 closure invariant failure(s): {failed}")

    output = {
        "status": "PASS",
        "phase": "9.1I",
        "checks": checks,
        "deferred_real_validation": {
            "physical_camera_calibration_and_measure_inspect": True,
            "real_persistent_scene_save_restart_reload": True,
            "reason": "No suitable real calibration/profile and persistent scene are currently available.",
        },
        "certified_metrology": False,
    }
    print(json.dumps(output, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
