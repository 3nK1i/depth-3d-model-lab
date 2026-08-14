#!/usr/bin/env python3
"""Phase 9.2I combined semantic-workspace closure smoke.

This closure runner reuses the already-qualified Phase 9.2A-H non-UI smokes.
It adds no production behavior.  Its job is to prove that the final Phase 9.2
software state still preserves semantic-only authority while covering project
identity, automatic proposals, prompted refinement, deterministic merge,
reviewed-mask Measure & Inspect handoff, and project/workspace persistence.
"""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any

ROOT = Path(__file__).resolve().parents[1]

SMOKES = (
    ("semantic_workspace", "scripts/run_phase9_2a_semantic_workspace_smoke.py", ()),
    ("segmentation_project", "scripts/run_phase9_2b_semantic_project_smoke.py", ()),
    ("auto_segment", "scripts/run_phase9_2c_auto_segment_smoke.py", ()),
    ("prompt_refinement", "scripts/run_phase9_2e_prompt_smoke.py", ()),
    ("deterministic_merge", "scripts/run_phase9_2f_merge_smoke.py", ()),
    ("reviewed_measurement", "scripts/run_phase9_2g_reviewed_measurement_smoke.py", ()),
    ("workspace_persistence", "scripts/run_phase9_2h_workspace_persistence_smoke.py", ()),
)


def _run_json_smoke(relative_path: str, extra_args: tuple[str, ...]) -> dict[str, Any]:
    path = ROOT / relative_path
    proc = subprocess.run(
        [sys.executable, str(path), *extra_args],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        timeout=180,
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
    with tempfile.TemporaryDirectory(prefix="phase9_2i_") as tmp:
        tmp_root = Path(tmp)
        results: dict[str, dict[str, Any]] = {}
        for name, relative_path, base_args in SMOKES:
            args = list(base_args)
            if name == "auto_segment":
                args += ["--output-dir", str(tmp_root / "auto_segment")]
            elif name == "reviewed_measurement":
                args += ["--output-dir", str(tmp_root / "reviewed_measurement")]
            elif name == "workspace_persistence":
                args += ["--output-dir", str(tmp_root / "workspace_persistence")]
            results[name] = _run_json_smoke(relative_path, tuple(args))

    semantic = results["semantic_workspace"]
    project = results["segmentation_project"]
    auto = results["auto_segment"]
    prompt = results["prompt_refinement"]
    merge = results["deterministic_merge"]
    measurement = results["reviewed_measurement"]
    persistence = results["workspace_persistence"]

    checks = {
        "accepted_semantic_object_preserves_mask": (
            semantic["accepted_object_count"] == 1 and semantic["mask_preserved"] is True
        ),
        "class_and_instance_identity_are_separate": (
            semantic["assigned_class"] == "robot_arm"
            and semantic["instance_name"] == "robot_arm_001"
        ),
        "segmentation_project_round_trips_without_external_database": (
            project["external_database_required"] is False
            and project["renamed_class"] == "pallet_station"
        ),
        "automatic_proposals_are_unlabeled": (
            auto["kept_proposals"] > 0
            and auto["unlabeled_proposals"] == auto["kept_proposals"]
        ),
        "prompt_refinement_resets_review_to_pending": (
            prompt["created_parentless"] is True
            and prompt["positive_and_negative_prompts"] is True
            and prompt["review_state_after_refinement"] == "pending"
        ),
        "merge_is_deterministic_semantic_union": (
            merge["source_object_count"] == 2
            and merge["workspace_object_count_after_merge"] == 1
            and merge["mask_source"] == "semantic_merge"
            and merge["review_state"] == "pending"
        ),
        "reviewed_masks_bypass_perception_but_not_depth": (
            measurement["semantic_source"] == "reviewed_auto_segment"
            and measurement["perception_model_id"] == "reviewed-auto-segment"
            and measurement["perception_inference_skipped"] is True
            and measurement["depth_model_still_executes"] is True
        ),
        "workspace_persistence_resumes_reviewed_state": (
            persistence["workspace_persisted"] is True
            and persistence["workspace_resumed"] is True
            and persistence["accepted_instance"] == "plant_001"
            and persistence["measure_inspect_review_compatible"] is True
        ),
        "used_class_retirement_fails_safe": (
            persistence["used_class_returns_to_unlabeled"] is True
            and persistence["used_class_returns_to_pending"] is True
        ),
        "retired_class_ids_are_not_reused": persistence["retired_id_reused"] is False,
        "semantic_state_never_becomes_geometry_metric_or_measurement_authority": all(
            result.get("geometry_authority", result.get("reviewed_masks_are_geometry_authority", False)) is False
            and result.get("metric_authority", result.get("reviewed_masks_are_metric_authority", False)) is False
            and result.get("measurement_authority", result.get("reviewed_masks_are_measurement_authority", False)) is False
            for result in results.values()
        ),
    }
    failed = sorted(name for name, passed in checks.items() if not passed)
    if failed:
        raise RuntimeError(f"Phase 9.2 closure invariant failure(s): {failed}")

    output = {
        "status": "PASS",
        "phase": "9.2I",
        "checks": checks,
        "covered_substeps": ["9.2A", "9.2B", "9.2C", "9.2D", "9.2E", "9.2F", "9.2G", "9.2H"],
        "ui_qualification": {
            "phase_9_2d_and_later_ui": "covered by Phase 9.2 UI tests and prior real manual validation",
            "new_ui_in_9_2i": False,
        },
        "authority": {
            "semantic_geometry_authority": False,
            "semantic_metric_authority": False,
            "semantic_measurement_authority": False,
            "certified_metrology": False,
        },
    }
    print(json.dumps(output, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
