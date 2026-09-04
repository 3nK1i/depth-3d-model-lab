#!/usr/bin/env python3
"""Run the I5.5B frozen control vs controlled dynamic-foreground replay."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.reconstruction.errors import ReconstructionValidationError
from core.reconstruction.i5_difficult_handheld_robustness import I51EvidenceBinding
from core.reconstruction.i5_dynamic_scene_handling import (
    analyze_i5_5b_mapping_no_sparse_model_rejection,
    analyze_i5_5b_materialized_pair,
    analyze_i5_5b_pre_sfm_ingestion_rejection,
    analyze_i5_5b_sfm_report,
    build_i5_5a_source_identity,
    build_i5_5b_dynamic_scene_replay,
    compare_i5_5b_sfm_outcomes,
    materialize_i5_5b_pair,
    validate_i5_5a_for_i5_5b,
    validate_i5_5b_receipt,
)

OUTPUT_NAME = "i5_5b_dynamic_scene_replay.json"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run I5.5B current frozen-foundation dynamic-scene replay.")
    parser.add_argument("--i5-5a-admission", type=Path, required=True)
    parser.add_argument("--i5-5a-sha256", required=True)
    parser.add_argument("--replay-receipt", type=Path, required=True)
    parser.add_argument("--replay-receipt-sha256", required=True)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sfm-python", required=True)
    parser.add_argument("--sfm-timeout-minutes", type=float, default=60.0)
    parser.add_argument("--maximum-image-size", type=int)
    parser.add_argument("--cpu", action="store_true")
    return parser


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _valid_sha256(value: str) -> bool:
    value = str(value).strip().lower()
    return len(value) == 64 and all(ch in "0123456789abcdef" for ch in value)


def _binding(path: Path, sha256: str, *, label: str) -> I51EvidenceBinding:
    resolved = path.expanduser().resolve()
    expected = str(sha256).strip().lower()
    if not resolved.is_file():
        raise ReconstructionValidationError(f"I5.5B {label} file missing: {resolved}")
    if not _valid_sha256(expected):
        raise ReconstructionValidationError(f"I5.5B {label} SHA256 is malformed.")
    actual = _sha256(resolved)
    if actual != expected:
        raise ReconstructionValidationError(f"I5.5B {label} SHA256 mismatch: expected {expected}, found {actual}")
    return I51EvidenceBinding(str(resolved), expected)


def _read_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ReconstructionValidationError(f"I5.5B JSON root must be an object: {path}")
    return data


def _write_json(path: Path, payload: Mapping[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    return path


def _verify_stage(stage: Mapping[str, Any]) -> bool:
    directory = Path(str(stage.get("directory", "")))
    files = stage.get("files")
    if not directory.is_dir() or not isinstance(files, list):
        return False
    for raw in files:
        if not isinstance(raw, dict):
            return False
        path = directory / str(raw.get("staged_name", ""))
        if not path.is_file() or _sha256(path) != str(raw.get("staged_sha256", "")):
            return False
    return True


def _sparse_component_count(output_dir: Path) -> int:
    root = output_dir / "workspace" / "sparse"
    if not root.is_dir():
        return 0
    count = 0
    for child in root.iterdir():
        if child.is_dir() and all((child / name).is_file() for name in ("cameras.bin", "images.bin", "points3D.bin")):
            count += 1
    return count


def _run_sfm(
    *,
    scenario_label: str,
    images_dir: Path,
    output_dir: Path,
    sfm_python: str,
    timeout_minutes: float,
    maximum_image_size: int | None,
    cpu: bool,
) -> tuple[int, Path | None, Path, list[str], str | None, str | None, int]:
    runner = ROOT / "scripts/run_reconstruction_pycolmap_smoke.py"
    command = [
        sys.executable,
        str(runner),
        "--images", str(images_dir),
        "--sfm-python", str(Path(sfm_python).expanduser().resolve()),
        "--output-dir", str(output_dir),
        "--profile", "efficient",
        "--matching", "exhaustive",
        "--max-keyframes", "19",
        "--image-sequence-fps", "10.0",
        "--timeout-minutes", str(float(timeout_minutes)),
    ]
    if maximum_image_size is not None:
        command += ["--maximum-image-size", str(int(maximum_image_size))]
    if cpu:
        command.append("--cpu")

    log_path = output_dir.parent / f"{scenario_label}.log"
    output_dir.mkdir(parents=True, exist_ok=True)
    print("I5.5B command: " + " ".join(command), flush=True)
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            cwd=str(ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
            log.flush()
        rc = int(process.wait())

    report_path = output_dir / "real_sfm_report.json"
    component_count = _sparse_component_count(output_dir)
    if report_path.is_file():
        if rc not in (0, 1):
            raise ReconstructionValidationError(f"I5.5B {scenario_label} produced a report with unexpected rc={rc}.")
        if component_count < 1:
            raise ReconstructionValidationError(f"I5.5B {scenario_label} produced a report but no complete sparse-model component.")
        return rc, report_path, log_path, command, None, None, component_count

    log_text = log_path.read_text(encoding="utf-8", errors="replace")
    ingestion_marker = "Fewer than three valid keyframes survived ingestion."
    mapping_marker = "PyCOLMAP incremental mapping produced no sparse model"
    if rc == 1 and ingestion_marker in log_text:
        reason = next((line.strip() for line in log_text.splitlines() if ingestion_marker in line), ingestion_marker)
        return rc, None, log_path, command, "ingestion", reason, 0
    if rc == 1 and mapping_marker in log_text:
        reason = next((line.strip() for line in log_text.splitlines() if mapping_marker in line), mapping_marker)
        return rc, None, log_path, command, "incremental_mapping", reason, 0
    raise ReconstructionValidationError(
        f"I5.5B {scenario_label} did not produce structured real-SfM evidence and was not a recognized "
        f"ingestion/mapping rejection (rc={rc}, report=False)."
    )


def run(args: argparse.Namespace) -> Path:
    if args.sfm_timeout_minutes <= 0:
        raise ReconstructionValidationError("I5.5B --sfm-timeout-minutes must be positive.")
    if args.maximum_image_size is not None and args.maximum_image_size <= 0:
        raise ReconstructionValidationError("I5.5B --maximum-image-size must be positive.")

    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    admission_binding = _binding(args.i5_5a_admission, args.i5_5a_sha256, label="I5.5A admission")
    receipt_binding = _binding(args.replay_receipt, args.replay_receipt_sha256, label="replay receipt")
    admission = validate_i5_5a_for_i5_5b(_read_json(Path(admission_binding.path)))
    receipt = validate_i5_5b_receipt(_read_json(Path(receipt_binding.path)))

    dataset = args.dataset_dir.expanduser().resolve()
    if not dataset.is_dir():
        raise ReconstructionValidationError(f"I5.5B dataset directory missing: {dataset}")
    before = build_i5_5a_source_identity(dataset)
    frozen = admission["source_identity"]
    if before["asset_set_sha256"] != frozen["asset_set_sha256"] or before["ordered_names"] != frozen["ordered_names"]:
        raise ReconstructionValidationError("I5.5B source identity drifted from frozen I5.5A admission before replay.")

    materialized = materialize_i5_5b_pair(dataset_dir=dataset, admission=admission, output_dir=output / "inputs")
    materialization_revalidation = analyze_i5_5b_materialized_pair(materialized)

    plan = admission["stress_plan"]
    scenario_specs = (
        ("control", "static_control", Path(materialized["control"]["directory"]), output / "01_control"),
        ("dynamic_proxy", "dynamic_foreground_proxy", Path(materialized["dynamic_proxy"]["directory"]), output / "02_dynamic_proxy"),
    )
    analyses: dict[str, Any] = {}
    execution: dict[str, Any] = {}
    staged_names = [str(row["staged_name"]) for row in materialized["control"]["files"]]

    for output_key, plan_key, images_dir, sfm_out in scenario_specs:
        rc, report_path, log_path, command, rejection_stage, rejection_reason, component_count = _run_sfm(
            scenario_label=output_key,
            images_dir=images_dir,
            output_dir=sfm_out,
            sfm_python=args.sfm_python,
            timeout_minutes=args.sfm_timeout_minutes,
            maximum_image_size=args.maximum_image_size,
            cpu=args.cpu,
        )
        if report_path is not None:
            analyses[output_key] = analyze_i5_5b_sfm_report(
                _read_json(report_path),
                scenario_plan=plan[plan_key],
                staged_names=staged_names,
                sparse_component_count=component_count,
            )
            report_value: str | None = str(report_path.resolve())
            report_sha: str | None = _sha256(report_path)
        elif rejection_stage == "ingestion":
            assert rejection_reason is not None
            analyses[output_key] = analyze_i5_5b_pre_sfm_ingestion_rejection(
                scenario_plan=plan[plan_key], staged_names=staged_names, reason=rejection_reason
            )
            report_value = None; report_sha = None
        elif rejection_stage == "incremental_mapping":
            assert rejection_reason is not None
            analyses[output_key] = analyze_i5_5b_mapping_no_sparse_model_rejection(
                scenario_plan=plan[plan_key], staged_names=staged_names, reason=rejection_reason
            )
            report_value = None; report_sha = None
        else:
            raise ReconstructionValidationError(f"I5.5B {output_key} has no structured outcome.")
        execution[output_key] = {
            "return_code": rc,
            "command": command,
            "report": report_value,
            "report_sha256": report_sha,
            "log": str(log_path.resolve()),
            "log_sha256": _sha256(log_path),
            "structured_evidence_complete": True,
            "rejection_stage": rejection_stage,
            "rejection_reason": rejection_reason,
            "sparse_component_count": component_count,
        }

    if not _verify_stage(materialized["control"]) or not _verify_stage(materialized["dynamic_proxy"]):
        raise ReconstructionValidationError("I5.5B staged input bytes changed during real replay.")
    materialization_after = analyze_i5_5b_materialized_pair(materialized)
    if materialization_after != materialization_revalidation:
        raise ReconstructionValidationError("I5.5B materialized-pair diagnostics changed during real replay.")
    after = build_i5_5a_source_identity(dataset)
    source_unchanged = after["asset_set_sha256"] == before["asset_set_sha256"] and after["ordered_names"] == before["ordered_names"]
    if not source_unchanged:
        raise ReconstructionValidationError("I5.5B source identity changed during real replay.")

    comparison = compare_i5_5b_sfm_outcomes(
        control=analyses["control"],
        dynamic_proxy=analyses["dynamic_proxy"],
        materialization_revalidation=materialization_revalidation,
    )
    result = build_i5_5b_dynamic_scene_replay(
        i5_5a_result=admission,
        i5_5a_binding=admission_binding,
        replay_receipt=receipt,
        replay_receipt_binding=receipt_binding,
        materialization_revalidation=materialization_revalidation,
        control_analysis=analyses["control"],
        dynamic_proxy_analysis=analyses["dynamic_proxy"],
        paired_comparison=comparison,
        source_reverified_before=True,
        source_reverified_after=True,
        staged_inputs_reverified=True,
    )
    result["execution_evidence"] = execution
    result["materialization_evidence"] = materialized
    result["source_reverification"] = {
        "before_asset_set_sha256": before["asset_set_sha256"],
        "after_asset_set_sha256": after["asset_set_sha256"],
        "unchanged": True,
    }
    evidence_path = _write_json(output / OUTPUT_NAME, result)

    c = analyses["control"]; s = analyses["dynamic_proxy"]
    q = result["qualification"]
    print("I5.5B Current Frozen-Foundation Dynamic-Scene Replay: EVIDENCE_COMPLETE")
    print(
        f"quality_status={result['quality_status']} "
        f"control={c['registered_images']}/{c['input_images']}({c['quality_status']}) "
        f"dynamic_proxy={s['registered_images']}/{s['input_images']}({s['quality_status']})"
    )
    print(
        f"dynamic_proxy_revalidated={q['controlled_dynamic_foreground_proxy_revalidated']} "
        f"feature_bearing={materialization_revalidation['dynamic_region_feature_bearing_revalidated']} "
        f"donor_orb={materialization_revalidation['donor_patch_orb_keypoints']} "
        f"stress_target_orb_min={materialization_revalidation['stress_target_orb_keypoints_min']}"
    )
    print(
        f"components control={c['sparse_component_count']} dynamic_proxy={s['sparse_component_count']} "
        f"landmarks control={comparison['landmarks_control']} dynamic_proxy={comparison['landmarks_dynamic_proxy']} "
        f"observations control={comparison['observations_control']} dynamic_proxy={comparison['observations_dynamic_proxy']}"
    )
    print(
        f"reprojection_px control={comparison['mean_reprojection_error_px_control']} "
        f"dynamic_proxy={comparison['mean_reprojection_error_px_dynamic_proxy']} "
        f"mean_track_length control={comparison['mean_track_length_control']} dynamic_proxy={comparison['mean_track_length_dynamic_proxy']}"
    )
    print(
        "natural_dynamic_scene_ground_truth=False cross_run_absolute_geometry_comparison=False "
        "pycolmap_authority=True threshold_relaxation=False production_default_modified=False "
        "dynamic_masking=False semantic_segmentation=False motion_outlier_filter=False dense_provider_rerun=False "
        "mp_sfm=False auto=False hybrid=False metric_scale=False metrology=False permanent_global_view_cap=None"
    )
    print(f"dynamic_scene_conclusion={q['dynamic_scene_robustness_conclusion']}")
    print("I5.5C dynamic-scene handling robustness qualification and closure: AUTHORIZED")
    print(f"evidence={evidence_path}")
    return evidence_path


def main() -> int:
    try:
        run(_parser().parse_args())
    except (ReconstructionValidationError, OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
        print(f"I5.5B ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
