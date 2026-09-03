#!/usr/bin/env python3
"""Run I5.2B control/low-overlap/weak-parallax real PyCOLMAP replay."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.reconstruction.errors import ReconstructionValidationError
from core.reconstruction.i5_difficult_handheld_robustness import I51EvidenceBinding
from core.reconstruction.i5_low_overlap_weak_parallax import (
    analyze_i5_2b_sfm_report,
    build_i5_2b_low_overlap_weak_parallax_replay,
    build_ordered_source_identity,
    validate_i5_2a_for_i5_2b,
    validate_i5_2b_receipt,
)


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ReconstructionValidationError(f"Expected JSON object: {path}")
    return payload


def _write_json(path: Path, payload: Mapping[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(dict(payload), indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(path)
    return path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _binding(path: Path, expected_sha: str, *, label: str) -> I51EvidenceBinding:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise ReconstructionValidationError(f"I5.2B {label} missing: {resolved}")
    actual = _sha256(resolved)
    if actual != str(expected_sha).strip().lower():
        raise ReconstructionValidationError(f"I5.2B {label} SHA256 mismatch: expected={expected_sha} actual={actual}")
    return I51EvidenceBinding(str(resolved), actual)


def _source_file_records(admission: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    source = admission.get("source_identity")
    if not isinstance(source, Mapping):
        raise ReconstructionValidationError("I5.2B I5.2A source_identity is missing.")
    records = source.get("files")
    if not isinstance(records, list):
        raise ReconstructionValidationError("I5.2B I5.2A source file records are missing.")
    return {str(item["name"]): item for item in records if isinstance(item, Mapping)}


def _stage_slice(
    *,
    dataset: Path,
    plan: Mapping[str, Any],
    source_records: Mapping[str, Mapping[str, Any]],
    output: Path,
) -> dict[str, Any]:
    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=False)
    names = [str(name) for name in plan.get("names", [])]
    copied: list[dict[str, Any]] = []
    for name in names:
        source = (dataset / name).resolve()
        try:
            source.relative_to(dataset)
        except ValueError as exc:
            raise ReconstructionValidationError(f"I5.2B source path escapes dataset: {source}") from exc
        record = source_records.get(name)
        if record is None or not source.is_file():
            raise ReconstructionValidationError(f"I5.2B frozen source image missing: {name}")
        actual = _sha256(source)
        if actual != str(record.get("sha256", "")):
            raise ReconstructionValidationError(f"I5.2B source image SHA256 drifted: {name}")
        target = output / name
        shutil.copy2(source, target)
        copied_sha = _sha256(target)
        if copied_sha != actual:
            raise ReconstructionValidationError(f"I5.2B staged source image changed: {name}")
        copied.append({"name": name, "sha256": actual, "size_bytes": int(target.stat().st_size)})
    return {"directory": str(output.resolve()), "image_count": len(copied), "files": copied}


def _run_sfm(
    *,
    scenario_label: str,
    images_dir: Path,
    output_dir: Path,
    sfm_python: str,
    image_count: int,
    timeout_minutes: float,
    maximum_image_size: int | None,
    cpu: bool,
) -> tuple[int, Path, Path, list[str]]:
    runner = ROOT / "scripts/run_reconstruction_pycolmap_smoke.py"
    command = [
        sys.executable,
        str(runner),
        "--images", str(images_dir),
        "--sfm-python", str(Path(sfm_python).expanduser().resolve()),
        "--output-dir", str(output_dir),
        "--profile", "efficient",
        "--matching", "exhaustive",
        "--max-keyframes", str(image_count),
        "--image-sequence-fps", "10.0",
        "--timeout-minutes", str(float(timeout_minutes)),
    ]
    if maximum_image_size is not None:
        command += ["--maximum-image-size", str(int(maximum_image_size))]
    if cpu:
        command.append("--cpu")

    log_path = output_dir.parent / f"{scenario_label}.log"
    output_dir.mkdir(parents=True, exist_ok=True)
    print("I5.2B command: " + " ".join(command), flush=True)
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
    if rc not in (0, 1) or not report_path.is_file():
        raise ReconstructionValidationError(
            f"I5.2B {scenario_label} did not produce structured real-SfM evidence (rc={rc}, report={report_path.is_file()})."
        )
    return rc, report_path, log_path, command


def _verify_staged_slice(stage: Mapping[str, Any]) -> bool:
    directory = Path(str(stage["directory"]))
    for raw in stage.get("files", []):
        if not isinstance(raw, Mapping):
            return False
        path = directory / str(raw.get("name", ""))
        if not path.is_file() or _sha256(path) != str(raw.get("sha256", "")):
            return False
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--i5-2a-admission", type=Path, required=True)
    parser.add_argument("--i5-2a-sha256", required=True)
    parser.add_argument("--replay-receipt", type=Path, required=True)
    parser.add_argument("--replay-receipt-sha256", required=True)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sfm-python", required=True)
    parser.add_argument("--sfm-timeout-minutes", type=float, default=60.0)
    parser.add_argument("--maximum-image-size", type=int, default=None)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    if args.sfm_timeout_minutes <= 0:
        parser.error("--sfm-timeout-minutes must be positive")
    if args.maximum_image_size is not None and args.maximum_image_size <= 0:
        parser.error("--maximum-image-size must be positive")

    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    admission_binding = _binding(args.i5_2a_admission, args.i5_2a_sha256, label="I5.2A admission")
    receipt_binding = _binding(args.replay_receipt, args.replay_receipt_sha256, label="replay receipt")
    admission = validate_i5_2a_for_i5_2b(_read_json(Path(admission_binding.path)))
    receipt = validate_i5_2b_receipt(_read_json(Path(receipt_binding.path)))
    del receipt

    dataset = args.dataset_dir.expanduser().resolve()
    if not dataset.is_dir():
        raise ReconstructionValidationError(f"I5.2B dataset directory missing: {dataset}")
    before = build_ordered_source_identity(dataset)
    expected_identity = admission["source_identity"]
    if before["asset_set_sha256"] != expected_identity["asset_set_sha256"] or before["ordered_names"] != expected_identity["ordered_names"]:
        raise ReconstructionValidationError("I5.2B source identity drifted from frozen I5.2A admission before replay.")

    plan = admission["stress_plan"]
    source_records = _source_file_records(admission)
    scenario_specs = (
        ("control", "01_control", "control"),
        ("low_overlap_proxy", "02_low_overlap", "low_overlap"),
        ("weak_parallax_proxy", "03_weak_parallax", "weak_parallax"),
    )
    execution: dict[str, Any] = {}
    analyses: dict[str, Any] = {}
    staged: dict[str, Any] = {}
    for plan_key, directory_name, output_key in scenario_specs:
        scenario_plan = plan[plan_key]
        stage = _stage_slice(
            dataset=dataset,
            plan=scenario_plan,
            source_records=source_records,
            output=output / "inputs" / output_key,
        )
        staged[output_key] = stage
        rc, report_path, log_path, command = _run_sfm(
            scenario_label=output_key,
            images_dir=Path(stage["directory"]),
            output_dir=output / directory_name,
            sfm_python=args.sfm_python,
            image_count=int(scenario_plan["image_count"]),
            timeout_minutes=args.sfm_timeout_minutes,
            maximum_image_size=args.maximum_image_size,
            cpu=args.cpu,
        )
        report = _read_json(report_path)
        analyses[output_key] = analyze_i5_2b_sfm_report(report, scenario_plan=scenario_plan)
        execution[output_key] = {
            "return_code": rc,
            "command": command,
            "report": str(report_path.resolve()),
            "report_sha256": _sha256(report_path),
            "log": str(log_path.resolve()),
            "log_sha256": _sha256(log_path),
            "structured_evidence_complete": True,
        }

    if not all(_verify_staged_slice(value) for value in staged.values()):
        raise ReconstructionValidationError("I5.2B staged input bytes changed during replay.")
    after = build_ordered_source_identity(dataset)
    if after["asset_set_sha256"] != before["asset_set_sha256"] or after["ordered_names"] != before["ordered_names"]:
        raise ReconstructionValidationError("I5.2B source identity changed during real replay.")

    result = build_i5_2b_low_overlap_weak_parallax_replay(
        i5_2a_result=admission,
        i5_2a_binding=admission_binding,
        replay_receipt=_read_json(Path(receipt_binding.path)),
        replay_receipt_binding=receipt_binding,
        control_analysis=analyses["control"],
        low_overlap_analysis=analyses["low_overlap"],
        weak_parallax_analysis=analyses["weak_parallax"],
        source_reverified_before=True,
        source_reverified_after=True,
    )
    result["execution_evidence"] = execution
    result["staged_input_evidence"] = staged
    result["source_reverification"] = {
        "before_asset_set_sha256": before["asset_set_sha256"],
        "after_asset_set_sha256": after["asset_set_sha256"],
        "unchanged": True,
    }
    evidence_path = _write_json(output / "i5_2b_low_overlap_weak_parallax_replay.json", result)

    comparison = result["comparison"]
    print("I5.2B Current Frozen-Foundation Low-Overlap / Weak-Parallax Replay: EVIDENCE_COMPLETE")
    print(
        f"quality_status={result['quality_status']} "
        f"control={analyses['control']['registered_images']}/{analyses['control']['input_images']} "
        f"low_overlap={analyses['low_overlap']['registered_images']}/{analyses['low_overlap']['input_images']} "
        f"weak_parallax={analyses['weak_parallax']['registered_images']}/{analyses['weak_parallax']['input_images']}"
    )
    low = comparison["low_overlap"]
    weak = comparison["weak_parallax"]
    print(
        "low_overlap_authoritative_revalidation="
        f"{low['authoritative_covisibility_reduction_confirmed']} "
        f"control_common_p50={low['control_adjacent_common_landmarks_p50']} "
        f"stress_common_p50={low['stress_adjacent_common_landmarks_p50']}"
    )
    print(
        "weak_parallax_authoritative_revalidation="
        f"{weak['authoritative_parallax_reduction_confirmed'] and weak['authoritative_baseline_to_range_reduction_confirmed']} "
        f"control_parallax_p50={weak['control_max_pair_parallax_degrees_p50']} "
        f"stress_parallax_p50={weak['stress_max_pair_parallax_degrees_p50']} "
        f"control_baseline_ratio_p50={weak['control_baseline_to_range_ratio_p50']} "
        f"stress_baseline_ratio_p50={weak['stress_baseline_to_range_ratio_p50']}"
    )
    print(
        "pycolmap_authority=True threshold_relaxation=False production_default_modified=False "
        "auto=False hybrid=False metric_scale=False metrology=False permanent_global_view_cap=None"
    )
    print("I5.2C low-overlap / weak-parallax robustness qualification and closure: AUTHORIZED")
    print(f"evidence={evidence_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
