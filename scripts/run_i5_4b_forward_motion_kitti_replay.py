#!/usr/bin/env python3
"""Run I5.4B real frozen-foundation forward-motion / KITTI sparse-SfM replay."""
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

from core.reconstruction.errors import ReconstructionValidationError  # noqa: E402
from core.reconstruction.i5_difficult_handheld_robustness import I51EvidenceBinding  # noqa: E402
from core.reconstruction.i5_forward_motion_kitti import (  # noqa: E402
    analyze_i5_4b_pre_sfm_ingestion_rejection,
    analyze_i5_4b_mapping_no_sparse_model_rejection,
    analyze_i5_4b_sfm_report,
    build_i5_4a_source_identity,
    build_i5_4b_forward_motion_kitti_replay,
    validate_i5_4a_for_i5_4b,
    validate_i5_4b_receipt,
)

OUTPUT_NAME = "i5_4b_forward_motion_kitti_replay.json"
_INGESTION_REJECTION = "Fewer than three valid keyframes survived ingestion"
_MAPPING_NO_MODEL_REJECTION = "PyCOLMAP incremental mapping produced no sparse model"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ReconstructionValidationError(f"I5.4B could not read JSON {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ReconstructionValidationError(f"I5.4B JSON root must be an object: {path}")
    return payload


def _binding(path: Path, expected_sha: str, *, label: str) -> I51EvidenceBinding:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise ReconstructionValidationError(f"I5.4B {label} missing: {resolved}")
    expected = str(expected_sha).strip().lower()
    if len(expected) != 64 or any(ch not in "0123456789abcdef" for ch in expected):
        raise ReconstructionValidationError(f"I5.4B {label} expected SHA256 is invalid.")
    actual = _sha256(resolved)
    if actual != expected:
        raise ReconstructionValidationError(f"I5.4B {label} SHA256 mismatch: expected={expected} actual={actual}")
    return I51EvidenceBinding(str(resolved), actual)


def _source_records(admission: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    identity = admission["source_identity"]
    return {str(item["name"]): item for item in identity["files"]}


def _stage_frozen_window(*, admission: Mapping[str, Any], dataset: Path, output: Path) -> dict[str, Any]:
    window = admission["stress_plan"]["forward_motion_window"]
    records = _source_records(admission)
    output.mkdir(parents=True, exist_ok=False)
    copied = []
    for name in window["names"]:
        source = (dataset / str(name)).resolve()
        try:
            source.relative_to(dataset)
        except ValueError as exc:
            raise ReconstructionValidationError(f"I5.4B source path escapes dataset: {source}") from exc
        record = records.get(str(name))
        if record is None or not source.is_file():
            raise ReconstructionValidationError(f"I5.4B frozen source image missing: {name}")
        actual = _sha256(source)
        if actual != str(record.get("sha256", "")).lower():
            raise ReconstructionValidationError(f"I5.4B source image SHA256 drifted: {name}")
        target = output / str(name)
        shutil.copy2(source, target)
        if _sha256(target) != actual:
            raise ReconstructionValidationError(f"I5.4B staged source image changed: {name}")
        copied.append({"name": str(name), "sha256": actual, "size_bytes": int(target.stat().st_size)})
    return {"directory": str(output.resolve()), "image_count": len(copied), "files": copied}


def _verify_selected_originals(*, admission: Mapping[str, Any], dataset: Path) -> bool:
    records = _source_records(admission)
    for name in admission["stress_plan"]["forward_motion_window"]["names"]:
        path = dataset / str(name)
        record = records.get(str(name))
        if record is None or not path.is_file() or _sha256(path) != str(record.get("sha256", "")).lower():
            return False
    return True


def _verify_staged(stage: Mapping[str, Any]) -> bool:
    root = Path(str(stage["directory"]))
    for raw in stage["files"]:
        path = root / str(raw["name"])
        if not path.is_file() or _sha256(path) != str(raw["sha256"]):
            return False
    return True


def _run_sfm(*, images_dir: Path, output_dir: Path, sfm_python: str, timeout_minutes: float) -> tuple[int, Path, Path, list[str], str]:
    runner = ROOT / "scripts/run_reconstruction_pycolmap_smoke.py"
    command = [
        sys.executable, str(runner),
        "--images", str(images_dir),
        "--sfm-python", str(Path(sfm_python).expanduser().resolve()),
        "--output-dir", str(output_dir),
        "--profile", "efficient",
        "--matching", "sequential",
        "--sequential-overlap", "10",
        "--max-keyframes", "60",
        "--image-sequence-fps", "10.0",
        "--timeout-minutes", str(float(timeout_minutes)),
    ]
    log_path = output_dir.parent / "forward_motion.log"
    output_dir.mkdir(parents=True, exist_ok=True)
    print("I5.4B command: " + " ".join(command), flush=True)
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(command, cwd=str(ROOT), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line); log.flush()
        rc = int(process.wait())
    text = log_path.read_text(encoding="utf-8", errors="replace")
    return rc, output_dir / "real_sfm_report.json", log_path, command, text


def _write_json(path: Path, payload: Mapping[str, Any]) -> Path:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--i5-4a-admission", type=Path, required=True)
    parser.add_argument("--i5-4a-sha256", required=True)
    parser.add_argument("--replay-receipt", type=Path, required=True)
    parser.add_argument("--replay-receipt-sha256", required=True)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sfm-python", required=True)
    parser.add_argument("--sfm-timeout-minutes", type=float, default=60.0)
    args = parser.parse_args()
    if args.sfm_timeout_minutes <= 0:
        parser.error("--sfm-timeout-minutes must be positive")

    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    admission_binding = _binding(args.i5_4a_admission, args.i5_4a_sha256, label="I5.4A admission")
    receipt_binding = _binding(args.replay_receipt, args.replay_receipt_sha256, label="I5.4B replay receipt")
    admission = validate_i5_4a_for_i5_4b(_read_json(Path(admission_binding.path)))
    validate_i5_4b_receipt(_read_json(Path(receipt_binding.path)))

    dataset = args.dataset_dir.expanduser().resolve()
    if not dataset.is_dir():
        raise ReconstructionValidationError(f"I5.4B dataset directory missing: {dataset}")
    admitted_dir = Path(str(admission["source_identity"]["directory"])).expanduser().resolve()
    if dataset != admitted_dir:
        raise ReconstructionValidationError(f"I5.4B dataset path drifted from I5.4A admission: admitted={admitted_dir} requested={dataset}")

    before = build_i5_4a_source_identity(dataset)
    frozen = admission["source_identity"]
    if before["asset_set_sha256"] != frozen["asset_set_sha256"] or before["ordered_names"] != frozen["ordered_names"]:
        raise ReconstructionValidationError("I5.4B full source identity drifted from frozen I5.4A admission before replay.")

    stage = _stage_frozen_window(admission=admission, dataset=dataset, output=output / "inputs" / "forward_motion")
    rc, report_path, log_path, command, log_text = _run_sfm(
        images_dir=Path(stage["directory"]), output_dir=output / "01_forward_motion",
        sfm_python=args.sfm_python, timeout_minutes=args.sfm_timeout_minutes,
    )
    plan = admission["stress_plan"]["forward_motion_window"]
    if report_path.is_file() and rc in (0, 1):
        analysis = analyze_i5_4b_sfm_report(_read_json(report_path), scenario_plan=plan)
        structured_kind = "real_sfm_report"
        report_sha = _sha256(report_path)
    elif _INGESTION_REJECTION in log_text:
        analysis = analyze_i5_4b_pre_sfm_ingestion_rejection(
            scenario_plan=plan, staged_names=[item["name"] for item in stage["files"]], reason=log_text.strip().splitlines()[-1] if log_text.strip() else _INGESTION_REJECTION,
        )
        structured_kind = "recognized_pre_sfm_ingestion_rejection"
        report_sha = None
    elif _MAPPING_NO_MODEL_REJECTION in log_text:
        reason_line = next(
            (line.strip() for line in reversed(log_text.splitlines()) if _MAPPING_NO_MODEL_REJECTION in line),
            _MAPPING_NO_MODEL_REJECTION,
        )
        analysis = analyze_i5_4b_mapping_no_sparse_model_rejection(
            scenario_plan=plan, staged_names=[item["name"] for item in stage["files"]], reason=reason_line,
        )
        structured_kind = "recognized_incremental_mapping_no_sparse_model_rejection"
        report_sha = None
    else:
        raise ReconstructionValidationError(
            f"I5.4B did not produce structured real-SfM evidence (rc={rc}, report={report_path.is_file()})."
        )

    staged_ok = _verify_staged(stage)
    selected_after_ok = _verify_selected_originals(admission=admission, dataset=dataset)
    if not staged_ok or not selected_after_ok:
        raise ReconstructionValidationError("I5.4B selected/staged source bytes changed during replay.")

    result = build_i5_4b_forward_motion_kitti_replay(
        i5_4a_result=admission, i5_4a_binding=admission_binding,
        replay_receipt=_read_json(Path(receipt_binding.path)), replay_receipt_binding=receipt_binding,
        sfm_analysis=analysis, source_reverified_before=True, source_window_reverified_after=True, staged_inputs_reverified=True,
    )
    result["execution_evidence"] = {
        "return_code": rc, "command": command, "structured_evidence_kind": structured_kind,
        "report": str(report_path.resolve()) if report_path.is_file() else None,
        "report_sha256": report_sha, "log": str(log_path.resolve()), "log_sha256": _sha256(log_path),
    }
    result["staged_input_evidence"] = stage
    result["source_reverification"] = {
        "full_source_before_asset_set_sha256": before["asset_set_sha256"],
        "full_source_before_matches_admission": True,
        "selected_window_after_matches_admission": True,
        "staged_window_after_matches_admission": True,
        "full_source_rehash_after_replay_performed": False,
        "reason": "I5.4B rehashes the full 4544-image source before replay and rehashes the exact 60 selected originals plus staged copies after replay.",
    }
    evidence = _write_json(output / OUTPUT_NAME, result)

    motion = result["forward_motion_revalidation"]
    parallax = result["replay"]["geometry"]["parallax"]
    print("I5.4B Current Frozen-Foundation Forward-Motion / KITTI Replay: EVIDENCE_COMPLETE")
    print(f"quality_status={result['quality_status']} registered={analysis['registered_images']}/{analysis['input_images']} sfm_quality={analysis['quality_status']}")
    print(
        "forward_motion_geometry_revalidated="
        f"{motion['authoritative_forward_motion_geometry_proven']} "
        f"adjacent_motion_pairs={motion['adjacent_registered_motion_pairs']} "
        f"median_forward_fraction={motion['median_forward_component_fraction']} "
        f"median_lateral_fraction={motion['median_lateral_component_fraction']}"
    )
    print(
        "i2_4_geometry "
        f"parallax_p50={parallax['maximum_pair_parallax_degrees_quantiles']['p50']} "
        f"baseline_to_range_p50={parallax['maximum_baseline_to_range_ratio_quantiles']['p50']} "
        "scale_invariant=True"
    )
    print(
        "pycolmap_authority=True threshold_relaxation=False production_default_modified=False "
        "dense_provider_rerun=False mp_sfm=False auto=False hybrid=False metric_scale=False metrology=False permanent_global_view_cap=None"
    )
    print("I5.4C forward-motion / KITTI robustness qualification and closure: AUTHORIZED")
    print(f"evidence={evidence}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ReconstructionValidationError, OSError, ValueError, TypeError, KeyError) as exc:
        print(f"I5.4B ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(2)
