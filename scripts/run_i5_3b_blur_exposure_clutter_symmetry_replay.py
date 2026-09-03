#!/usr/bin/env python3
"""Run I5.3B control/blur/exposure/clutter/symmetry real PyCOLMAP replay."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any, Mapping, Sequence

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.reconstruction.errors import ReconstructionValidationError  # noqa: E402
from core.reconstruction.i5_blur_exposure_clutter_symmetry import (  # noqa: E402
    analyze_i5_3b_image_set,
    analyze_i5_3b_pre_sfm_ingestion_rejection,
    analyze_i5_3b_sfm_report,
    apply_i5_3b_transform,
    build_i5_3b_blur_exposure_clutter_symmetry_replay,
    compare_i5_3b_image_domain,
    validate_i5_3a_for_i5_3b,
    validate_i5_3b_receipt,
)
from core.reconstruction.i5_difficult_handheld_robustness import I51EvidenceBinding  # noqa: E402
from core.reconstruction.i5_low_overlap_weak_parallax import build_ordered_source_identity  # noqa: E402

OUTPUT_NAME = "i5_3b_blur_exposure_clutter_symmetry_replay.json"


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--i5-3a-admission", type=Path, required=True)
    p.add_argument("--i5-3a-sha256", required=True)
    p.add_argument("--replay-receipt", type=Path, required=True)
    p.add_argument("--replay-receipt-sha256", required=True)
    p.add_argument("--dataset-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--sfm-python", required=True)
    p.add_argument("--sfm-timeout-minutes", type=float, default=60.0)
    p.add_argument("--maximum-image-size", type=int, default=None)
    p.add_argument("--cpu", action="store_true")
    return p


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _pixel_sha256(image: np.ndarray) -> str:
    digest = hashlib.sha256()
    digest.update(str(tuple(int(value) for value in image.shape)).encode("ascii"))
    digest.update(b"\0")
    digest.update(str(image.dtype).encode("ascii"))
    digest.update(b"\0")
    digest.update(np.ascontiguousarray(image).tobytes())
    return digest.hexdigest()


def _binding(path: Path, expected_sha: str, *, label: str) -> I51EvidenceBinding:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise ReconstructionValidationError(f"I5.3B {label} missing: {resolved}")
    expected = str(expected_sha).strip().lower()
    actual = _sha256(resolved)
    if actual != expected:
        raise ReconstructionValidationError(f"I5.3B {label} SHA256 mismatch: expected={expected} actual={actual}")
    return I51EvidenceBinding(str(resolved), actual)


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


def _source_records(admission: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    source = admission.get("source_identity")
    if not isinstance(source, Mapping):
        raise ReconstructionValidationError("I5.3B I5.3A source_identity is missing.")
    records = source.get("files")
    if not isinstance(records, list):
        raise ReconstructionValidationError("I5.3B I5.3A source file records are missing.")
    return {str(item["name"]): item for item in records if isinstance(item, Mapping)}


def _decode_source(dataset: Path, name: str, record: Mapping[str, Any]) -> tuple[Path, np.ndarray, str]:
    source = (dataset / name).resolve()
    try:
        source.relative_to(dataset)
    except ValueError as exc:
        raise ReconstructionValidationError(f"I5.3B source path escapes dataset: {source}") from exc
    if not source.is_file():
        raise ReconstructionValidationError(f"I5.3B frozen source image missing: {name}")
    actual = _sha256(source)
    if actual != str(record.get("sha256", "")):
        raise ReconstructionValidationError(f"I5.3B source image SHA256 drifted: {name}")
    image = cv2.imread(str(source), cv2.IMREAD_COLOR)
    if image is None:
        raise ReconstructionValidationError(f"I5.3B cannot decode frozen source image: {source}")
    return source, image, actual


def _write_lossless_png(path: Path, image: np.ndarray) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(path), image, [cv2.IMWRITE_PNG_COMPRESSION, 3])
    if not ok:
        raise ReconstructionValidationError(f"I5.3B failed to materialize lossless PNG: {path}")
    decoded = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if decoded is None or decoded.shape != image.shape or not np.array_equal(decoded, image):
        raise ReconstructionValidationError(f"I5.3B lossless PNG pixel verification failed: {path}")
    return _sha256(path)


def _materialize_scenario(
    *,
    dataset: Path,
    plan: Mapping[str, Any],
    source_records: Mapping[str, Mapping[str, Any]],
    output: Path,
) -> dict[str, Any]:
    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=False)
    names = [str(value) for value in plan.get("names", [])]
    if len(names) != int(plan.get("image_count", -1)):
        raise ReconstructionValidationError("I5.3B frozen scenario membership is invalid.")
    transform = plan.get("transform")
    if not isinstance(transform, Mapping):
        raise ReconstructionValidationError("I5.3B frozen scenario transform is missing.")
    algorithm = str(transform.get("algorithm", ""))
    offset = int(transform.get("source_offset", 0)) if algorithm == "source_derived_distractor_band" else 0
    staged_names = [f"{index:04d}.png" for index in range(len(names))]
    files: list[dict[str, Any]] = []
    for index, (name, staged_name) in enumerate(zip(names, staged_names)):
        record = source_records.get(name)
        if record is None:
            raise ReconstructionValidationError(f"I5.3B frozen source record missing: {name}")
        source, image, source_sha = _decode_source(dataset, name, record)
        distractor_name = None
        distractor = None
        if algorithm == "source_derived_distractor_band":
            distractor_name = names[(index + offset) % len(names)]
            distractor_record = source_records.get(distractor_name)
            if distractor_record is None:
                raise ReconstructionValidationError(f"I5.3B clutter distractor source record missing: {distractor_name}")
            _, distractor, _ = _decode_source(dataset, distractor_name, distractor_record)
        transformed = apply_i5_3b_transform(image, transform, distractor_image=distractor)
        target = output / staged_name
        staged_sha = _write_lossless_png(target, transformed)
        files.append({
            "source_index": index,
            "source_name": name,
            "source_path": str(source),
            "source_sha256": source_sha,
            "source_pixel_sha256": _pixel_sha256(image),
            "staged_name": staged_name,
            "staged_sha256": staged_sha,
            "staged_pixel_sha256": _pixel_sha256(transformed),
            "size_bytes": int(target.stat().st_size),
            "transform_algorithm": algorithm,
            "clutter_distractor_source_name": distractor_name,
        })
    return {
        "directory": str(output.resolve()),
        "image_count": len(files),
        "source_names": names,
        "staged_names": staged_names,
        "encoding": "lossless_png",
        "png_compression_level": 3,
        "files": files,
    }


def _verify_stage(stage: Mapping[str, Any]) -> bool:
    directory = Path(str(stage.get("directory", "")))
    files = stage.get("files")
    if not directory.is_dir() or not isinstance(files, Sequence):
        return False
    for raw in files:
        if not isinstance(raw, Mapping):
            return False
        path = directory / str(raw.get("staged_name", ""))
        if not path.is_file() or _sha256(path) != str(raw.get("staged_sha256", "")):
            return False
    return True


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
) -> tuple[int, Path | None, Path, list[str], str | None]:
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
    print("I5.3B command: " + " ".join(command), flush=True)
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
    if report_path.is_file():
        if rc not in (0, 1):
            raise ReconstructionValidationError(
                f"I5.3B {scenario_label} produced a report with unexpected rc={rc}."
            )
        return rc, report_path, log_path, command, None

    log_text = log_path.read_text(encoding="utf-8", errors="replace")
    ingestion_marker = "Fewer than three valid keyframes survived ingestion."
    if rc == 1 and ingestion_marker in log_text:
        reason_line = next((line.strip() for line in log_text.splitlines() if ingestion_marker in line), ingestion_marker)
        return rc, None, log_path, command, reason_line

    raise ReconstructionValidationError(
        f"I5.3B {scenario_label} did not produce structured real-SfM evidence "
        f"and was not a recognized pre-SfM ingestion rejection (rc={rc}, report=False)."
    )


def run(args: argparse.Namespace) -> Path:
    if args.sfm_timeout_minutes <= 0:
        raise ReconstructionValidationError("I5.3B --sfm-timeout-minutes must be positive.")
    if args.maximum_image_size is not None and args.maximum_image_size <= 0:
        raise ReconstructionValidationError("I5.3B --maximum-image-size must be positive.")

    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    admission_binding = _binding(args.i5_3a_admission, args.i5_3a_sha256, label="I5.3A admission")
    receipt_binding = _binding(args.replay_receipt, args.replay_receipt_sha256, label="replay receipt")
    admission = validate_i5_3a_for_i5_3b(_read_json(Path(admission_binding.path)))
    validate_i5_3b_receipt(_read_json(Path(receipt_binding.path)))

    dataset = args.dataset_dir.expanduser().resolve()
    if not dataset.is_dir():
        raise ReconstructionValidationError(f"I5.3B dataset directory missing: {dataset}")
    before = build_ordered_source_identity(dataset)
    frozen = admission["source_identity"]
    if before["asset_set_sha256"] != frozen["asset_set_sha256"] or before["ordered_names"] != frozen["ordered_names"]:
        raise ReconstructionValidationError("I5.3B source identity drifted from frozen I5.3A admission before replay.")

    plan = admission["stress_plan"]
    records = _source_records(admission)
    scenario_specs = (
        ("control", "control", "01_control"),
        ("blur_proxy", "blur", "02_blur"),
        ("underexposure_proxy", "underexposure", "03_underexposure"),
        ("overexposure_proxy", "overexposure", "04_overexposure"),
        ("clutter_proxy", "clutter", "05_clutter"),
        ("symmetry_proxy", "symmetry", "06_symmetry"),
    )

    staged: dict[str, Any] = {}
    diagnostics: dict[str, Any] = {}
    for plan_key, output_key, _ in scenario_specs:
        stage = _materialize_scenario(
            dataset=dataset,
            plan=plan[plan_key],
            source_records=records,
            output=output / "inputs" / output_key,
        )
        staged[output_key] = stage
        diagnostics[output_key] = analyze_i5_3b_image_set(
            stage["directory"],
            names=stage["staged_names"],
            band_fraction=float(plan["clutter_proxy"]["transform"]["band_fraction"]),
            symmetry_band_fraction=float(plan["symmetry_proxy"]["transform"]["band_fraction"]),
        )

    image_comparison = compare_i5_3b_image_domain(
        control=diagnostics["control"],
        stresses={key: diagnostics[key] for key in ("blur", "underexposure", "overexposure", "clutter", "symmetry")},
    )

    analyses: dict[str, Any] = {}
    execution: dict[str, Any] = {}
    for plan_key, output_key, directory_name in scenario_specs:
        stage = staged[output_key]
        rc, report_path, log_path, command, ingestion_rejection = _run_sfm(
            scenario_label=output_key,
            images_dir=Path(stage["directory"]),
            output_dir=output / directory_name,
            sfm_python=args.sfm_python,
            image_count=int(plan[plan_key]["image_count"]),
            timeout_minutes=args.sfm_timeout_minutes,
            maximum_image_size=args.maximum_image_size,
            cpu=args.cpu,
        )
        if report_path is not None:
            report = _read_json(report_path)
            analyses[output_key] = analyze_i5_3b_sfm_report(
                report,
                scenario_plan=plan[plan_key],
                staged_names=stage["staged_names"],
            )
            report_value: str | None = str(report_path.resolve())
            report_sha: str | None = _sha256(report_path)
        else:
            assert ingestion_rejection is not None
            analyses[output_key] = analyze_i5_3b_pre_sfm_ingestion_rejection(
                scenario_plan=plan[plan_key],
                staged_names=stage["staged_names"],
                reason=ingestion_rejection,
            )
            report_value = None
            report_sha = None
        execution[output_key] = {
            "return_code": rc,
            "command": command,
            "report": report_value,
            "report_sha256": report_sha,
            "log": str(log_path.resolve()),
            "log_sha256": _sha256(log_path),
            "structured_evidence_complete": True,
            "pre_sfm_ingestion_rejected": bool(ingestion_rejection is not None),
            "pre_sfm_ingestion_rejection_reason": ingestion_rejection,
        }

    staged_ok = all(_verify_stage(stage) for stage in staged.values())
    if not staged_ok:
        raise ReconstructionValidationError("I5.3B staged input bytes changed during replay.")
    after = build_ordered_source_identity(dataset)
    source_unchanged = after["asset_set_sha256"] == before["asset_set_sha256"] and after["ordered_names"] == before["ordered_names"]
    if not source_unchanged:
        raise ReconstructionValidationError("I5.3B source identity changed during real replay.")

    result = build_i5_3b_blur_exposure_clutter_symmetry_replay(
        i5_3a_result=admission,
        i5_3a_binding=admission_binding,
        replay_receipt=_read_json(Path(receipt_binding.path)),
        replay_receipt_binding=receipt_binding,
        image_domain_comparison=image_comparison,
        scenario_analyses=analyses,
        source_reverified_before=True,
        source_reverified_after=True,
        staged_inputs_reverified=True,
    )
    result["image_domain_diagnostics"] = diagnostics
    result["execution_evidence"] = execution
    result["staged_input_evidence"] = staged
    result["source_reverification"] = {
        "before_asset_set_sha256": before["asset_set_sha256"],
        "after_asset_set_sha256": after["asset_set_sha256"],
        "unchanged": True,
    }
    result["materialization"] = {
        "all_scenarios_lossless_png": True,
        "control_pixel_values_preserved_from_decoded_frozen_source": True,
        "same_lossless_encoding_for_control_and_all_stresses": True,
        "lossy_reencoding_confound_introduced": False,
    }
    evidence_path = _write_json(output / OUTPUT_NAME, result)

    comp = result["image_domain_revalidation"]
    print("I5.3B Current Frozen-Foundation Blur / Exposure / Clutter / Symmetry Replay: EVIDENCE_COMPLETE")
    print(
        "quality_status=" + result["quality_status"] + " " +
        " ".join(
            f"{key}={analyses[key]['registered_images']}/{analyses[key]['input_images']}"
            for key in ("control", "blur", "underexposure", "overexposure", "clutter", "symmetry")
        )
    )
    print(
        f"blur_revalidated={comp['blur']['semantic_stress_revalidated']} "
        f"control_blur_p50={comp['blur']['control_blur_variance_p50']} stress_blur_p50={comp['blur']['stress_blur_variance_p50']}"
    )
    print(
        f"underexposure_revalidated={comp['underexposure']['semantic_stress_revalidated']} "
        f"control_luma_p50={comp['underexposure']['control_mean_luma_p50']} stress_luma_p50={comp['underexposure']['stress_mean_luma_p50']}"
    )
    print(
        f"overexposure_revalidated={comp['overexposure']['semantic_stress_revalidated']} "
        f"control_luma_p50={comp['overexposure']['control_mean_luma_p50']} stress_luma_p50={comp['overexposure']['stress_mean_luma_p50']}"
    )
    print(
        f"clutter_revalidated={comp['clutter']['semantic_stress_revalidated']} "
        f"control_band_orb_density_p50={comp['clutter']['control_bottom_band_orb_feature_density_p50']} "
        f"stress_band_orb_density_p50={comp['clutter']['stress_bottom_band_orb_feature_density_p50']} natural_ground_truth=False"
    )
    print(
        f"symmetry_revalidated={comp['symmetry']['semantic_stress_revalidated']} "
        f"control_similarity_p50={comp['symmetry']['control_mirrored_side_similarity_p50']} "
        f"stress_similarity_p50={comp['symmetry']['stress_mirrored_side_similarity_p50']} natural_ground_truth=False"
    )
    print(
        "pycolmap_authority=True threshold_relaxation=False production_default_modified=False "
        "dense_provider_rerun=False mp_sfm=False auto=False hybrid=False metric_scale=False metrology=False permanent_global_view_cap=None"
    )
    print("I5.3C blur / exposure / clutter / symmetry robustness qualification and closure: AUTHORIZED")
    print(f"evidence={evidence_path}")
    return evidence_path


def main() -> int:
    try:
        run(_parser().parse_args())
    except (ReconstructionValidationError, OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
        print(f"I5.3B ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
