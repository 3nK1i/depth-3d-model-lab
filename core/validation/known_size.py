"""Deterministic known-size validation for Phase 7.5 measurement exports.

The validator never reruns perception or depth.  It consumes finalized Step 9
inspection exports plus operator-recorded ground-truth dimensions and produces
machine-readable and human-readable accuracy reports.
"""

from __future__ import annotations

import csv
import hashlib
import html
import json
import math
import statistics
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

SCHEMA_VERSION = 1
REPORT_SCHEMA_VERSION = 1
DIMENSIONS = ("width", "height", "depth")


class KnownSizeBenchmarkError(ValueError):
    """Raised when a benchmark manifest or export is malformed."""


@dataclass(frozen=True, slots=True)
class KnownSizeValidationResult:
    """Completed known-size validation result."""

    payload: dict[str, Any]

    @property
    def passed(self) -> bool:
        return bool(self.payload.get("passed", False))

    @property
    def case_count(self) -> int:
        return int(self.payload.get("summary", {}).get("case_count", 0))

    @property
    def available_dimension_count(self) -> int:
        return int(self.payload.get("summary", {}).get("available_dimension_count", 0))


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        raise KnownSizeBenchmarkError(f"Cannot read JSON {path}: {exc}") from exc


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _as_finite_float(value: Any, *, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise KnownSizeBenchmarkError(f"{name} must be numeric, got {value!r}") from exc
    if not math.isfinite(result):
        raise KnownSizeBenchmarkError(f"{name} must be finite, got {value!r}")
    return result


def _verify_export_manifest(export_dir: Path) -> tuple[bool, list[str]]:
    manifest_path = export_dir / "manifest.json"
    if not manifest_path.is_file():
        return False, ["manifest.json is missing"]
    payload = _read_json(manifest_path)
    entries = payload.get("files", [])
    if not isinstance(entries, list) or not entries:
        return False, ["manifest contains no files"]
    issues: list[str] = []
    for entry in entries:
        if not isinstance(entry, Mapping):
            issues.append("manifest contains a non-object entry")
            continue
        relative = str(entry.get("path", ""))
        expected = str(entry.get("sha256", ""))
        path = export_dir / relative
        if not path.is_file():
            issues.append(f"missing export file: {relative}")
            continue
        if expected and _sha256(path) != expected:
            issues.append(f"checksum mismatch: {relative}")
    return not issues, issues


def create_benchmark_template(output_path: str | Path) -> Path:
    """Write an editable benchmark manifest with explicit provisional thresholds."""

    path = Path(output_path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "benchmark_name": "Depth & 3D Model Lab Phase 7.5 Known-Size Benchmark",
        "operator": "",
        "measurement_instrument": "tape measure or caliper",
        "thresholds_confirmed": False,
        "thresholds": {
            "minimum_cases": 3,
            "minimum_available_dimensions": 9,
            "require_all_case_dimensions_available": True,
            "max_mean_absolute_error_mm": 50.0,
            "max_mean_relative_error_percent": 10.0,
            "max_single_dimension_error_mm": 150.0,
            "max_single_dimension_relative_error_percent": 20.0,
            "minimum_uncertainty_coverage_percent": 60.0,
        },
        "cases": [
            {
                "case_id": "box_front_1",
                "export_path": "/absolute/path/to/inspection_export_YYYYMMDD_HHMMSS",
                "observation_id": "replace-with-export-observation-id",
                "object_index": 0,
                "class_name": "box",
                "capture": {
                    "distance_m": 1.5,
                    "view": "front-oblique",
                    "lighting": "even indoor",
                    "notes": "Object fully visible and not touching the image border.",
                },
                "ground_truth_mm": {
                    "width": 400.0,
                    "height": 300.0,
                    "depth": 250.0,
                },
                "measurement_mapping": {
                    "width": "width",
                    "height": "height",
                    "depth": "depth",
                },
            }
        ],
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def load_benchmark_manifest(path: str | Path) -> dict[str, Any]:
    manifest_path = Path(path).expanduser().resolve()
    payload = _read_json(manifest_path)
    if int(payload.get("schema_version", -1)) != SCHEMA_VERSION:
        raise KnownSizeBenchmarkError(
            f"Unsupported benchmark schema {payload.get('schema_version')!r}; expected {SCHEMA_VERSION}."
        )
    cases = payload.get("cases")
    if not isinstance(cases, list) or not cases:
        raise KnownSizeBenchmarkError("Benchmark manifest must contain at least one case.")
    thresholds = payload.get("thresholds")
    if not isinstance(thresholds, Mapping):
        raise KnownSizeBenchmarkError("Benchmark manifest thresholds are missing.")
    return payload


def _resolve_export_path(case: Mapping[str, Any], manifest_path: Path) -> Path:
    raw = str(case.get("export_path", "")).strip()
    if not raw:
        raise KnownSizeBenchmarkError(f"Case {case.get('case_id')!r} has no export_path.")
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = manifest_path.parent / path
    return path.resolve()


def _select_object(objects: Sequence[Mapping[str, Any]], case: Mapping[str, Any]) -> Mapping[str, Any]:
    observation_id = str(case.get("observation_id", "")).strip()
    if observation_id:
        matches = [obj for obj in objects if str(obj.get("observation_id", "")) == observation_id]
    elif "object_index" in case:
        wanted = int(case["object_index"])
        matches = [obj for obj in objects if int(obj.get("object_index", -1)) == wanted]
    else:
        class_name = str(case.get("class_name", "")).strip().lower()
        matches = [obj for obj in objects if str(obj.get("class_name", "")).strip().lower() == class_name]
    if len(matches) != 1:
        raise KnownSizeBenchmarkError(
            f"Case {case.get('case_id')!r} resolved to {len(matches)} objects; "
            "set a unique observation_id or object_index."
        )
    return matches[0]


def _measurement_map(obj: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    measurements = obj.get("measurements", [])
    if not isinstance(measurements, list):
        raise KnownSizeBenchmarkError("Export object measurements must be a list.")
    return {
        str(item.get("kind", "")): item
        for item in measurements
        if isinstance(item, Mapping) and str(item.get("kind", ""))
    }


def _dimension_result(
    *,
    ground_truth_name: str,
    predicted_kind: str,
    ground_truth_mm: float,
    measurement: Mapping[str, Any] | None,
    thresholds: Mapping[str, Any],
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "ground_truth_dimension": ground_truth_name,
        "predicted_measurement": predicted_kind,
        "ground_truth_mm": ground_truth_mm,
        "status": "unavailable",
        "passed": False,
        "covered_by_uncertainty": None,
    }
    if measurement is None:
        result["reason"] = "measurement_missing"
        return result
    result["measurement_status"] = str(measurement.get("status", ""))
    result["reliability"] = str(measurement.get("reliability", ""))
    result["rejection_reasons"] = list(measurement.get("rejection_reasons", []))
    if measurement.get("status") != "available":
        result["reason"] = "measurement_not_available"
        return result

    estimate_mm = _as_finite_float(measurement.get("value_si"), name=f"{predicted_kind}.value_si") * 1000.0
    uncertainty_mm = _as_finite_float(
        measurement.get("expanded_uncertainty_si", 0.0),
        name=f"{predicted_kind}.expanded_uncertainty_si",
    ) * 1000.0
    signed_error_mm = estimate_mm - ground_truth_mm
    absolute_error_mm = abs(signed_error_mm)
    relative_error_percent = absolute_error_mm / ground_truth_mm * 100.0 if ground_truth_mm else 0.0
    max_abs = _as_finite_float(
        thresholds.get("max_single_dimension_error_mm", math.inf),
        name="max_single_dimension_error_mm",
    )
    max_rel = _as_finite_float(
        thresholds.get(
            "max_single_dimension_relative_error_percent",
            thresholds.get("max_mean_relative_error_percent", math.inf),
        ),
        name="max_single_dimension_relative_error_percent",
    )
    result.update(
        {
            "status": "available",
            "estimated_mm": estimate_mm,
            "expanded_uncertainty_mm": uncertainty_mm,
            "signed_error_mm": signed_error_mm,
            "absolute_error_mm": absolute_error_mm,
            "relative_error_percent": relative_error_percent,
            "covered_by_uncertainty": absolute_error_mm <= uncertainty_mm,
            "passed": absolute_error_mm <= max_abs and relative_error_percent <= max_rel,
        }
    )
    return result


def run_known_size_validation(
    manifest_path: str | Path,
    *,
    verify_export_checksums: bool = True,
) -> KnownSizeValidationResult:
    """Validate all benchmark cases and compute aggregate accuracy metrics."""

    resolved_manifest = Path(manifest_path).expanduser().resolve()
    manifest = load_benchmark_manifest(resolved_manifest)
    thresholds = manifest["thresholds"]
    thresholds_confirmed = bool(manifest.get("thresholds_confirmed", False))
    case_results: list[dict[str, Any]] = []
    global_issues: list[str] = []

    for raw_case in manifest["cases"]:
        if not isinstance(raw_case, Mapping):
            raise KnownSizeBenchmarkError("Every benchmark case must be an object.")
        case_id = str(raw_case.get("case_id", "")).strip()
        if not case_id:
            raise KnownSizeBenchmarkError("Every benchmark case requires case_id.")
        export_dir = _resolve_export_path(raw_case, resolved_manifest)
        case_issues: list[str] = []
        if not export_dir.is_dir():
            case_issues.append(f"missing export directory: {export_dir}")
            case_results.append({"case_id": case_id, "passed": False, "issues": case_issues, "dimensions": []})
            continue
        integrity_passed = True
        if verify_export_checksums:
            integrity_passed, integrity_issues = _verify_export_manifest(export_dir)
            case_issues.extend(integrity_issues)
        measurements_path = export_dir / "measurements.json"
        if not measurements_path.is_file():
            case_issues.append("measurements.json is missing")
            case_results.append({"case_id": case_id, "passed": False, "issues": case_issues, "dimensions": []})
            continue
        export_payload = _read_json(measurements_path)
        objects = export_payload.get("objects", [])
        if not isinstance(objects, list):
            raise KnownSizeBenchmarkError(f"Case {case_id}: export objects must be a list.")
        obj = _select_object(objects, raw_case)
        measurements = _measurement_map(obj)
        gt = raw_case.get("ground_truth_mm")
        mapping = raw_case.get("measurement_mapping", {})
        if not isinstance(gt, Mapping):
            raise KnownSizeBenchmarkError(f"Case {case_id}: ground_truth_mm is missing.")
        if not isinstance(mapping, Mapping):
            raise KnownSizeBenchmarkError(f"Case {case_id}: measurement_mapping must be an object.")

        dimensions: list[dict[str, Any]] = []
        for dimension in DIMENSIONS:
            ground_truth_mm = _as_finite_float(gt.get(dimension), name=f"{case_id}.{dimension}")
            if ground_truth_mm <= 0:
                raise KnownSizeBenchmarkError(f"{case_id}.{dimension} must be positive.")
            predicted_kind = str(mapping.get(dimension, dimension))
            dimensions.append(
                _dimension_result(
                    ground_truth_name=dimension,
                    predicted_kind=predicted_kind,
                    ground_truth_mm=ground_truth_mm,
                    measurement=measurements.get(predicted_kind),
                    thresholds=thresholds,
                )
            )

        require_all = bool(thresholds.get("require_all_case_dimensions_available", True))
        available = [item for item in dimensions if item["status"] == "available"]
        case_passed = integrity_passed and bool(available) and all(item["passed"] for item in available)
        if require_all and len(available) != len(dimensions):
            case_passed = False
            case_issues.append("not all required dimensions are available")
        case_results.append(
            {
                "case_id": case_id,
                "export_path": str(export_dir),
                "export_integrity_passed": integrity_passed,
                "observation_id": obj.get("observation_id"),
                "object_index": obj.get("object_index"),
                "class_name": obj.get("class_name"),
                "preferred_box_id": obj.get("preferred_box_id"),
                "mask_source": obj.get("mask_source"),
                "mask_revision": obj.get("mask_revision"),
                "calibration_status": export_payload.get("calibration_status", "not_calibrated"),
                "capture": dict(raw_case.get("capture", {})),
                "dimensions": dimensions,
                "issues": case_issues,
                "passed": case_passed,
            }
        )

    available_dimensions = [
        dimension
        for case in case_results
        for dimension in case.get("dimensions", [])
        if dimension.get("status") == "available"
    ]
    abs_errors = [float(item["absolute_error_mm"]) for item in available_dimensions]
    rel_errors = [float(item["relative_error_percent"]) for item in available_dimensions]
    coverage_values = [bool(item["covered_by_uncertainty"]) for item in available_dimensions]
    summary = {
        "case_count": len(case_results),
        "passed_case_count": sum(bool(case.get("passed")) for case in case_results),
        "available_dimension_count": len(available_dimensions),
        "mean_absolute_error_mm": statistics.fmean(abs_errors) if abs_errors else None,
        "rmse_mm": math.sqrt(statistics.fmean([value * value for value in abs_errors])) if abs_errors else None,
        "maximum_absolute_error_mm": max(abs_errors) if abs_errors else None,
        "mean_relative_error_percent": statistics.fmean(rel_errors) if rel_errors else None,
        "uncertainty_coverage_percent": (
            100.0 * sum(coverage_values) / len(coverage_values) if coverage_values else None
        ),
    }

    if not thresholds_confirmed:
        global_issues.append("acceptance thresholds are provisional; set thresholds_confirmed=true after review")
    minimum_cases = int(thresholds.get("minimum_cases", 1))
    minimum_dimensions = int(thresholds.get("minimum_available_dimensions", 1))
    if summary["case_count"] < minimum_cases:
        global_issues.append(f"requires at least {minimum_cases} cases")
    if summary["available_dimension_count"] < minimum_dimensions:
        global_issues.append(f"requires at least {minimum_dimensions} available dimensions")
    if any(not case.get("passed", False) for case in case_results):
        global_issues.append("one or more benchmark cases failed")

    aggregate_checks: list[dict[str, Any]] = []
    aggregate_specs = (
        ("mean_absolute_error_mm", "max_mean_absolute_error_mm", "maximum"),
        ("mean_relative_error_percent", "max_mean_relative_error_percent", "maximum"),
        ("maximum_absolute_error_mm", "max_single_dimension_error_mm", "maximum"),
        ("uncertainty_coverage_percent", "minimum_uncertainty_coverage_percent", "minimum"),
    )
    for metric, threshold_key, direction in aggregate_specs:
        value = summary[metric]
        threshold = _as_finite_float(thresholds.get(threshold_key), name=threshold_key)
        passed = value is not None and (value <= threshold if direction == "maximum" else value >= threshold)
        aggregate_checks.append(
            {
                "metric": metric,
                "value": value,
                "threshold": threshold,
                "direction": direction,
                "passed": passed,
            }
        )
        if not passed:
            global_issues.append(f"aggregate threshold failed: {metric}")

    passed = thresholds_confirmed and not global_issues and all(item["passed"] for item in aggregate_checks)
    payload = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "benchmark_name": str(manifest.get("benchmark_name", "Known-size benchmark")),
        "operator": str(manifest.get("operator", "")),
        "measurement_instrument": str(manifest.get("measurement_instrument", "")),
        "manifest_path": str(resolved_manifest),
        "thresholds_confirmed": thresholds_confirmed,
        "thresholds": dict(thresholds),
        "summary": summary,
        "aggregate_checks": aggregate_checks,
        "cases": case_results,
        "issues": list(dict.fromkeys(global_issues)),
        "uncertainty_notice": "Estimated Uncertainty - Not Calibrated",
        "certified_metrology": False,
        "passed": passed,
    }
    return KnownSizeValidationResult(payload)


def _write_csv(path: Path, payload: Mapping[str, Any]) -> None:
    fields = [
        "case_id", "class_name", "observation_id", "ground_truth_dimension",
        "predicted_measurement", "status", "ground_truth_mm", "estimated_mm",
        "expanded_uncertainty_mm", "signed_error_mm", "absolute_error_mm",
        "relative_error_percent", "covered_by_uncertainty", "reliability", "passed",
    ]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for case in payload.get("cases", []):
            for dimension in case.get("dimensions", []):
                writer.writerow(
                    {
                        "case_id": case.get("case_id", ""),
                        "class_name": case.get("class_name", ""),
                        "observation_id": case.get("observation_id", ""),
                        **{key: dimension.get(key, "") for key in fields if key not in {"case_id", "class_name", "observation_id"}},
                    }
                )


def _write_markdown(path: Path, payload: Mapping[str, Any]) -> None:
    summary = payload["summary"]
    lines = [
        f"# {payload['benchmark_name']}",
        "",
        f"**Result:** {'PASS' if payload['passed'] else 'FAIL'}",
        "",
        "> Estimated Uncertainty - Not Calibrated. Engineering validation only; not a certified metrology record.",
        "",
        "## Summary",
        "",
        f"- Cases: {summary['passed_case_count']} / {summary['case_count']} passed",
        f"- Available dimensions: {summary['available_dimension_count']}",
        f"- Mean absolute error: {_fmt(summary['mean_absolute_error_mm'], ' mm')}",
        f"- RMSE: {_fmt(summary['rmse_mm'], ' mm')}",
        f"- Maximum absolute error: {_fmt(summary['maximum_absolute_error_mm'], ' mm')}",
        f"- Mean relative error: {_fmt(summary['mean_relative_error_percent'], '%')}",
        f"- Uncertainty coverage: {_fmt(summary['uncertainty_coverage_percent'], '%')}",
        "",
        "## Cases",
        "",
        "| Case | Object | Dimension | Ground truth | Estimate | Error | Uncertainty | Covered | Result |",
        "|---|---|---|---:|---:|---:|---:|:---:|:---:|",
    ]
    for case in payload.get("cases", []):
        for item in case.get("dimensions", []):
            estimate = _fmt(item.get("estimated_mm"), " mm")
            error = _fmt(item.get("absolute_error_mm"), " mm")
            uncertainty = _fmt(item.get("expanded_uncertainty_mm"), " mm")
            covered = "yes" if item.get("covered_by_uncertainty") is True else ("no" if item.get("covered_by_uncertainty") is False else "-")
            lines.append(
                f"| {case['case_id']} | {case.get('class_name', '')} | {item['ground_truth_dimension']} | "
                f"{item['ground_truth_mm']:.2f} mm | {estimate} | {error} | {uncertainty} | {covered} | "
                f"{'PASS' if item.get('passed') else 'FAIL'} |"
            )
    if payload.get("issues"):
        lines.extend(["", "## Blocking issues", ""])
        lines.extend(f"- {issue}" for issue in payload["issues"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _fmt(value: Any, suffix: str = "") -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.2f}{suffix}"


def _write_html(path: Path, payload: Mapping[str, Any]) -> None:
    summary = payload["summary"]
    rows: list[str] = []
    for case in payload.get("cases", []):
        for item in case.get("dimensions", []):
            rows.append(
                "<tr>"
                f"<td>{html.escape(str(case['case_id']))}</td>"
                f"<td>{html.escape(str(case.get('class_name', '')))}</td>"
                f"<td>{html.escape(str(item['ground_truth_dimension']))}</td>"
                f"<td>{item['ground_truth_mm']:.2f} mm</td>"
                f"<td>{html.escape(_fmt(item.get('estimated_mm'), ' mm'))}</td>"
                f"<td>{html.escape(_fmt(item.get('absolute_error_mm'), ' mm'))}</td>"
                f"<td>{html.escape(_fmt(item.get('relative_error_percent'), '%'))}</td>"
                f"<td>{'yes' if item.get('covered_by_uncertainty') is True else 'no' if item.get('covered_by_uncertainty') is False else '-'}</td>"
                f"<td class={'pass' if item.get('passed') else 'fail'}>{'PASS' if item.get('passed') else 'FAIL'}</td>"
                "</tr>"
            )
    issues = "".join(f"<li>{html.escape(str(issue))}</li>" for issue in payload.get("issues", []))
    document = f"""<!doctype html>
<html><head><meta charset='utf-8'><title>{html.escape(str(payload['benchmark_name']))}</title>
<style>
body{{font-family:Inter,Arial,sans-serif;margin:32px;color:#172033}} .card{{border:1px solid #d8deea;border-radius:12px;padding:18px;margin:14px 0}}
table{{border-collapse:collapse;width:100%;font-size:13px}} th,td{{border-bottom:1px solid #e5e9f1;padding:8px;text-align:left}} .pass{{color:#128243;font-weight:700}} .fail{{color:#c32632;font-weight:700}}
.notice{{background:#fff7dd;border-left:4px solid #d99500;padding:12px}} .metric{{display:inline-block;margin-right:28px}}
</style></head><body>
<h1>{html.escape(str(payload['benchmark_name']))}</h1>
<p class={'pass' if payload['passed'] else 'fail'}>{'PASS' if payload['passed'] else 'FAIL'}</p>
<div class='notice'><strong>Estimated Uncertainty - Not Calibrated.</strong> Engineering validation only; not a certified metrology record.</div>
<div class='card'><h2>Summary</h2>
<span class='metric'>Cases: {summary['passed_case_count']} / {summary['case_count']}</span>
<span class='metric'>MAE: {_fmt(summary['mean_absolute_error_mm'], ' mm')}</span>
<span class='metric'>RMSE: {_fmt(summary['rmse_mm'], ' mm')}</span>
<span class='metric'>Mean relative error: {_fmt(summary['mean_relative_error_percent'], '%')}</span>
<span class='metric'>Uncertainty coverage: {_fmt(summary['uncertainty_coverage_percent'], '%')}</span></div>
<div class='card'><h2>Measurements</h2><table><thead><tr><th>Case</th><th>Object</th><th>Dimension</th><th>Ground truth</th><th>Estimate</th><th>Abs. error</th><th>Rel. error</th><th>Covered</th><th>Result</th></tr></thead><tbody>{''.join(rows)}</tbody></table></div>
<div class='card'><h2>Blocking issues</h2><ul>{issues or '<li>None</li>'}</ul></div>
</body></html>"""
    path.write_text(document, encoding="utf-8")


def write_validation_report(result: KnownSizeValidationResult, output_dir: str | Path) -> Path:
    """Atomically write JSON, CSV, Markdown and HTML validation reports."""

    destination = Path(output_dir).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{destination.name}_", dir=destination.parent) as temporary_name:
        temporary = Path(temporary_name)
        (temporary / "known_size_validation.json").write_text(
            json.dumps(result.payload, indent=2, sort_keys=True), encoding="utf-8"
        )
        _write_csv(temporary / "known_size_measurements.csv", result.payload)
        _write_markdown(temporary / "known_size_validation.md", result.payload)
        _write_html(temporary / "known_size_validation.html", result.payload)
        manifest_entries = []
        for path in sorted(temporary.iterdir()):
            if path.name == "manifest.json" or not path.is_file():
                continue
            manifest_entries.append(
                {"path": path.name, "size_bytes": path.stat().st_size, "sha256": _sha256(path)}
            )
        (temporary / "manifest.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "passed": result.passed,
                    "files": manifest_entries,
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        if destination.exists():
            import shutil
            shutil.rmtree(destination)
        temporary.rename(destination)
    return destination
