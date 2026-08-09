"""Phase 8.7D3 finalization evidence validation and closure reporting.

This module is dependency-light and intentionally does not import any Gaussian
CUDA backend.  It validates JSON evidence emitted by the already-qualified
external workers and audits the production repository isolation boundary.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
EXPECTED_BACKENDS = {"gsplat_external", "graphdeco_reference_external"}
EXPECTED_ROBUSTNESS_SCENARIOS = {"cancellation", "timeout", "crash", "partial_artifact"}
FORBIDDEN_CORE_IMPORTS = (
    "pycolmap",
    "stella_vslam",
    "gsplat",
    "nerfstudio",
    "gaussian_renderer",
    "diff_gaussian_rasterization",
    "simple_knn",
)
FORBIDDEN_PRODUCTION_DEPENDENCIES = (
    "gsplat",
    "nerfstudio",
    "gaussian-splatting",
    "diff-gaussian-rasterization",
    "simple-knn",
)


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _false(value: object) -> bool:
    return value is False


def _backend_results(summary: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    report = summary.get("report")
    if not isinstance(report, Mapping):
        return {}
    values = report.get("results")
    if not isinstance(values, list):
        return {}
    result: dict[str, Mapping[str, Any]] = {}
    for value in values:
        if isinstance(value, Mapping) and isinstance(value.get("backend_kind"), str):
            result[str(value["backend_kind"])] = value
    return result


def _isolation_audit(root: Path) -> dict[str, Any]:
    violations: list[str] = []
    import_pattern = re.compile(r"^\s*(?:from|import)\s+([A-Za-z0-9_\.]+)", re.MULTILINE)
    core_root = root / "core" / "reconstruction"
    for path in sorted(core_root.glob("*.py")):
        text = path.read_text(encoding="utf-8")
        for match in import_pattern.finditer(text):
            imported = match.group(1).lower()
            if any(imported == token or imported.startswith(token + ".") for token in FORBIDDEN_CORE_IMPORTS):
                violations.append(f"heavy_core_import:{path.relative_to(root)}:{imported}")
    pyproject = (root / "pyproject.toml").read_text(encoding="utf-8").lower()
    for token in FORBIDDEN_PRODUCTION_DEPENDENCIES:
        if token in pyproject:
            violations.append(f"production_dependency:{token}")
    app_yaml = (root / "config" / "app.yaml").read_text(encoding="utf-8").lower()
    ui_enabled = "reconstruction_mode_enabled: true" in app_yaml
    if ui_enabled:
        violations.append("reconstruction_ui_enabled")
    return {
        "pass": not violations,
        "violations": violations,
        "reconstruction_ui_enabled": ui_enabled,
        "heavy_core_imports_absent": not any(v.startswith("heavy_core_import:") for v in violations),
        "production_gaussian_dependencies_absent": not any(v.startswith("production_dependency:") for v in violations),
    }


def _validate_d2(summary: Mapping[str, Any]) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    if summary.get("phase") != "8.7D2" or summary.get("pass") is not True:
        reasons.append("d2_not_passed")
    if not _false(summary.get("canonical_geometry_authoritative")):
        reasons.append("d2_canonical_authority_violation")
    if not _false(summary.get("certified_metrology")):
        reasons.append("d2_metrology_violation")
    if summary.get("reconstruction_ui_enabled") is not False:
        reasons.append("d2_ui_state_invalid")
    qualifications = summary.get("qualifications")
    if not isinstance(qualifications, Mapping) or set(qualifications) != {"gsplat", "graphdeco"}:
        reasons.append("d2_backend_set_incomplete")
    else:
        for name, value in qualifications.items():
            if not isinstance(value, Mapping) or value.get("qualified") is not True or value.get("status") != "qualified":
                reasons.append(f"d2_backend_not_qualified:{name}")
    return not reasons, reasons


def _validate_step2(summary: Mapping[str, Any]) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    if summary.get("phase") != "8.7D3" or summary.get("step") != 2 or summary.get("pass") is not True:
        reasons.append("step2_not_passed")
    if summary.get("quality_comparison_ready") is not True:
        reasons.append("step2_quality_not_ready")
    if summary.get("performance_metrics_complete") is not False:
        reasons.append("step2_scope_violation")
    if not _false(summary.get("canonical_geometry_authoritative")) or not _false(summary.get("certified_metrology")):
        reasons.append("step2_authority_violation")
    results = _backend_results(summary)
    if set(results) != EXPECTED_BACKENDS:
        reasons.append("step2_backend_set_incomplete")
    for backend, result in results.items():
        quality = result.get("quality") if isinstance(result, Mapping) else None
        if not isinstance(quality, Mapping) or quality.get("measured") is not True:
            reasons.append(f"step2_quality_missing:{backend}")
        elif any(quality.get(key) is None for key in ("psnr_db", "ssim", "lpips")):
            reasons.append(f"step2_metric_missing:{backend}")
    return not reasons, reasons


def _validate_step3(summary: Mapping[str, Any], expected_hardware_profile: str) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    if summary.get("phase") != "8.7D3" or summary.get("step") != 3 or summary.get("pass") is not True:
        reasons.append("step3_not_passed")
    for key in ("quality_comparison_ready", "performance_metrics_complete", "hardware_evidence_complete", "hardware_consistent_across_backends", "comparison_ready"):
        if summary.get(key) is not True:
            reasons.append(f"step3_not_ready:{key}")
    if summary.get("measured_hardware_profile") != expected_hardware_profile:
        reasons.append("step3_hardware_profile_mismatch")
    if not _false(summary.get("canonical_geometry_authoritative")) or not _false(summary.get("certified_metrology")):
        reasons.append("step3_authority_violation")
    readiness = summary.get("rtx_5090_readiness_profile")
    if not isinstance(readiness, Mapping) or readiness.get("profile") != "rtx_5090_32gb":
        reasons.append("rtx5090_readiness_profile_missing")
    results = _backend_results(summary)
    if set(results) != EXPECTED_BACKENDS:
        reasons.append("step3_backend_set_incomplete")
    for backend, result in results.items():
        performance = result.get("performance") if isinstance(result, Mapping) else None
        hardware = result.get("hardware") if isinstance(result, Mapping) else None
        quality = result.get("quality") if isinstance(result, Mapping) else None
        if not isinstance(performance, Mapping) or performance.get("detailed_measured") is not True:
            reasons.append(f"step3_performance_missing:{backend}")
        else:
            for key in ("peak_training_vram_mib", "peak_render_vram_mib", "training_duration_s", "render_duration_s", "render_fps"):
                value = performance.get(key)
                if not isinstance(value, (int, float)) or float(value) <= 0.0:
                    reasons.append(f"step3_invalid_performance:{backend}:{key}")
        if not isinstance(hardware, Mapping) or hardware.get("measured") is not True or hardware.get("profile_compatible") is not True:
            reasons.append(f"step3_hardware_missing:{backend}")
        if not isinstance(quality, Mapping) or quality.get("measured") is not True:
            reasons.append(f"step3_quality_missing:{backend}")
    return not reasons, reasons


def _validate_step4(summary: Mapping[str, Any]) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    if summary.get("phase") != "8.7D3" or summary.get("step") != 4 or summary.get("pass") is not True:
        reasons.append("step4_not_passed")
    if summary.get("backend_process_scope_only") is not True or summary.get("engine_level_recovery_deferred_to_phase8_7g") is not True:
        reasons.append("step4_scope_violation")
    if not _false(summary.get("canonical_geometry_authoritative")) or not _false(summary.get("certified_metrology")):
        reasons.append("step4_authority_violation")
    report = summary.get("report")
    evidence = report.get("evidence") if isinstance(report, Mapping) else None
    if not isinstance(evidence, list) or len(evidence) != 8:
        reasons.append("step4_evidence_count_invalid")
        evidence = []
    seen: set[tuple[str, str]] = set()
    for item in evidence:
        if not isinstance(item, Mapping):
            reasons.append("step4_invalid_evidence_record")
            continue
        backend = str(item.get("backend_kind", ""))
        scenario = str(item.get("scenario", ""))
        seen.add((backend, scenario))
        if backend not in EXPECTED_BACKENDS or scenario not in EXPECTED_ROBUSTNESS_SCENARIOS:
            reasons.append(f"step4_unknown_case:{backend}:{scenario}")
        for key in ("passed", "process_group_clean", "recovery_result_valid"):
            if item.get(key) is not True:
                reasons.append(f"step4_case_failed:{backend}:{scenario}:{key}")
        recovery = item.get("recovery_lifecycle")
        if not isinstance(recovery, Mapping) or recovery.get("state") != "recovered" or recovery.get("recovery_succeeded") is not True:
            reasons.append(f"step4_recovery_invalid:{backend}:{scenario}")
        if scenario == "partial_artifact":
            if item.get("partial_artifacts_detected") is not True or item.get("partial_artifacts_cleaned") is not True:
                reasons.append(f"step4_partial_artifact_invalid:{backend}")
    expected = {(backend, scenario) for backend in EXPECTED_BACKENDS for scenario in EXPECTED_ROBUSTNESS_SCENARIOS}
    if seen != expected:
        reasons.append("step4_case_matrix_incomplete")
    return not reasons, reasons


def _comparison_from_step3(summary: Mapping[str, Any]) -> dict[str, Any]:
    comparison: dict[str, Any] = {}
    for backend, result in _backend_results(summary).items():
        quality = result.get("quality", {})
        performance = result.get("performance", {})
        hardware = result.get("hardware", {})
        comparison[backend] = {
            "psnr_db": quality.get("psnr_db"),
            "ssim": quality.get("ssim"),
            "lpips": quality.get("lpips"),
            "training_duration_s": performance.get("training_duration_s"),
            "peak_training_vram_mib": performance.get("peak_training_vram_mib"),
            "peak_render_vram_mib": performance.get("peak_render_vram_mib"),
            "render_fps": performance.get("render_fps"),
            "gpu_name": hardware.get("gpu_name"),
            "driver_version": hardware.get("driver_version"),
            "cuda_runtime_version": hardware.get("cuda_runtime_version"),
        }
    return comparison


@dataclass(frozen=True)
class GaussianD3FinalizationResult:
    passed: bool
    summary: dict[str, Any]


def finalize_gaussian_d3(
    *,
    d2_summary_path: Path,
    step2_summary_path: Path,
    step3_summary_path: Path,
    step4_summary_path: Path,
    expected_hardware_profile: str,
    repository_root: Path = ROOT,
) -> GaussianD3FinalizationResult:
    d2 = _load_json(d2_summary_path)
    step2 = _load_json(step2_summary_path)
    step3 = _load_json(step3_summary_path)
    step4 = _load_json(step4_summary_path)
    gates: dict[str, dict[str, Any]] = {}
    for name, validator, payload in (
        ("d2_real_backend_qualification", _validate_d2, d2),
        ("d3_step2_quality", _validate_step2, step2),
        ("d3_step4_robustness", _validate_step4, step4),
    ):
        passed, reasons = validator(payload)
        gates[name] = {"pass": passed, "reasons": reasons}
    passed3, reasons3 = _validate_step3(step3, expected_hardware_profile)
    gates["d3_step3_performance"] = {"pass": passed3, "reasons": reasons3}
    isolation = _isolation_audit(repository_root)
    gates["production_isolation"] = {"pass": isolation["pass"], "reasons": isolation["violations"]}

    passed = all(value["pass"] for value in gates.values())
    summary = {
        "phase": "8.7D3",
        "stage": "finalization",
        "pass": passed,
        "d3_closed": passed,
        "next_phase": "8.7E1" if passed else None,
        "expected_hardware_profile": expected_hardware_profile,
        "gates": gates,
        "isolation": isolation,
        "backend_comparison": _comparison_from_step3(step3),
        "quality_evidence_complete": gates["d3_step2_quality"]["pass"],
        "performance_evidence_complete": gates["d3_step3_performance"]["pass"],
        "robustness_evidence_complete": gates["d3_step4_robustness"]["pass"],
        "real_backends_qualified": gates["d2_real_backend_qualification"]["pass"],
        "canonical_geometry_authoritative": False,
        "certified_metrology": False,
        "gaussian_scene_role": "appearance_only",
        "engine_level_recovery_deferred_to_phase8_7g": True,
        "rtx_5090_real_measurement_claimed": expected_hardware_profile == "rtx_5090_32gb",
        "source_evidence": {
            "d2_summary": str(d2_summary_path.resolve()),
            "step2_summary": str(step2_summary_path.resolve()),
            "step3_summary": str(step3_summary_path.resolve()),
            "step4_summary": str(step4_summary_path.resolve()),
        },
    }
    return GaussianD3FinalizationResult(passed=passed, summary=summary)


def write_finalization_outputs(result: GaussianD3FinalizationResult, output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "d3_final_summary.json"
    json_path.write_text(json.dumps(result.summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    comparison = result.summary["backend_comparison"]
    lines = [
        "# Phase 8.7D3 Final Closure Report",
        "",
        f"- Status: **{'PASS' if result.passed else 'FAIL'}**",
        f"- D3 closed: `{str(result.summary['d3_closed']).lower()}`",
        f"- Next phase: `{result.summary['next_phase']}`",
        f"- Measured hardware profile: `{result.summary['expected_hardware_profile']}`",
        "- Gaussian scene role: `appearance_only`",
        "- Canonical geometry authoritative: `false`",
        "- Certified metrology: `false`",
        "- Whole-engine recovery remains Phase 8.7G.",
        "",
        "## Closure gates",
        "",
    ]
    for name, gate in result.summary["gates"].items():
        lines.append(f"- {name}: **{'PASS' if gate['pass'] else 'FAIL'}**")
        for reason in gate["reasons"]:
            lines.append(f"  - {reason}")
    lines += ["", "## Smoke-scale backend comparison", "", "These values are bounded D3 validation evidence, not production-scale ranking claims.", ""]
    for backend in sorted(comparison):
        values = comparison[backend]
        lines += [
            f"### {backend}",
            "",
            f"- PSNR: `{values.get('psnr_db')}`",
            f"- SSIM: `{values.get('ssim')}`",
            f"- LPIPS: `{values.get('lpips')}`",
            f"- Training duration (s): `{values.get('training_duration_s')}`",
            f"- Peak training VRAM (MiB): `{values.get('peak_training_vram_mib')}`",
            f"- Peak render VRAM (MiB): `{values.get('peak_render_vram_mib')}`",
            f"- Render FPS: `{values.get('render_fps')}`",
            f"- GPU: `{values.get('gpu_name')}`",
            "",
        ]
    md_path = output_dir / "PHASE8_7D3_FINAL_REPORT.md"
    md_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    return json_path, md_path
