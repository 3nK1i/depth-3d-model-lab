"""I5.10C real controlled MP-SfM matching-strategy evaluation.

I5.10C executes exactly one previously admitted counterfactual: the existing
production-native PyCOLMAP sparse-SfM path uses exhaustive matching on the exact
same frozen 60-image KITTI-style window that the historical I5.4 sequential
(overlap=10) baseline rejected after incremental mapping produced no sparse
model.

This module preserves independent arbitrary SfM gauge, PyCOLMAP camera authority,
and the frozen quality policy. It does not rerun the historical baseline, retry
until pass, relax thresholds, reduce the image set, promote exhaustive matching
to the production default, establish metric scale, or start Auto/Hybrid policy.
"""
from __future__ import annotations

from copy import deepcopy
from math import isfinite, sqrt
from pathlib import Path
from statistics import median
from typing import Any, Mapping, Sequence

from core.reconstruction.errors import ReconstructionValidationError

I510C_SCHEMA_VERSION = 1
I510C_PHASE = "I5.10C"
I510C_RECEIPT_PHASE = "I5.10C_EVALUATION_RECEIPT"
I510C_I510B_FROZEN_SHA256 = "0db52edcf9805d66656f874256b5219ded26afe2b25666024c7b9bd3ed02dae4"
I510C_POST_I510B_DELTA_FROZEN_SHA256 = "5fc21a70bc9825036037c5aa953aa67f44dc94f154a795282a50a590388bcf3a"
I510C_PRIMARY_SCENARIO = "kitti_forward_motion"
I510C_INPUT_IMAGES = 60
I510C_EXHAUSTIVE_PAIR_COUNT = 1770
I510C_TIMEOUT_MINUTES = 60.0
I510C_NO_SPARSE_MODEL_TEXT = "PyCOLMAP incremental mapping produced no sparse model"


def _mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ReconstructionValidationError(f"I5.10C {label} must be an object.")
    return value


def _sequence(value: Any, *, label: str) -> Sequence[Any]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ReconstructionValidationError(f"I5.10C {label} must be a sequence.")
    return value


def _require(node: Mapping[str, Any], key: str, expected: Any, *, label: str) -> None:
    actual = node.get(key)
    if actual != expected:
        raise ReconstructionValidationError(
            f"I5.10C {label}.{key} contract drift: expected={expected!r} actual={actual!r}."
        )


def _valid_sha(value: Any) -> bool:
    text = str(value or "").strip().lower()
    return len(text) == 64 and all(ch in "0123456789abcdef" for ch in text)


def _validate_binding(binding: Mapping[str, Any], *, label: str, expected_sha256: str | None = None) -> dict[str, str]:
    node = _mapping(binding, label=label)
    path = str(node.get("path", "")).strip()
    sha = str(node.get("sha256", "")).strip().lower()
    if not path or not _valid_sha(sha):
        raise ReconstructionValidationError(f"I5.10C {label} binding is malformed.")
    if expected_sha256 is not None and sha != expected_sha256.lower():
        raise ReconstructionValidationError(
            f"I5.10C {label} SHA256 drifted: expected={expected_sha256} actual={sha}."
        )
    return {"path": path, "sha256": sha}


def validate_i5_10b_for_i5_10c(payload: Mapping[str, Any]) -> dict[str, Any]:
    root = _mapping(payload, label="I5.10B preflight")
    _require(root, "schema_version", 1, label="I5.10B preflight")
    _require(root, "phase", "I5.10B", label="I5.10B preflight")
    _require(root, "status", "mp_sfm_runtime_and_evidence_preflight_frozen", label="I5.10B preflight")
    _require(root, "pass", True, label="I5.10B preflight")

    baseline = _mapping(root.get("historical_baseline"), label="historical baseline")
    _require(baseline, "baseline_matching_strategy", "sequential", label="historical baseline")
    _require(baseline, "baseline_sequential_overlap", 10, label="historical baseline")
    _require(baseline, "input_images", 60, label="historical baseline")
    _require(baseline, "registered_images", 0, label="historical baseline")
    _require(baseline, "mapping_produced_sparse_model", False, label="historical baseline")
    _require(baseline, "primary_bottleneck", "incremental_mapping_no_sparse_model", label="historical baseline")
    _require(baseline, "quality_status", "rejected", label="historical baseline")

    source = _mapping(root.get("source_revalidation"), label="source revalidation")
    _require(source, "window_image_count", I510C_INPUT_IMAGES, label="source revalidation")
    for key in ("full_source_identity_verified", "exact_60_image_window_verified", "selected_source_sha256_verified"):
        _require(source, key, True, label="source revalidation")
    selected = _sequence(source.get("selected_files"), label="selected files")
    if len(selected) != I510C_INPUT_IMAGES:
        raise ReconstructionValidationError("I5.10C requires exactly 60 selected source records.")
    expected_names = [f"{i:010d}.png" for i in range(I510C_INPUT_IMAGES)]
    actual_names: list[str] = []
    for i, raw in enumerate(selected):
        item = _mapping(raw, label=f"selected_files[{i}]")
        name = str(item.get("name", ""))
        if int(item.get("order", -1)) != i or not _valid_sha(item.get("sha256")) or int(item.get("size_bytes", -1)) <= 0:
            raise ReconstructionValidationError(f"I5.10C selected source record {i} is malformed.")
        actual_names.append(name)
    if actual_names != expected_names:
        raise ReconstructionValidationError("I5.10C exact 60-image name/order window drifted.")

    runtime = _mapping(root.get("runtime_preflight"), label="runtime preflight")
    for key in ("available", "backend_ready", "recommended_version", "required_apis_available"):
        _require(runtime, key, True, label="runtime preflight")
    _require(runtime, "pycolmap_version", "4.1.1", label="runtime preflight")
    apis = _mapping(runtime.get("apis"), label="runtime APIs")
    for key in ("match_sequential", "match_exhaustive", "incremental_mapping"):
        _require(apis, key, True, label="runtime APIs")
    if runtime.get("device_mode") not in {"cuda", "cpu_fallback_explicit"}:
        raise ReconstructionValidationError("I5.10C runtime device mode is not admitted.")

    candidate = _mapping(root.get("candidate_plan"), label="candidate plan")
    frozen_candidate = {
        "attempt_count": 1,
        "baseline_rerun": False,
        "camera_mode": "single",
        "camera_model": "SIMPLE_RADIAL",
        "exhaustive_pair_count": I510C_EXHAUSTIVE_PAIR_COUNT,
        "feature_strategy": "colmap_sift",
        "image_sequence_fps": 10.0,
        "input_images": I510C_INPUT_IMAGES,
        "matching_strategy": "exhaustive",
        "max_keyframes": I510C_INPUT_IMAGES,
        "output_directory_must_be_new_and_immutable": True,
        "profile": "efficient",
        "scenario_id": I510C_PRIMARY_SCENARIO,
        "timeout_minutes": I510C_TIMEOUT_MINUTES,
    }
    for key, expected in frozen_candidate.items():
        _require(candidate, key, expected, label="candidate plan")

    resource = _mapping(root.get("resource_policy"), label="resource policy")
    _require(resource, "input_images", 60, label="resource policy")
    _require(resource, "exhaustive_pair_count", 1770, label="resource policy")
    _require(resource, "adaptive_image_reduction", False, label="resource policy")
    _require(resource, "i5_11_owns_failure_injection", True, label="resource policy")
    _require(resource, "oom_recovery_experiment", False, label="resource policy")
    _require(resource, "disk_failure_experiment", False, label="resource policy")

    freeze = _mapping(root.get("i5_10c_execution_freeze"), label="execution freeze")
    for key in ("authorized", "matching_policy_is_primary_changed_variable", "recognized_quality_rejection_is_scientific_evidence", "runtime_integrity_failure_is_experiment_incomplete", "same_exact_60_image_window_required", "same_exact_source_bytes_required", "sequential_overlap_not_applicable_to_exhaustive"):
        _require(freeze, key, True, label="execution freeze")
    _require(freeze, "candidate_attempt_count", 1, label="execution freeze")
    _require(freeze, "real_candidate_run_count", 1, label="execution freeze")
    for key in ("baseline_rerun", "production_default_modified", "result_driven_retuning", "retry_until_pass", "threshold_relaxation"):
        _require(freeze, key, False, label="execution freeze")

    auth = _mapping(root.get("authority"), label="authority")
    _require(auth, "camera_pose_authority_owner", "pycolmap_sfm", label="authority")
    _require(auth, "candidate_reconstruction_is_evaluation_only", True, label="authority")
    _require(auth, "candidate_reconstruction_uses_pycolmap_authority", True, label="authority")
    _require(auth, "candidate_can_replace_production_default", False, label="authority")
    for key in ("certified_metrology", "measurement_authoritative", "metric_scale_claimed", "surface_mesh_generated"):
        _require(auth, key, False, label="authority")

    decision = _mapping(root.get("decision"), label="decision")
    _require(decision, "i5_10a_complete", True, label="decision")
    _require(decision, "i5_10b_complete", True, label="decision")
    _require(decision, "i5_10c_authorized", True, label="decision")
    for key in ("i5_10c_started", "i5_10d_authorized", "i5_10d_started", "i5_11_started", "i5_12_started", "i5_13_started", "i5_14_started", "production_default_modified", "real_mp_sfm_execution_started", "threshold_relaxation_authorized", "universal_matching_strategy_winner_declared", "auto_policy_implemented", "hybrid_authorized"):
        _require(decision, key, False, label="decision")
    return deepcopy(dict(root))


def validate_i5_10c_receipt(payload: Mapping[str, Any]) -> dict[str, Any]:
    root = _mapping(payload, label="evaluation receipt")
    _require(root, "schema_version", 1, label="evaluation receipt")
    _require(root, "phase", I510C_RECEIPT_PHASE, label="evaluation receipt")
    _require(root, "status", "pass", label="evaluation receipt")
    _require(root, "pass", True, label="evaluation receipt")
    scope = _mapping(root.get("scope"), label="receipt scope")
    _require(scope, "focused_i5_10c_tests", True, label="receipt scope")
    _require(scope, "predecessor_i5_10b_regression", True, label="receipt scope")
    _require(scope, "real_sfm_executed_by_test_receipt", False, label="receipt scope")
    counts = _mapping(root.get("counts"), label="receipt counts")
    focused = counts.get("focused_tests"); predecessor = counts.get("predecessor_total_tests"); combined = counts.get("combined_exercised_tests")
    if not all(isinstance(v, int) and v >= 0 for v in (focused, predecessor, combined)):
        raise ReconstructionValidationError("I5.10C receipt counts are invalid.")
    if focused < 20 or predecessor < 60 or combined != focused + predecessor:
        raise ReconstructionValidationError("I5.10C receipt coverage is below the required floor.")
    rcs = _mapping(root.get("return_codes"), label="receipt return codes")
    for key in ("focused", "predecessor_unit", "predecessor_integration", "predecessor_stress", "import_probe", "runner_help_probe"):
        _require(rcs, key, 0, label="receipt return codes")
    return deepcopy(dict(root))


def build_i5_10c_candidate_command(*, current_python: str, sfm_python: str, staged_window: Path, sfm_output: Path, use_gpu: bool) -> list[str]:
    if not str(current_python).strip() or not str(sfm_python).strip():
        raise ReconstructionValidationError("I5.10C candidate command requires Python interpreters.")
    cmd = [
        str(current_python), "scripts/run_reconstruction_pycolmap_smoke.py",
        "--images", str(Path(staged_window)),
        "--sfm-python", str(sfm_python),
        "--output-dir", str(Path(sfm_output)),
        "--profile", "efficient",
        "--matching", "exhaustive",
        "--max-keyframes", "60",
        "--image-sequence-fps", "10.0",
        "--timeout-minutes", "60.0",
    ]
    if not use_gpu:
        cmd.append("--cpu")
    return cmd


def _matrix4(value: Any, *, label: str) -> list[list[float]]:
    rows = _sequence(value, label=label)
    if len(rows) != 4:
        raise ReconstructionValidationError(f"I5.10C {label} must be 4x4.")
    matrix: list[list[float]] = []
    for row in rows:
        values = [float(x) for x in _sequence(row, label=label)]
        if len(values) != 4 or not all(isfinite(v) for v in values):
            raise ReconstructionValidationError(f"I5.10C {label} must be finite 4x4.")
        matrix.append(values)
    if any(abs(matrix[3][i] - [0.0, 0.0, 0.0, 1.0][i]) > 1e-7 for i in range(4)):
        raise ReconstructionValidationError(f"I5.10C {label} is not homogeneous.")
    return matrix


def _forward_motion_diagnostic(names: list[str], camera_by_name: Mapping[str, list[list[float]]]) -> dict[str, Any]:
    forward: list[float] = []
    lateral: list[float] = []
    records: list[dict[str, Any]] = []
    for a_name, b_name in zip(names[:-1], names[1:]):
        if a_name not in camera_by_name or b_name not in camera_by_name:
            continue
        a = camera_by_name[a_name]; b = camera_by_name[b_name]
        delta = [b[i][3] - a[i][3] for i in range(3)]
        norm = sqrt(sum(v*v for v in delta))
        if not isfinite(norm) or norm <= 1e-12:
            continue
        # R^T * delta where R is reconstruction_from_camera rotation.
        local = [sum(a[r][c] * delta[r] for r in range(3)) for c in range(3)]
        f = abs(local[2]) / norm
        l = sqrt(local[0]*local[0] + local[1]*local[1]) / norm
        forward.append(f); lateral.append(l)
        records.append({"name_a": a_name, "name_b": b_name, "forward_component_fraction": f, "lateral_component_fraction": l})
    mf = float(median(forward)) if forward else None
    ml = float(median(lateral)) if lateral else None
    proven = bool(len(records) >= 2 and mf is not None and ml is not None and mf > ml)
    return {
        "adjacent_registered_motion_pairs": len(records),
        "median_forward_component_fraction": mf,
        "median_lateral_component_fraction": ml,
        "forward_axis_dominant": proven,
        "records": records,
        "scale_invariant": True,
        "absolute_translation_magnitude_cross_run_comparison_permitted": False,
    }


def analyze_i5_10c_sfm_report(report: Mapping[str, Any], *, preflight: Mapping[str, Any]) -> dict[str, Any]:
    frozen = validate_i5_10b_for_i5_10c(preflight)
    root = _mapping(report, label="real SfM report")
    _require(root, "phase", "8.3D", label="real SfM report")
    _require(root, "real_sfm_executed", True, label="real SfM report")
    result = _mapping(root.get("sfm_result"), label="SfM result")
    _require(result, "backend_id", "pycolmap-offline-sfm", label="SfM result")
    gauge = _mapping(result.get("gauge"), label="gauge")
    _require(gauge, "scale_status", "arbitrary", label="gauge")
    _require(gauge, "unit_to_metre", None, label="gauge")
    metadata = _mapping(result.get("metadata"), label="metadata")
    _require(metadata, "real_sfm", True, label="metadata")
    _require(metadata, "matching_strategy", "exhaustive", label="metadata")

    selected = _sequence(frozen["source_revalidation"]["selected_files"], label="selected files")
    names = [str(_mapping(item, label="selected file").get("name")) for item in selected]
    workspace = _mapping(result.get("workspace"), label="workspace")
    raw_images = _sequence(workspace.get("images"), label="workspace images")
    if len(raw_images) != I510C_INPUT_IMAGES:
        raise ReconstructionValidationError("I5.10C workspace does not contain 60 images.")
    key_to_name: dict[str, str] = {}
    for raw in raw_images:
        item = _mapping(raw, label="workspace image")
        key = str(item.get("keyframe_id", ""))
        candidates = [Path(str(item.get("source_locator", ""))).name, Path(str(item.get("staged_image_path", ""))).name]
        name = next((candidate for candidate in candidates if candidate in names), "")
        if not key or not name or key in key_to_name:
            raise ReconstructionValidationError("I5.10C workspace identity drifted from exact frozen source window.")
        key_to_name[key] = name
    if set(key_to_name.values()) != set(names):
        raise ReconstructionValidationError("I5.10C workspace name set drifted from exact frozen source window.")

    cameras = _sequence(result.get("cameras"), label="cameras")
    stats = _mapping(result.get("statistics"), label="statistics")
    registered = int(stats.get("registered_keyframes", -1))
    if int(stats.get("input_keyframes", -1)) != I510C_INPUT_IMAGES or registered != len(cameras) or not (0 <= registered <= I510C_INPUT_IMAGES):
        raise ReconstructionValidationError("I5.10C sparse statistics/camera counts are inconsistent.")
    camera_by_name: dict[str, list[list[float]]] = {}
    for raw in cameras:
        item = _mapping(raw, label="camera")
        key = str(item.get("keyframe_id", ""))
        if key not in key_to_name:
            raise ReconstructionValidationError("I5.10C camera references an unknown workspace keyframe.")
        name = key_to_name[key]
        if name in camera_by_name:
            raise ReconstructionValidationError("I5.10C duplicate registered source name.")
        camera_by_name[name] = _matrix4(item.get("T_reconstruction_from_camera"), label="camera pose")

    quality = _mapping(root.get("quality_assessment", metadata.get("quality_assessment", {})), label="quality assessment")
    quality_raw = str(quality.get("status", ""))
    status_map = {"accepted": "pass", "degraded": "degraded", "rejected": "rejected"}
    if quality_raw not in status_map:
        raise ReconstructionValidationError(f"I5.10C unknown production SfM quality status: {quality_raw!r}.")
    terminal = status_map[quality_raw]
    landmarks = int(stats.get("landmarks", -1)); observations = int(stats.get("observations", -1))
    reproj = stats.get("mean_reprojection_error_px")
    if landmarks < 0 or observations < 0:
        raise ReconstructionValidationError("I5.10C sparse statistics are invalid.")
    if reproj is not None and not isfinite(float(reproj)):
        raise ReconstructionValidationError("I5.10C mean reprojection error must be finite when present.")
    mapping_produced = registered > 0 and landmarks > 0
    return {
        "terminal_status": terminal,
        "quality_status": terminal,
        "production_quality_status": quality_raw,
        "runtime_integrity_failure": False,
        "structured_evidence_kind": "real_sfm_report",
        "input_images": I510C_INPUT_IMAGES,
        "registered_images": registered,
        "registration_ratio": float(registered / I510C_INPUT_IMAGES),
        "mapping_produced_sparse_model": mapping_produced,
        "landmarks": landmarks,
        "observations": observations,
        "mean_reprojection_error_px": None if reproj is None else float(reproj),
        "mean_track_length": quality.get("mean_track_length"),
        "quality_reasons": deepcopy(list(quality.get("reasons", []))) if isinstance(quality.get("reasons", []), list) else [],
        "forward_motion_diagnostic": _forward_motion_diagnostic(names, camera_by_name),
        "gauge": {"scale_status": "arbitrary", "unit_to_metre": None},
        "retry_authorized": False,
        "production_default_promotion_authorized": False,
        "authority": {
            "camera_pose_authority_owner": "pycolmap_sfm",
            "independent_sfm_gauge_preserved": True,
            "metric_scale_claimed": False,
            "measurement_authoritative": False,
            "certified_metrology": False,
            "candidate_reconstruction_is_evaluation_only": True,
        },
    }


def _recognized_no_model_rejection(console_text: str) -> dict[str, Any]:
    line = next((line.strip() for line in reversed(str(console_text).splitlines()) if I510C_NO_SPARSE_MODEL_TEXT in line), I510C_NO_SPARSE_MODEL_TEXT)
    return {
        "terminal_status": "rejected",
        "quality_status": "rejected",
        "production_quality_status": "rejected_without_sparse_report",
        "runtime_integrity_failure": False,
        "structured_evidence_kind": "recognized_incremental_mapping_no_sparse_model_rejection",
        "input_images": 60,
        "registered_images": 0,
        "registration_ratio": 0.0,
        "mapping_produced_sparse_model": False,
        "landmarks": 0,
        "observations": 0,
        "mean_reprojection_error_px": None,
        "mean_track_length": None,
        "quality_reasons": [line],
        "forward_motion_diagnostic": {
            "adjacent_registered_motion_pairs": 0,
            "median_forward_component_fraction": None,
            "median_lateral_component_fraction": None,
            "forward_axis_dominant": False,
            "records": [],
            "scale_invariant": True,
            "absolute_translation_magnitude_cross_run_comparison_permitted": False,
        },
        "gauge": {"scale_status": "unresolved_no_sparse_model", "unit_to_metre": None},
        "retry_authorized": False,
        "production_default_promotion_authorized": False,
        "authority": {
            "camera_pose_authority_owner": "pycolmap_sfm",
            "independent_sfm_gauge_preserved": True,
            "metric_scale_claimed": False,
            "measurement_authoritative": False,
            "certified_metrology": False,
            "candidate_reconstruction_is_evaluation_only": True,
        },
    }


def classify_i5_10c_execution(*, return_code: int, report: Mapping[str, Any] | None, console_text: str, preflight: Mapping[str, Any]) -> dict[str, Any]:
    validate_i5_10b_for_i5_10c(preflight)
    if report is not None and int(return_code) in {0, 1}:
        return analyze_i5_10c_sfm_report(report, preflight=preflight)
    if I510C_NO_SPARSE_MODEL_TEXT in str(console_text):
        return _recognized_no_model_rejection(console_text)
    return {
        "terminal_status": "runtime_integrity_failure",
        "quality_status": None,
        "production_quality_status": None,
        "runtime_integrity_failure": True,
        "structured_evidence_kind": "unrecognized_runtime_integrity_failure",
        "input_images": 60,
        "registered_images": None,
        "registration_ratio": None,
        "mapping_produced_sparse_model": None,
        "quality_reasons": [],
        "runtime_failure": {"return_code": int(return_code), "console_tail": str(console_text)[-4000:]},
        "retry_authorized": False,
        "production_default_promotion_authorized": False,
        "authority": {
            "camera_pose_authority_owner": "pycolmap_sfm",
            "metric_scale_claimed": False,
            "measurement_authoritative": False,
            "certified_metrology": False,
            "candidate_reconstruction_is_evaluation_only": True,
        },
    }


def build_i5_10c_real_evaluation(
    *,
    preflight: Mapping[str, Any],
    preflight_binding: Mapping[str, Any],
    post_i5_10b_delta_binding: Mapping[str, Any],
    evaluation_receipt: Mapping[str, Any],
    evaluation_receipt_binding: Mapping[str, Any],
    execution_analysis: Mapping[str, Any],
    execution_record: Mapping[str, Any],
    source_integrity_before: bool,
    source_integrity_after: bool,
    staged_integrity_after: bool,
    worker_integrity_before: bool,
    worker_integrity_after: bool,
) -> dict[str, Any]:
    frozen = validate_i5_10b_for_i5_10c(preflight)
    receipt = validate_i5_10c_receipt(evaluation_receipt)
    preflight_bind = _validate_binding(preflight_binding, label="I5.10B preflight", expected_sha256=I510C_I510B_FROZEN_SHA256)
    delta_bind = _validate_binding(post_i5_10b_delta_binding, label="post-I5.10B delta checkpoint", expected_sha256=I510C_POST_I510B_DELTA_FROZEN_SHA256)
    receipt_bind = _validate_binding(evaluation_receipt_binding, label="I5.10C evaluation receipt")
    analysis = deepcopy(dict(_mapping(execution_analysis, label="execution analysis")))
    terminal = str(analysis.get("terminal_status", ""))
    if terminal not in {"pass", "degraded", "rejected", "runtime_integrity_failure"}:
        raise ReconstructionValidationError("I5.10C execution analysis has an invalid terminal status.")

    record = deepcopy(dict(_mapping(execution_record, label="execution record")))
    command = _sequence(record.get("candidate_command"), label="candidate command")
    command_text = " ".join(str(x) for x in command)
    if "--matching exhaustive" not in command_text or "--sequential-overlap" in command_text:
        raise ReconstructionValidationError("I5.10C execution command drifted from frozen exhaustive candidate.")
    for key in ("attempt_started_utc", "attempt_finished_utc", "attempt_directory"):
        if not str(record.get(key, "")).strip():
            raise ReconstructionValidationError(f"I5.10C execution record {key} is missing.")
    if not isinstance(record.get("return_code"), int) or not _valid_sha(record.get("console_log_sha256")):
        raise ReconstructionValidationError("I5.10C execution record return code/log SHA is invalid.")
    report_sha = record.get("report_sha256")
    if report_sha is not None and not _valid_sha(report_sha):
        raise ReconstructionValidationError("I5.10C report SHA256 is invalid.")

    integrity = {
        "source_integrity_before": bool(source_integrity_before),
        "source_integrity_after": bool(source_integrity_after),
        "staged_integrity_after": bool(staged_integrity_after),
        "worker_integrity_before": bool(worker_integrity_before),
        "worker_integrity_after": bool(worker_integrity_after),
    }
    scientific_terminal = terminal in {"pass", "degraded", "rejected"}
    if scientific_terminal and not all(integrity.values()):
        raise ReconstructionValidationError("I5.10C cannot interpret reconstruction quality after an integrity failure.")
    complete = scientific_terminal and all(integrity.values())
    phase_pass = complete
    quality_status = terminal if scientific_terminal else "runtime_integrity_failure"

    return {
        "schema_version": I510C_SCHEMA_VERSION,
        "phase": I510C_PHASE,
        "status": "mp_sfm_real_controlled_evaluation_completed" if complete else "mp_sfm_real_controlled_evaluation_incomplete",
        "pass": phase_pass,
        "quality_status": quality_status,
        "terminal_status": terminal,
        "historical_baseline": deepcopy(dict(frozen["historical_baseline"])),
        "candidate_result": analysis,
        "execution_record": record,
        "integrity": integrity,
        "evidence_bindings": {
            "i5_10b_preflight": preflight_bind,
            "post_i5_10b_delta_checkpoint": delta_bind,
            "i5_10c_evaluation_receipt": receipt_bind,
            "post_i5_10a_source_checkpoint": deepcopy(dict(frozen["evidence_bindings"]["post_i5_10a_source_checkpoint"])),
        },
        "validation_receipt": receipt,
        "source_contract": {
            "same_exact_60_image_window_preserved": True,
            "same_exact_source_bytes_preserved": bool(source_integrity_before and source_integrity_after and staged_integrity_after),
            "window_asset_set_sha256": frozen["source_revalidation"]["window_asset_set_sha256"],
            "baseline_rerun": False,
        },
        "comparison_boundary": {
            "matching_policy_is_primary_changed_variable": True,
            "historical_baseline_matching": "sequential",
            "historical_baseline_overlap": 10,
            "candidate_matching": "exhaustive",
            "absolute_pose_or_translation_cross_run_comparison_allowed": False,
            "metric_scale_comparison_allowed": False,
            "runtime_performance_ranking_authorized": False,
            "production_default_promotion_authorized": False,
            "universal_matching_strategy_winner_authorized": False,
        },
        "scope": {
            "real_mp_sfm_execution": True,
            "real_candidate_run_count": 1,
            "baseline_rerun": False,
            "retry_until_pass": False,
            "result_driven_retuning": False,
            "threshold_relaxation": False,
            "adaptive_image_reduction": False,
            "production_default_modified": False,
            "sfm_candidate_only": True,
            "mvs_rerun": False,
            "provider_inference": False,
            "oom_recovery_experiment": False,
            "disk_failure_experiment": False,
            "auto_policy_implemented": False,
            "hybrid_authorized": False,
            "universal_matching_strategy_winner_declared": False,
        },
        "authority": {
            "camera_pose_authority_owner": "pycolmap_sfm",
            "candidate_reconstruction_is_evaluation_only": True,
            "candidate_can_replace_production_default": False,
            "independent_sfm_gauges_preserved": True,
            "metric_scale_claimed": False,
            "measurement_authoritative": False,
            "certified_metrology": False,
            "surface_mesh_generated": False,
            "permanent_global_view_cap": None,
        },
        "decision": {
            "i5_10a_complete": True,
            "i5_10b_complete": True,
            "i5_10c_complete": complete,
            "i5_10d_authorized": complete,
            "i5_10d_started": False,
            "i5_11_started": False,
            "i5_12_started": False,
            "i5_13_started": False,
            "i5_14_started": False,
            "real_mp_sfm_execution_started": True,
            "runtime_integrity_failure": terminal == "runtime_integrity_failure" or not all(integrity.values()),
            "production_default_modified": False,
            "threshold_relaxation_authorized": False,
            "universal_matching_strategy_winner_declared": False,
            "auto_policy_implemented": False,
            "hybrid_authorized": False,
        },
        "next": "I5.10D MP-SfM evidence-only qualification and closure." if complete else "Diagnose I5.10C runtime/integrity failure without interpreting it as reconstruction quality.",
    }


__all__ = [
    "I510C_SCHEMA_VERSION", "I510C_PHASE", "I510C_RECEIPT_PHASE",
    "I510C_I510B_FROZEN_SHA256", "I510C_POST_I510B_DELTA_FROZEN_SHA256",
    "validate_i5_10b_for_i5_10c", "validate_i5_10c_receipt",
    "build_i5_10c_candidate_command", "analyze_i5_10c_sfm_report",
    "classify_i5_10c_execution", "build_i5_10c_real_evaluation",
]
