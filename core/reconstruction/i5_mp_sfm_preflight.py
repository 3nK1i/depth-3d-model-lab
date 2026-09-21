"""I5.10B MP-SfM runtime/evidence preflight.

This step is deliberately execution-free. It binds the frozen I5.10A admission,
the historical I5.4A/B/C forward-motion evidence chain, the post-I5.10A
no-Git source checkpoint, the exact KITTI-style source bytes, and the current
PyCOLMAP production runtime. It freezes one exhaustive-matching candidate
command for I5.10C without executing SfM.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
from pathlib import Path
from typing import Any, Mapping, Sequence

from core.reconstruction.errors import ReconstructionValidationError
from core.reconstruction.i5_difficult_handheld_robustness import I51EvidenceBinding
from core.reconstruction.i5_forward_motion_kitti import (
    build_i5_4a_source_identity,
    validate_i5_4a_for_i5_4b,
    validate_i5_4b_for_i5_4c,
)
from core.reconstruction.sfm_environment import PYCOLMAP_RECOMMENDED_VERSION, PYCOLMAP_REQUIRED_APIS

I510B_SCHEMA_VERSION = 1
I510B_PHASE = "I5.10B"
I510B_RECEIPT_PHASE = "I5.10B_PREFLIGHT_RECEIPT"
I510B_I510A_FROZEN_SHA256 = "fca08621136404bd5ecdb7b861123f28f4c624dd3b38be0c35e51989a4f4e2db"
I510B_PRIMARY_SCENARIO = "kitti_forward_motion"
I510B_INPUT_IMAGES = 60
I510B_EXHAUSTIVE_PAIR_COUNT = I510B_INPUT_IMAGES * (I510B_INPUT_IMAGES - 1) // 2
I510B_TIMEOUT_MINUTES = 60.0

I510B_REPOSITORY_ANCHORS: dict[str, str] = {
    "scripts/run_reconstruction_pycolmap_smoke.py": "4585bc467166f8e9d4d689f97cb04b0fbefddb33c76d27ad5f4457192741180b",
    "scripts/reconstruction_pycolmap_worker.py": "75277ced2640d61b7ebadb1541b1b28ae5b6dc8f5ba23183fb20e442cdd97dac",
    "core/reconstruction/sfm_pycolmap.py": "bdcc8a9cd2a885c51000f795ebe2edbc9a43449441fdbfae42c1d3e2f9455f59",
    "core/reconstruction/sfm_environment.py": "7195c15e1e31ad7079ab9671c10cd85d10b7a28274d3217b8e3e61ff0daa9b4e",
    "core/reconstruction/sfm_contracts.py": "09c3044732b34e88ab1170fc6d413d586445b5f8662a814950b3542bf57b7cd0",
    "core/reconstruction/sfm_quality.py": "66e75d1bd0ae2b6ebe78705f90bc5cd58c8f63287ccfa83f9fb45fb6a5e27a1f",
    "core/reconstruction/i5_forward_motion_kitti.py": "6daae700868fe8da6cda9f8a85548ba70a46340849853cd82386776ff44f6855",
    "core/reconstruction/i5_mp_sfm_evaluation.py": "db70b70fd4d5d5fc09cb094da75700e313df5ea838749fdc39acf1a3e6417f97",
}


def _mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ReconstructionValidationError(f"I5.10B {label} must be an object.")
    return value


def _require(node: Mapping[str, Any], key: str, expected: Any, *, label: str) -> None:
    actual = node.get(key)
    if actual != expected:
        raise ReconstructionValidationError(
            f"I5.10B {label}.{key} contract drift: expected={expected!r} actual={actual!r}."
        )


def _valid_sha(value: Any) -> bool:
    text = str(value or "").strip().lower()
    return len(text) == 64 and all(ch in "0123456789abcdef" for ch in text)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_binding(
    binding: I51EvidenceBinding,
    *,
    label: str,
    expected_sha256: str | None = None,
) -> None:
    if not str(binding.path).strip() or not _valid_sha(binding.sha256):
        raise ReconstructionValidationError(f"I5.10B {label} binding is malformed.")
    if expected_sha256 is not None and binding.sha256.lower() != expected_sha256.lower():
        raise ReconstructionValidationError(
            f"I5.10B {label} SHA256 drifted: expected={expected_sha256} actual={binding.sha256}."
        )


def verify_i5_10b_repository_anchors(repository_root: Path) -> dict[str, dict[str, Any]]:
    root = Path(repository_root).expanduser().resolve()
    if not root.is_dir():
        raise ReconstructionValidationError(f"I5.10B repository root missing: {root}")
    out: dict[str, dict[str, Any]] = {}
    for relative, expected in I510B_REPOSITORY_ANCHORS.items():
        path = root / relative
        if not path.is_file():
            raise ReconstructionValidationError(f"I5.10B repository anchor missing: {relative}")
        actual = _sha256(path)
        if actual != expected:
            raise ReconstructionValidationError(
                f"I5.10B repository anchor drifted: {relative} expected={expected} actual={actual}"
            )
        out[relative] = {"path": str(path), "sha256": actual, "verified": True}
    return out


def validate_i5_10a_for_i5_10b(payload: Mapping[str, Any]) -> dict[str, Any]:
    root = _mapping(payload, label="I5.10A admission")
    _require(root, "schema_version", 1, label="I5.10A admission")
    _require(root, "phase", "I5.10A", label="I5.10A admission")
    _require(root, "status", "mp_sfm_inventory_and_experiment_admitted", label="I5.10A admission")
    _require(root, "pass", True, label="I5.10A admission")

    design = _mapping(root.get("experiment_design"), label="I5.10A experiment design")
    primary = _mapping(design.get("primary_scenario"), label="I5.10A primary scenario")
    candidate = _mapping(design.get("candidate"), label="I5.10A candidate")
    _require(primary, "scenario_id", I510B_PRIMARY_SCENARIO, label="I5.10A primary scenario")
    _require(primary, "historical_baseline_matching", "sequential", label="I5.10A primary scenario")
    _require(primary, "historical_baseline_overlap", 10, label="I5.10A primary scenario")
    _require(primary, "historical_baseline_quality", "rejected", label="I5.10A primary scenario")
    _require(primary, "historical_baseline_failure", "incremental_mapping_produced_no_sparse_model", label="I5.10A primary scenario")
    _require(primary, "baseline_reused_not_rerun", True, label="I5.10A primary scenario")
    _require(candidate, "matching_strategy", "exhaustive", label="I5.10A candidate")
    _require(candidate, "attempt_count", 1, label="I5.10A candidate")
    _require(candidate, "reruns_historical_sequential_baseline", False, label="I5.10A candidate")
    _require(candidate, "same_exact_source_window_required", True, label="I5.10A candidate")
    _require(candidate, "same_exact_source_bytes_required", True, label="I5.10A candidate")
    _require(candidate, "same_profile", "efficient", label="I5.10A candidate")
    _require(candidate, "same_max_keyframes", 60, label="I5.10A candidate")
    _require(candidate, "same_image_sequence_fps", 10.0, label="I5.10A candidate")
    _require(candidate, "same_feature_strategy", "colmap_sift", label="I5.10A candidate")
    _require(candidate, "same_camera_mode", "single", label="I5.10A candidate")
    _require(candidate, "same_camera_model", "SIMPLE_RADIAL", label="I5.10A candidate")

    requirements = _mapping(root.get("i5_10b_preflight_requirements"), label="I5.10A B requirements")
    for key in (
        "bind_i5_10a_admission", "bind_i5_4a_admission", "bind_i5_4b_replay", "bind_i5_4c_closure",
        "revalidate_i5_4_predecessor_chain", "revalidate_full_source_identity",
        "revalidate_exact_60_image_window", "revalidate_selected_source_sha256",
        "probe_pycolmap_4_1_1_and_required_apis", "require_match_sequential_api",
        "require_match_exhaustive_api", "require_cuda_or_explicitly_record_cpu_fallback",
        "freeze_candidate_command", "freeze_source_and_worker_hashes",
    ):
        _require(requirements, key, True, label="I5.10A B requirements")
    _require(requirements, "execute_sfm", False, label="I5.10A B requirements")

    decision = _mapping(root.get("decision"), label="I5.10A decision")
    _require(decision, "i5_10a_complete", True, label="I5.10A decision")
    _require(decision, "i5_10b_authorized", True, label="I5.10A decision")
    for key in (
        "i5_10b_started", "i5_10c_authorized", "i5_10c_started", "i5_10d_authorized",
        "i5_10d_started", "i5_11_started", "i5_12_started", "i5_13_started", "i5_14_started",
        "real_mp_sfm_execution_started", "auto_policy_implemented", "hybrid_authorized",
        "production_default_modified",
    ):
        _require(decision, key, False, label="I5.10A decision")
    return deepcopy(dict(root))


def _validate_i5_4c_for_i5_10b(payload: Mapping[str, Any]) -> dict[str, Any]:
    root = _mapping(payload, label="I5.4C closure")
    _require(root, "schema_version", 1, label="I5.4C closure")
    _require(root, "phase", "I5.4C", label="I5.4C closure")
    _require(root, "status", "forward_motion_kitti_robustness_qualified_and_closed", label="I5.4C closure")
    _require(root, "pass", True, label="I5.4C closure")
    _require(root, "quality_status", "rejected", label="I5.4C closure")
    scope = _mapping(root.get("scope"), label="I5.4C scope")
    _require(scope, "evidence_only_qualification_and_closure", True, label="I5.4C scope")
    for key in ("sfm_rerun", "source_rematerialized", "window_retuned", "thresholds_modified", "production_default_modified"):
        _require(scope, key, False, label="I5.4C scope")
    q = _mapping(root.get("qualification"), label="I5.4C qualification")
    _require(q, "primary_bottleneck", "incremental_mapping_no_sparse_model", label="I5.4C qualification")
    _require(q, "registered_images", 0, label="I5.4C qualification")
    _require(q, "input_images", 60, label="I5.4C qualification")
    _require(q, "mapping_produced_sparse_model", False, label="I5.4C qualification")
    _require(q, "authoritative_forward_motion_geometry_proven", False, label="I5.4C qualification")
    return deepcopy(dict(root))


def validate_i5_4_chain_for_i5_10b(
    i5_4a_admission: Mapping[str, Any],
    i5_4b_replay: Mapping[str, Any],
    i5_4c_closure: Mapping[str, Any],
    *,
    i5_4a_sha256: str,
    i5_4b_sha256: str,
) -> dict[str, Any]:
    a = validate_i5_4a_for_i5_4b(i5_4a_admission)
    b = validate_i5_4b_for_i5_4c(i5_4b_replay)
    c = _validate_i5_4c_for_i5_10b(i5_4c_closure)
    if not _valid_sha(i5_4a_sha256) or not _valid_sha(i5_4b_sha256):
        raise ReconstructionValidationError("I5.10B I5.4 chain SHA256 values are malformed.")
    b_a = _mapping(_mapping(b.get("evidence_bindings"), label="I5.4B bindings").get("i5_4a_admission"), label="I5.4B A binding")
    if str(b_a.get("sha256", "")).lower() != i5_4a_sha256.lower():
        raise ReconstructionValidationError("I5.10B I5.4B does not bind the supplied I5.4A admission SHA256.")
    c_b = _mapping(_mapping(c.get("evidence_bindings"), label="I5.4C bindings").get("i5_4b_real_replay"), label="I5.4C B binding")
    if str(c_b.get("sha256", "")).lower() != i5_4b_sha256.lower():
        raise ReconstructionValidationError("I5.10B I5.4C does not bind the supplied I5.4B replay SHA256.")

    a_source = _mapping(a.get("source_identity"), label="I5.4A source identity")
    a_window = _mapping(_mapping(a.get("stress_plan"), label="I5.4A stress plan").get("forward_motion_window"), label="I5.4A window")
    b_source = _mapping(b.get("source_binding"), label="I5.4B source binding")
    if str(a_source.get("asset_set_sha256", "")).lower() != str(b_source.get("source_asset_set_sha256", "")).lower():
        raise ReconstructionValidationError("I5.10B I5.4 full source binding drifted between A and B.")
    if str(a_window.get("window_asset_set_sha256", "")).lower() != str(b_source.get("window_asset_set_sha256", "")).lower():
        raise ReconstructionValidationError("I5.10B I5.4 60-image window binding drifted between A and B.")

    return {
        "chain_verified": True,
        "baseline_matching_strategy": "sequential",
        "baseline_sequential_overlap": 10,
        "input_images": 60,
        "registered_images": 0,
        "mapping_produced_sparse_model": False,
        "quality_status": "rejected",
        "primary_bottleneck": "incremental_mapping_no_sparse_model",
        "authoritative_forward_motion_geometry_proven": False,
        "source_asset_set_sha256": str(a_source["asset_set_sha256"]).lower(),
        "window_asset_set_sha256": str(a_window["window_asset_set_sha256"]).lower(),
    }


def revalidate_i5_10b_source(i5_4a_admission: Mapping[str, Any], dataset_dir: Path) -> dict[str, Any]:
    a = validate_i5_4a_for_i5_4b(i5_4a_admission)
    admitted_identity = _mapping(a.get("source_identity"), label="I5.4A source identity")
    dataset = Path(dataset_dir).expanduser().resolve()
    admitted_dir = Path(str(admitted_identity.get("directory", ""))).expanduser().resolve()
    if dataset != admitted_dir:
        raise ReconstructionValidationError(
            f"I5.10B dataset path drifted from I5.4A: admitted={admitted_dir} requested={dataset}"
        )
    current = build_i5_4a_source_identity(dataset)
    if current["asset_set_sha256"] != admitted_identity["asset_set_sha256"]:
        raise ReconstructionValidationError("I5.10B full source asset-set SHA256 drifted from I5.4A.")
    if current["ordered_names"] != admitted_identity["ordered_names"]:
        raise ReconstructionValidationError("I5.10B full source ordered-name identity drifted from I5.4A.")

    plan = _mapping(_mapping(a.get("stress_plan"), label="I5.4A stress plan").get("forward_motion_window"), label="I5.4A window")
    names = [str(x) for x in plan.get("names", [])]
    if len(names) != 60 or list(plan.get("indices", [])) != list(range(60)):
        raise ReconstructionValidationError("I5.10B requires the exact frozen first 60 native-order I5.4 images.")
    records = {str(x["name"]): x for x in current["files"]}
    selected: list[dict[str, Any]] = []
    aggregate = hashlib.sha256()
    for index, name in enumerate(names):
        record = records.get(name)
        if record is None:
            raise ReconstructionValidationError(f"I5.10B frozen source image missing from current identity: {name}")
        sha = str(record["sha256"]).lower()
        size = int(record["size_bytes"])
        selected.append({"order": index, "name": name, "size_bytes": size, "sha256": sha})
        aggregate.update(str(index).encode("ascii")); aggregate.update(b"\0")
        aggregate.update(name.encode("utf-8")); aggregate.update(b"\0")
        aggregate.update(str(size).encode("ascii")); aggregate.update(b"\0")
        aggregate.update(sha.encode("ascii")); aggregate.update(b"\n")
    window_sha = aggregate.hexdigest()
    if window_sha != str(plan.get("window_asset_set_sha256", "")).lower():
        raise ReconstructionValidationError("I5.10B selected 60-image window SHA256 drifted from I5.4A.")
    return {
        "dataset_directory": str(dataset),
        "full_source_image_count": int(current["image_count"]),
        "full_source_asset_set_sha256": str(current["asset_set_sha256"]).lower(),
        "full_source_identity_verified": True,
        "window_image_count": 60,
        "window_asset_set_sha256": window_sha,
        "exact_60_image_window_verified": True,
        "selected_source_sha256_verified": True,
        "selected_files": selected,
        "images_transformed": False,
        "source_materialized": False,
    }


def validate_i5_10b_runtime_probe(payload: Mapping[str, Any]) -> dict[str, Any]:
    root = deepcopy(dict(_mapping(payload, label="PyCOLMAP runtime probe")))
    _require(root, "available", True, label="PyCOLMAP runtime probe")
    _require(root, "pycolmap_version", PYCOLMAP_RECOMMENDED_VERSION, label="PyCOLMAP runtime probe")
    _require(root, "backend_ready", True, label="PyCOLMAP runtime probe")
    apis = _mapping(root.get("apis"), label="PyCOLMAP APIs")
    for name in PYCOLMAP_REQUIRED_APIS:
        _require(apis, name, True, label="PyCOLMAP APIs")
    for name in ("match_sequential", "match_exhaustive"):
        _require(apis, name, True, label="PyCOLMAP APIs")
    cuda_devices = int(root.get("cuda_devices", 0))
    root["cuda_devices"] = cuda_devices
    root["device_mode"] = "cuda" if cuda_devices > 0 else "cpu_fallback_explicit"
    root["cpu_fallback"] = cuda_devices == 0
    root["recommended_version"] = True
    root["required_apis_available"] = True
    return root


def build_i5_10b_candidate_command_template(sfm_python: str, *, use_gpu: bool) -> list[str]:
    executable = str(sfm_python).strip()
    if not executable:
        raise ReconstructionValidationError("I5.10B sfm_python must be non-empty.")
    command = [
        "${CURRENT_PYTHON}",
        "scripts/run_reconstruction_pycolmap_smoke.py",
        "--images", "${I510C_STAGED_WINDOW}",
        "--sfm-python", executable,
        "--output-dir", "${I510C_SFM_OUTPUT}",
        "--profile", "efficient",
        "--matching", "exhaustive",
        "--max-keyframes", "60",
        "--image-sequence-fps", "10.0",
        "--timeout-minutes", str(I510B_TIMEOUT_MINUTES),
    ]
    if not use_gpu:
        command.append("--cpu")
    return command


def validate_i5_10b_receipt(payload: Mapping[str, Any]) -> dict[str, Any]:
    root = _mapping(payload, label="I5.10B receipt")
    _require(root, "phase", I510B_RECEIPT_PHASE, label="I5.10B receipt")
    _require(root, "status", "pass", label="I5.10B receipt")
    _require(root, "pass", True, label="I5.10B receipt")
    scope = _mapping(root.get("scope"), label="I5.10B receipt scope")
    for key in (
        "focused_i5_10b_tests", "i5_10a_regression", "i5_4_chain_regression", "source_identity_regression",
        "pycolmap_environment_regression", "candidate_command_regression", "authority_guard_regression",
        "roadmap_guard_regression", "import_probe", "runner_help_probe",
    ):
        _require(scope, key, True, label="I5.10B receipt scope")
    for key in (
        "real_mp_sfm_execution", "sfm_rerun", "mvs_rerun", "thresholds_modified", "production_default_modified",
        "retry_until_pass", "auto_policy_implemented", "hybrid_authorized",
    ):
        _require(scope, key, False, label="I5.10B receipt scope")
    rcs = _mapping(root.get("return_codes"), label="I5.10B receipt return codes")
    for key in ("focused", "predecessor_unit", "predecessor_integration", "predecessor_stress", "import_probe", "runner_help_probe"):
        _require(rcs, key, 0, label="I5.10B receipt return codes")
    counts = _mapping(root.get("counts"), label="I5.10B receipt counts")
    focused = int(counts.get("focused_tests", -1)); pred = int(counts.get("predecessor_total_tests", -1)); combined = int(counts.get("combined_exercised_tests", -1))
    if focused <= 0 or pred <= 0 or combined != focused + pred:
        raise ReconstructionValidationError("I5.10B receipt test counts are inconsistent.")
    return deepcopy(dict(root))


def build_i5_10b_mp_sfm_preflight(
    *,
    i5_10a_admission: Mapping[str, Any],
    i5_10a_binding: I51EvidenceBinding,
    i5_4a_admission: Mapping[str, Any],
    i5_4a_binding: I51EvidenceBinding,
    i5_4b_replay: Mapping[str, Any],
    i5_4b_binding: I51EvidenceBinding,
    i5_4c_closure: Mapping[str, Any],
    i5_4c_binding: I51EvidenceBinding,
    source_revalidation: Mapping[str, Any],
    runtime_probe: Mapping[str, Any],
    post_i5_10a_checkpoint_binding: I51EvidenceBinding,
    preflight_receipt: Mapping[str, Any],
    preflight_receipt_binding: I51EvidenceBinding,
    repository_anchors: Mapping[str, Mapping[str, Any]] | None,
) -> dict[str, Any]:
    a10 = validate_i5_10a_for_i5_10b(i5_10a_admission)
    _validate_binding(i5_10a_binding, label="I5.10A admission", expected_sha256=I510B_I510A_FROZEN_SHA256)
    for binding, label in (
        (i5_4a_binding, "I5.4A admission"), (i5_4b_binding, "I5.4B replay"),
        (i5_4c_binding, "I5.4C closure"), (post_i5_10a_checkpoint_binding, "post-I5.10A source checkpoint"),
        (preflight_receipt_binding, "I5.10B preflight receipt"),
    ):
        _validate_binding(binding, label=label)
    chain = validate_i5_4_chain_for_i5_10b(
        i5_4a_admission, i5_4b_replay, i5_4c_closure,
        i5_4a_sha256=i5_4a_binding.sha256, i5_4b_sha256=i5_4b_binding.sha256,
    )
    source = deepcopy(dict(_mapping(source_revalidation, label="source revalidation")))
    for key in ("full_source_identity_verified", "exact_60_image_window_verified", "selected_source_sha256_verified"):
        _require(source, key, True, label="source revalidation")
    _require(source, "window_image_count", 60, label="source revalidation")
    if source.get("full_source_asset_set_sha256") != chain["source_asset_set_sha256"]:
        raise ReconstructionValidationError("I5.10B source revalidation does not match the frozen I5.4 source SHA256.")
    if source.get("window_asset_set_sha256") != chain["window_asset_set_sha256"]:
        raise ReconstructionValidationError("I5.10B source revalidation does not match the frozen I5.4 window SHA256.")
    runtime = validate_i5_10b_runtime_probe(runtime_probe)
    receipt = validate_i5_10b_receipt(preflight_receipt)

    if repository_anchors is None:
        anchors = {k: {"sha256": v, "verified": True} for k, v in I510B_REPOSITORY_ANCHORS.items()}
    else:
        anchors = deepcopy(dict(repository_anchors))
        if set(anchors) != set(I510B_REPOSITORY_ANCHORS):
            raise ReconstructionValidationError("I5.10B repository anchor set drifted.")
        for relative, expected in I510B_REPOSITORY_ANCHORS.items():
            node = _mapping(anchors[relative], label=f"repository anchor {relative}")
            _require(node, "sha256", expected, label=f"repository anchor {relative}")
            _require(node, "verified", True, label=f"repository anchor {relative}")

    command = build_i5_10b_candidate_command_template(
        str(runtime["python_executable"]), use_gpu=runtime["device_mode"] == "cuda"
    )
    return {
        "schema_version": I510B_SCHEMA_VERSION,
        "phase": I510B_PHASE,
        "status": "mp_sfm_runtime_and_evidence_preflight_frozen",
        "pass": True,
        "scope": {
            "execution_free": True,
            "runtime_preflight": True,
            "evidence_chain_revalidated": True,
            "source_identity_revalidated": True,
            "candidate_command_frozen": True,
            "real_mp_sfm_execution": False,
            "sfm_rerun": False,
            "bundle_adjustment_rerun": False,
            "mvs_rerun": False,
            "provider_inference": False,
            "source_materialized": False,
            "thresholds_modified": False,
            "production_default_modified": False,
            "result_driven_retuning": False,
            "retry_until_pass": False,
            "auto_policy_implemented": False,
            "hybrid_authorized": False,
        },
        "evidence_bindings": {
            "i5_10a_admission": i5_10a_binding.as_dict(),
            "i5_4a_admission": i5_4a_binding.as_dict(),
            "i5_4b_replay": i5_4b_binding.as_dict(),
            "i5_4c_closure": i5_4c_binding.as_dict(),
            "post_i5_10a_source_checkpoint": post_i5_10a_checkpoint_binding.as_dict(),
            "i5_10b_preflight_receipt": preflight_receipt_binding.as_dict(),
        },
        "source_control_policy": {
            "git_or_gitea_required": False,
            "immutable_checkpoint_sha_required": True,
            "checkpoint_role": "post_i5_10a_source_baseline",
            "github_migration_deferred_until_explicit_operator_decision": True,
        },
        "repository_anchors": anchors,
        "historical_baseline": chain,
        "source_revalidation": source,
        "runtime_preflight": runtime,
        "candidate_plan": {
            "scenario_id": I510B_PRIMARY_SCENARIO,
            "matching_strategy": "exhaustive",
            "input_images": 60,
            "exhaustive_pair_count": I510B_EXHAUSTIVE_PAIR_COUNT,
            "attempt_count": 1,
            "baseline_rerun": False,
            "profile": "efficient",
            "max_keyframes": 60,
            "image_sequence_fps": 10.0,
            "feature_strategy": "colmap_sift",
            "camera_mode": "single",
            "camera_model": "SIMPLE_RADIAL",
            "timeout_minutes": I510B_TIMEOUT_MINUTES,
            "device_mode": runtime["device_mode"],
            "cpu_fallback": runtime["cpu_fallback"],
            "command_template": command,
            "staged_window_placeholder": "${I510C_STAGED_WINDOW}",
            "output_placeholder": "${I510C_SFM_OUTPUT}",
            "output_directory_must_be_new_and_immutable": True,
        },
        "resource_policy": {
            "input_images": 60,
            "exhaustive_pair_count": I510B_EXHAUSTIVE_PAIR_COUNT,
            "adaptive_image_reduction": False,
            "cuda_preferred": True,
            "cpu_fallback_must_be_explicit": True,
            "timeout_minutes": I510B_TIMEOUT_MINUTES,
            "oom_recovery_experiment": False,
            "disk_failure_experiment": False,
            "i5_11_owns_failure_injection": True,
        },
        "i5_10c_evidence_schema": {
            "schema_version": 1,
            "attempt_directory_must_be_new_and_immutable": True,
            "required_pre_execution_bindings": [
                "i5_10b_preflight",
                "post_i5_10a_source_checkpoint",
                "exact_60_image_source_manifest",
                "runtime_probe",
                "repository_anchors",
            ],
            "required_execution_records": [
                "candidate_command",
                "attempt_started_utc",
                "attempt_finished_utc",
                "return_code",
                "console_log_sha256",
                "source_integrity_after",
                "worker_integrity_after",
            ],
            "terminal_outcomes": [
                "pass",
                "degraded",
                "rejected",
                "runtime_integrity_failure",
            ],
            "real_sfm_report_required_when_sparse_model_exists": True,
            "recognized_no_sparse_model_rejection_preserved_without_report": True,
            "runtime_integrity_failure_is_not_quality_rejection": True,
            "no_retry_artifact_overwrite": True,
        },
        "i5_10c_execution_freeze": {
            "authorized": True,
            "real_candidate_run_count": 1,
            "candidate_attempt_count": 1,
            "baseline_rerun": False,
            "same_exact_source_bytes_required": True,
            "same_exact_60_image_window_required": True,
            "matching_policy_is_primary_changed_variable": True,
            "sequential_overlap_not_applicable_to_exhaustive": True,
            "retry_until_pass": False,
            "result_driven_retuning": False,
            "threshold_relaxation": False,
            "production_default_modified": False,
            "recognized_quality_rejection_is_scientific_evidence": True,
            "runtime_integrity_failure_is_experiment_incomplete": True,
        },
        "comparison_contract": deepcopy(dict(_mapping(a10.get("comparison_contract"), label="I5.10A comparison contract"))),
        "authority": {
            "camera_pose_authority_owner": "pycolmap_sfm",
            "candidate_reconstruction_uses_pycolmap_authority": True,
            "candidate_reconstruction_is_evaluation_only": True,
            "candidate_can_replace_production_default": False,
            "independent_sfm_gauges_preserved": True,
            "metric_scale_claimed": False,
            "measurement_authoritative": False,
            "certified_metrology": False,
            "surface_mesh_generated": False,
            "permanent_global_view_cap": None,
        },
        "regression_summary": {
            "focused_i5_10b_tests": receipt["counts"]["focused_tests"],
            "predecessor_tests": receipt["counts"]["predecessor_total_tests"],
            "combined_exercised_tests": receipt["counts"]["combined_exercised_tests"],
            "test_counts_are_not_runtime_quality_scores": True,
        },
        "decision": {
            "i5_10_started": True,
            "i5_10a_complete": True,
            "i5_10b_complete": True,
            "i5_10c_authorized": True,
            "i5_10c_started": False,
            "i5_10d_authorized": False,
            "i5_10d_started": False,
            "i5_10_complete": False,
            "i5_11_started": False,
            "i5_12_started": False,
            "i5_13_started": False,
            "i5_14_started": False,
            "real_mp_sfm_execution_started": False,
            "universal_matching_strategy_winner_declared": False,
            "global_scalar_score_created": False,
            "auto_policy_implemented": False,
            "hybrid_authorized": False,
            "threshold_relaxation_authorized": False,
            "production_default_modified": False,
        },
        "roadmap_guard": {
            "i5_10c_real_evaluation_next": True,
            "i5_10d_qualification_deferred_until_c_freeze": True,
            "i5_11_oom_disk_crash_recovery_deferred": True,
            "i5_12_hybrid_deferred_if_justified": True,
            "i5_13_first_auto_policy_deferred": True,
            "i5_14_final_i5_closure_deferred": True,
        },
        "next": "I5.10C real controlled MP-SfM matching-strategy evaluation.",
    }


__all__ = [
    "I510B_SCHEMA_VERSION", "I510B_PHASE", "I510B_RECEIPT_PHASE",
    "I510B_I510A_FROZEN_SHA256", "I510B_PRIMARY_SCENARIO", "I510B_INPUT_IMAGES",
    "I510B_EXHAUSTIVE_PAIR_COUNT", "I510B_REPOSITORY_ANCHORS",
    "verify_i5_10b_repository_anchors", "validate_i5_10a_for_i5_10b",
    "validate_i5_4_chain_for_i5_10b", "revalidate_i5_10b_source",
    "validate_i5_10b_runtime_probe", "build_i5_10b_candidate_command_template",
    "validate_i5_10b_receipt", "build_i5_10b_mp_sfm_preflight",
]
