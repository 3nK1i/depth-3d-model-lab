#!/usr/bin/env python3
"""Finalize the interrupted I5.6B run from already-existing native MVS artifacts only."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.reconstruction.errors import ReconstructionValidationError
from core.reconstruction.i5_difficult_handheld_robustness import I51EvidenceBinding
from core.reconstruction.i5_mvs_robustness import (
    I56B_RECOVERY_PHASE,
    I56B_RECOVERY_SCENARIO_ID,
    I56B_SCENARIO_IDS,
    build_i5_6b_mvs_robustness_replay,
    recover_i5_6b_interrupted_native_scenario,
    validate_i5_6a_for_i5_6b,
    validate_i5_6b_existing_scenario_record,
    validate_i5_6b_receipt,
)
from core.reconstruction.mvs_environment import probe_mvs_host_resources

OUTPUT_NAME = "i5_6b_mvs_robustness_replay.json"
RECOVERY_SUMMARY_NAME = "i5_6b_recovery_provider_summary.json"
RECOVERY_MANIFEST_NAME = "i5_6b_recovered_native_manifest.json"
RECOVERY_FINALIZATION_NAME = "i5_6b_recovery_finalization.json"
SCENARIO_RECORD_NAME = "i5_6b_scenario_record.json"


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--i5-6a-admission", type=Path, required=True)
    p.add_argument("--i5-6a-sha256", required=True)
    p.add_argument("--replay-receipt", type=Path, required=True)
    p.add_argument("--replay-receipt-sha256", required=True)
    p.add_argument("--recovery-receipt", type=Path, required=True)
    p.add_argument("--recovery-receipt-sha256", required=True)
    p.add_argument("--real-dir", type=Path, required=True, help="Existing interrupted I5.6B real directory.")
    return p


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _valid_sha(value: Any) -> bool:
    text = str(value or "").strip().lower()
    return len(text) == 64 and all(c in "0123456789abcdef" for c in text)


def _read_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ReconstructionValidationError(f"I5.6B recovery could not read {label}: {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ReconstructionValidationError(f"I5.6B recovery {label} root must be an object: {path}")
    return payload


def _write_json(path: Path, payload: Mapping[str, Any]) -> Path:
    if path.exists():
        raise ReconstructionValidationError(f"I5.6B recovery refuses to overwrite existing evidence: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(dict(payload), f, indent=2, sort_keys=True, allow_nan=False)
        f.write("\n")
    return path


def _binding(path: Path, expected_sha: str, *, label: str) -> I51EvidenceBinding:
    resolved = path.expanduser().resolve()
    expected = str(expected_sha).strip().lower()
    if not resolved.is_file():
        raise ReconstructionValidationError(f"I5.6B recovery {label} missing: {resolved}")
    if not _valid_sha(expected):
        raise ReconstructionValidationError(f"I5.6B recovery {label} SHA256 malformed.")
    actual = _sha256(resolved)
    if actual != expected:
        raise ReconstructionValidationError(
            f"I5.6B recovery {label} SHA256 mismatch: expected {expected}, found {actual}"
        )
    return I51EvidenceBinding(str(resolved), expected)


def _path_binding(path: Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise ReconstructionValidationError(f"I5.6B recovery artifact missing: {resolved}")
    return {"path": str(resolved), "sha256": _sha256(resolved), "size_bytes": resolved.stat().st_size}


def _verify_binding_dict(binding: Mapping[str, Any], *, label: str) -> None:
    path = Path(str(binding.get("path", ""))).expanduser().resolve()
    expected = str(binding.get("sha256", "")).strip().lower()
    if not path.is_file() or not _valid_sha(expected):
        raise ReconstructionValidationError(f"I5.6B recovery {label} binding is missing or malformed.")
    actual = _sha256(path)
    if actual != expected:
        raise ReconstructionValidationError(f"I5.6B recovery {label} artifact drifted: {path}")
    if binding.get("size_bytes") is not None and int(binding["size_bytes"]) != path.stat().st_size:
        raise ReconstructionValidationError(f"I5.6B recovery {label} size drifted: {path}")


def _verify_existing_record_artifacts(record: Mapping[str, Any], *, scenario_id: str) -> None:
    bindings = record.get("artifact_bindings")
    if not isinstance(bindings, Mapping):
        raise ReconstructionValidationError(f"I5.6B recovery {scenario_id} has no artifact bindings.")
    required = {"frozen_sfm_report", "adaptive_execution_plan", "provider_summary"}
    if record.get("runtime_status") == "pass":
        required.update({"fused_ply", "native_visibility"})
    missing = sorted(required - set(bindings))
    if missing:
        raise ReconstructionValidationError(f"I5.6B recovery {scenario_id} missing artifact bindings: {missing}")
    for name, item in bindings.items():
        if isinstance(item, Mapping) and "path" in item and "sha256" in item:
            _verify_binding_dict(item, label=f"{scenario_id}.{name}")


def _native_manifest(depth_dir: Path, normal_dir: Path, fused: Path, vis: Path) -> dict[str, Any]:
    def entries(directory: Path) -> list[dict[str, Any]]:
        if not directory.is_dir():
            raise ReconstructionValidationError(f"I5.6B recovery native directory missing: {directory}")
        out = []
        for path in sorted(directory.glob("*.geometric.bin")):
            out.append({"name": path.name, "size_bytes": path.stat().st_size, "sha256": _sha256(path)})
        return out
    return {
        "schema_version": 1,
        "phase": I56B_RECOVERY_PHASE,
        "scenario_id": I56B_RECOVERY_SCENARIO_ID,
        "depth_maps": entries(depth_dir),
        "normal_maps": entries(normal_dir),
        "fused_ply": _path_binding(fused),
        "native_visibility": _path_binding(vis),
        "mvs_rerun": False,
    }


def run(args: argparse.Namespace) -> Path:
    real = args.real_dir.expanduser().resolve()
    if not real.is_dir():
        raise ReconstructionValidationError(f"I5.6B recovery real directory missing: {real}")

    final_path = real / OUTPUT_NAME
    if final_path.exists():
        raise ReconstructionValidationError("I5.6B recovery refuses to overwrite an existing top-level replay artifact.")

    admission_binding = _binding(args.i5_6a_admission, args.i5_6a_sha256, label="I5.6A admission")
    receipt_binding = _binding(args.replay_receipt, args.replay_receipt_sha256, label="I5.6B original replay receipt")
    recovery_receipt_binding = _binding(args.recovery_receipt, args.recovery_receipt_sha256, label="I5.6B recovery validation receipt")
    admission = validate_i5_6a_for_i5_6b(_read_json(Path(admission_binding.path), label="I5.6A admission"))
    receipt = validate_i5_6b_receipt(_read_json(Path(receipt_binding.path), label="I5.6B original replay receipt"))
    recovery_receipt = validate_i5_6b_receipt(_read_json(Path(recovery_receipt_binding.path), label="I5.6B recovery validation receipt"))
    if int(recovery_receipt.get("counts", {}).get("combined_exercised_tests", 0)) < int(receipt.get("counts", {}).get("combined_exercised_tests", 0)):
        raise ReconstructionValidationError("I5.6B recovery validation receipt cannot exercise fewer tests than the original replay receipt.")

    preflight_path = real / "runtime_preflight.json"
    preflight = _read_json(preflight_path, label="runtime preflight")
    if preflight.get("mvs_runtime_ready") is not True or preflight.get("provider_available") is not True:
        raise ReconstructionValidationError("I5.6B recovery requires the original qualified MVS runtime preflight.")
    if int(preflight.get("cuda_devices", 0)) < 1:
        raise ReconstructionValidationError("I5.6B recovery original preflight did not expose a CUDA device.")

    records: dict[str, Any] = {}
    for index, sid in enumerate(I56B_SCENARIO_IDS[:5], start=1):
        record_path = real / f"{index:02d}_{sid}" / SCENARIO_RECORD_NAME
        record = validate_i5_6b_existing_scenario_record(_read_json(record_path, label=f"{sid} scenario record"), scenario_id=sid)
        _verify_existing_record_artifacts(record, scenario_id=sid)
        records[sid] = record

    sid = I56B_RECOVERY_SCENARIO_ID
    scenario_dir = real / "06_castle_dynamic_proxy"
    provider_dir = scenario_dir / "providers" / "colmap_mvs"
    worker_dir = provider_dir / "worker"
    dense = provider_dir / "dense"
    adaptive_path = scenario_dir / "adaptive_execution_plan.json"
    request_path = worker_dir / "request.json"
    state_path = worker_dir / "execution_state.json"
    worker_result_path = worker_dir / "result.json"
    normal_summary_path = provider_dir / "i4_3b_summary.json"
    record_path = scenario_dir / SCENARIO_RECORD_NAME
    recovery_summary_path = provider_dir / RECOVERY_SUMMARY_NAME
    manifest_path = scenario_dir / RECOVERY_MANIFEST_NAME

    for path, label in (
        (adaptive_path, "adaptive plan"), (request_path, "worker request"), (state_path, "worker state"),
    ):
        if not path.is_file():
            raise ReconstructionValidationError(f"I5.6B recovery scenario 6 {label} is missing: {path}")
    for path, label in (
        (worker_result_path, "worker result"), (normal_summary_path, "normal provider summary"),
        (record_path, "scenario record"), (recovery_summary_path, "recovery provider summary"), (manifest_path, "recovery manifest"),
    ):
        if path.exists():
            raise ReconstructionValidationError(f"I5.6B recovery expected interrupted-finalization {label} to be absent: {path}")

    fused = dense / "fused.ply"
    vis = Path(f"{fused}.vis")
    depth_dir = dense / "stereo" / "depth_maps"
    normal_dir = dense / "stereo" / "normal_maps"
    adaptive = _read_json(adaptive_path, label="scenario 6 adaptive plan")
    request = _read_json(request_path, label="scenario 6 worker request")
    state = _read_json(state_path, label="scenario 6 worker state")

    # Bind the frozen SfM report carried by I5.6A without rerunning SfM.
    admission_node = admission["scenario_matrix"][sid]
    report_binding = admission_node.get("report_binding")
    if not isinstance(report_binding, Mapping):
        raise ReconstructionValidationError("I5.6B recovery scenario 6 frozen SfM report binding is missing.")
    _binding(Path(str(report_binding.get("path", ""))), str(report_binding.get("sha256", "")), label="scenario 6 frozen SfM report")

    manifest = _native_manifest(depth_dir, normal_dir, fused, vis)
    _write_json(manifest_path, manifest)

    artifact_bindings = {
        "frozen_sfm_report": _path_binding(Path(str(report_binding["path"]))),
        "adaptive_execution_plan": _path_binding(adaptive_path),
        "worker_request": _path_binding(request_path),
        "worker_state": _path_binding(state_path),
        "recovered_native_manifest": _path_binding(manifest_path),
        "fused_ply": _path_binding(fused),
        "native_visibility": _path_binding(vis),
    }
    host_after = probe_mvs_host_resources(scenario_dir).as_dict()
    record, recovery_summary = recover_i5_6b_interrupted_native_scenario(
        admission_node=admission_node,
        adaptive_plan=adaptive,
        worker_request=request,
        worker_state=state,
        fused_ply=fused,
        native_visibility=vis,
        depth_maps_dir=depth_dir,
        normal_maps_dir=normal_dir,
        artifact_bindings=artifact_bindings,
        resource_probe_after=host_after,
    )
    _write_json(recovery_summary_path, recovery_summary)
    record["artifact_bindings"]["recovery_provider_summary"] = _path_binding(recovery_summary_path)
    _write_json(record_path, record)
    records[sid] = record

    recovery_provenance = {
        "schema_version": 1,
        "phase": I56B_RECOVERY_PHASE,
        "recovery_required": True,
        "interrupted_parent_exit_code": 130,
        "recovered_scenario_id": sid,
        "scenarios_1_through_5_reused_without_rerun": True,
        "scenario_6_native_artifacts_reused_without_rerun": True,
        "worker_result_missing": True,
        "normal_provider_summary_missing": True,
        "worker_termination_cause": "unresolved_abrupt_termination_after_native_fusion_artifact_write",
        "native_fusion_structurally_validated": True,
        "authoritative_inputs_revalidated": True,
        "sfm_rerun": False,
        "patch_match_rerun": False,
        "stereo_fusion_rerun": False,
        "mvs_rerun": False,
        "thresholds_modified": False,
        "production_default_modified": False,
        "retry_until_pass": False,
        "artifact_bindings": {
            "runtime_preflight": _path_binding(preflight_path),
            "original_replay_receipt": _path_binding(Path(receipt_binding.path)),
            "recovery_validation_receipt": _path_binding(Path(recovery_receipt_binding.path)),
            "recovered_scenario_record": _path_binding(record_path),
            "recovery_provider_summary": _path_binding(recovery_summary_path),
            "recovered_native_manifest": _path_binding(manifest_path),
        },
    }

    result = build_i5_6b_mvs_robustness_replay(
        i5_6a_admission=admission,
        i5_6a_binding=admission_binding,
        replay_receipt=receipt,
        replay_receipt_binding=receipt_binding,
        runtime_preflight=preflight,
        scenario_records=records,
        recovery_provenance=recovery_provenance,
    )
    evidence_path = _write_json(final_path, result)

    finalization = {
        "schema_version": 1,
        "phase": I56B_RECOVERY_PHASE,
        "status": "interrupted_i5_6b_finalized_without_mvs_rerun",
        "pass": True,
        "quality_status": result["quality_status"],
        "original_replay_receipt": _path_binding(Path(receipt_binding.path)),
        "recovery_validation_receipt": _path_binding(Path(recovery_receipt_binding.path)),
        "recovery_provenance": recovery_provenance,
        "final_evidence": _path_binding(evidence_path),
        "mvs_rerun": False,
        "sfm_rerun": False,
        "i5_6c_authorized": True,
    }
    finalization_path = _write_json(real / RECOVERY_FINALIZATION_NAME, finalization)

    print("I5.6B Interrupted-Parent Native-MVS Recovery Finalization: PASS")
    print("scenarios_1_through_5=reused_without_rerun")
    metrics = record["native_metrics"]
    print(
        f"castle_dynamic_proxy recovered_native_pass=True depth_maps={metrics['geometric_depth_maps']} "
        f"normal_maps={metrics['geometric_normal_maps']} fused_points={metrics['fused_point_count']} "
        f"visibility_points={metrics['visibility_point_count']} support_mean={metrics['visibility_support_mean']}"
    )
    print(
        "worker_result_missing=True provider_adapter_returned=False native_fusion_structurally_validated=True "
        "authoritative_inputs_revalidated=True termination_cause=unresolved"
    )
    print(
        "sfm_rerun=False patch_match_rerun=False stereo_fusion_rerun=False mvs_rerun=False "
        "thresholds_modified=False production_default_modified=False retry_until_pass=False"
    )
    print(f"quality_status={result['quality_status']}")
    print("I5.6B Current Frozen-Foundation MVS Robustness Replay: EVIDENCE_COMPLETE_RECOVERED")
    print("I5.6C MVS robustness qualification and closure: AUTHORIZED")
    print(f"evidence={evidence_path}")
    print(f"recovery_finalization={finalization_path}")
    return evidence_path


def main() -> int:
    try:
        run(_parser().parse_args())
    except (ReconstructionValidationError, OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
        print(f"I5.6B RECOVERY ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
