#!/usr/bin/env python3
"""I4.3G4A execution-free single-image source/runtime/evidence preflight."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.reconstruction.depth_anything_v2_dense_provider import probe_depth_anything_v2_environment  # noqa: E402
from core.reconstruction.depth_pro_dense_provider import probe_depth_pro_environment  # noqa: E402
from core.reconstruction.errors import ReconstructionValidationError  # noqa: E402
from core.reconstruction.marigold_dense_provider import probe_marigold_environment  # noqa: E402
from core.reconstruction.metric3d_v2_dense_provider import probe_metric3d_v2_environment  # noqa: E402
from core.reconstruction.moge2_dense_provider import probe_moge_2_environment  # noqa: E402
from core.reconstruction.mvs_da3_qualification import build_source_identity  # noqa: E402
from core.reconstruction.single_image_provider_qualification import (  # noqa: E402
    I43G3C_FROZEN_CLOSURE_SHA256,
    I43G4A_G2_SCENARIO_MAP,
    I43G4A_PROVIDER_IMPLEMENTATIONS,
    I43G4A_SCENARIOS,
    build_g4a_scenario_binding,
    build_i4_3g4a_preflight_snapshot,
)
from core.reconstruction.unidepth_v2_dense_provider import probe_unidepth_v2_environment  # noqa: E402

SNAPSHOT_NAME = "i4_3g4a_single_image_preflight_snapshot.json"
FROZEN_G1_SHA256 = "d5a169366aaa328604833686d1d65cf648c2ed88fed7ceceb5834568e94e456f"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--i4-3g1-snapshot", type=Path, required=True)
    parser.add_argument("--i4-3g3c-closure", type=Path, required=True)
    parser.add_argument("--expected-g1-sha256", default=FROZEN_G1_SHA256)
    parser.add_argument("--expected-g3c-sha256", default=I43G3C_FROZEN_CLOSURE_SHA256)
    parser.add_argument("--complete-room-g2b-snapshot", type=Path, required=True)
    parser.add_argument("--low-texture-g2b-snapshot", type=Path, required=True)
    parser.add_argument("--weak-overlap-g2b-snapshot", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, default=ROOT)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--metric3d-checkpoint", type=Path)
    parser.add_argument("--unidepth-checkpoint-dir", type=Path)
    parser.add_argument("--unidepth-variant", choices=("auto", "small", "base", "large"), default="auto")
    parser.add_argument("--moge-variant", choices=("auto", "small", "base", "large"), default="auto")
    parser.add_argument("--depth-pro-checkpoint", type=Path)
    parser.add_argument("--dav2-source-dir", type=Path)
    parser.add_argument("--dav2-checkpoint", type=Path)
    parser.add_argument("--dav2-variant", choices=("auto", "small", "base", "large"), default="auto")
    parser.add_argument("--marigold-checkpoint-dir", type=Path)
    parser.add_argument("--marigold-profile", choices=("auto", "efficient", "balanced", "ultra"), default="auto")
    for implementation in I43G4A_PROVIDER_IMPLEMENTATIONS:
        parser.add_argument(
            f"--{implementation.replace('_', '-')}-runtime-json",
            type=Path,
            help=f"Optional captured {implementation} runtime probe JSON; avoids probing the live runtime.",
        )
    parser.add_argument("--storage-path", type=Path, default=Path("/home/eiatech/Desktop"))
    parser.add_argument("--minimum-free-gib", type=float, default=32.0)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _valid_sha256(value: str) -> bool:
    return len(value) == 64 and all(ch in "0123456789abcdef" for ch in value.lower())


def _read_json(path: Path, *, label: str) -> dict[str, Any]:
    source = path.expanduser().resolve()
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ReconstructionValidationError(f"Unable to read {label}: {source}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ReconstructionValidationError(f"{label} must contain a JSON object: {source}")
    return payload


def _verify_expected_sha(path: Path, expected: str, *, label: str) -> str:
    expected = str(expected).strip().lower()
    if not _valid_sha256(expected):
        raise ReconstructionValidationError(f"{label} expected SHA256 is invalid: {expected!r}")
    actual = _sha256(path)
    if actual != expected:
        raise ReconstructionValidationError(f"{label} SHA256 mismatch: expected={expected} actual={actual}")
    return actual


def _verify_artifact(artifact: Mapping[str, Any], *, label: str) -> Path:
    path_raw = str(artifact.get("path") or "").strip()
    expected = str(artifact.get("sha256") or "").strip().lower()
    if not path_raw or not _valid_sha256(expected):
        raise ReconstructionValidationError(f"{label} artifact identity is incomplete.")
    path = Path(path_raw).expanduser().resolve()
    if not path.is_file():
        raise ReconstructionValidationError(f"{label} artifact is missing: {path}")
    actual = _sha256(path)
    if actual != expected:
        raise ReconstructionValidationError(f"{label} artifact SHA256 mismatch: expected={expected} actual={actual}")
    return path


def _verify_g2b_context(path: Path, *, g4_scenario_id: str) -> tuple[dict[str, Any], str]:
    resolved = path.expanduser().resolve()
    payload = _read_json(resolved, label=f"{g4_scenario_id} G2B production snapshot")
    snapshot_sha = _sha256(resolved)
    expected_g2 = I43G4A_G2_SCENARIO_MAP[g4_scenario_id]
    scenario = payload.get("scenario")
    if not isinstance(scenario, Mapping) or scenario.get("scenario_id") != expected_g2:
        raise ReconstructionValidationError(f"{g4_scenario_id} snapshot does not contain {expected_g2}.")

    source_artifact = payload.get("source_identity")
    sfm = payload.get("authoritative_sfm")
    if not isinstance(source_artifact, Mapping) or not isinstance(sfm, Mapping) or not isinstance(sfm.get("snapshot"), Mapping):
        raise ReconstructionValidationError(f"{g4_scenario_id} G2B snapshot is missing frozen source/SfM artifacts.")
    source_manifest_path = _verify_artifact(source_artifact, label=f"{g4_scenario_id} source manifest")
    _verify_artifact(sfm["snapshot"], label=f"{g4_scenario_id} authoritative SfM snapshot")

    source_manifest = _read_json(source_manifest_path, label=f"{g4_scenario_id} source manifest")
    source_directory = Path(str(scenario.get("source_directory") or "")).expanduser().resolve()
    current_identity = build_source_identity(source_directory, minimum_images=3)
    if current_identity.image_count != int(scenario.get("source_image_count") or 0):
        raise ReconstructionValidationError(f"{g4_scenario_id} source image count drifted since frozen G2 evidence.")
    if current_identity.asset_set_sha256 != str(scenario.get("source_asset_set_sha256") or ""):
        raise ReconstructionValidationError(f"{g4_scenario_id} source image bytes drifted since frozen G2 evidence.")
    if source_manifest.get("asset_set_sha256") != current_identity.asset_set_sha256:
        raise ReconstructionValidationError(f"{g4_scenario_id} frozen source manifest disagrees with current source identity.")
    return payload, snapshot_sha


def _runtime_json_arg(args: argparse.Namespace, implementation: str) -> Path | None:
    return getattr(args, f"{implementation}_runtime_json")


def _runtime_probes(args: argparse.Namespace) -> dict[str, dict[str, Any]]:
    overrides: dict[str, dict[str, Any]] = {}
    for implementation in I43G4A_PROVIDER_IMPLEMENTATIONS:
        path = _runtime_json_arg(args, implementation)
        if path is not None:
            overrides[implementation] = _read_json(path, label=f"{implementation} runtime probe")

    project = args.project_root.expanduser().resolve()
    python = args.python.expanduser().resolve()
    result: dict[str, dict[str, Any]] = {}
    result["metric3d_v2"] = overrides.get("metric3d_v2") or probe_metric3d_v2_environment(
        project_root=project,
        python_executable=python,
        checkpoint_path=args.metric3d_checkpoint.expanduser().resolve() if args.metric3d_checkpoint else None,
    ).as_dict()
    result["unidepth_v2"] = overrides.get("unidepth_v2") or probe_unidepth_v2_environment(
        project_root=project,
        python_executable=python,
        checkpoint_path=args.unidepth_checkpoint_dir.expanduser().resolve() if args.unidepth_checkpoint_dir else None,
        variant=args.unidepth_variant,
    ).as_dict()
    result["moge_2"] = overrides.get("moge_2") or probe_moge_2_environment(
        project_root=project,
        python_executable=python,
        variant=args.moge_variant,
    ).as_dict()
    result["depth_pro"] = overrides.get("depth_pro") or probe_depth_pro_environment(
        project_root=project,
        python_executable=python,
        checkpoint_path=args.depth_pro_checkpoint.expanduser().resolve() if args.depth_pro_checkpoint else None,
    ).as_dict()
    result["depth_anything_v2"] = overrides.get("depth_anything_v2") or probe_depth_anything_v2_environment(
        project_root=project,
        python_executable=python,
        variant=args.dav2_variant,
        source_directory=args.dav2_source_dir.expanduser().resolve() if args.dav2_source_dir else None,
        checkpoint_path=args.dav2_checkpoint.expanduser().resolve() if args.dav2_checkpoint else None,
    ).as_dict()
    result["marigold"] = overrides.get("marigold") or probe_marigold_environment(
        project_root=project,
        python_executable=python,
        profile=args.marigold_profile,
        checkpoint_dir=args.marigold_checkpoint_dir.expanduser().resolve() if args.marigold_checkpoint_dir else None,
    ).as_dict()
    return result


def _write_json(path: Path, payload: Mapping[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    return path


def main() -> int:
    args = _parser().parse_args()
    try:
        out = args.output_dir.expanduser().resolve()
        if out.exists() and any(out.iterdir()):
            raise ReconstructionValidationError(f"I4.3G4A output directory must be empty or absent: {out}")
        out.mkdir(parents=True, exist_ok=True)

        g1_path = args.i4_3g1_snapshot.expanduser().resolve()
        g3c_path = args.i4_3g3c_closure.expanduser().resolve()
        g1_sha = _verify_expected_sha(g1_path, args.expected_g1_sha256, label="I4.3G1 snapshot")
        g3c_sha = _verify_expected_sha(g3c_path, args.expected_g3c_sha256, label="I4.3G3C closure")
        g1 = _read_json(g1_path, label="I4.3G1 snapshot")
        g3c = _read_json(g3c_path, label="I4.3G3C closure")

        g2_paths = {
            "g4_single_image_complete_room": args.complete_room_g2b_snapshot,
            "g4_single_image_low_texture": args.low_texture_g2b_snapshot,
            "g4_single_image_weak_overlap": args.weak_overlap_g2b_snapshot,
        }
        bindings = []
        g2_predecessors: dict[str, Any] = {}
        for scenario_id in I43G4A_SCENARIOS:
            path = g2_paths[scenario_id].expanduser().resolve()
            payload, snapshot_sha = _verify_g2b_context(path, g4_scenario_id=scenario_id)
            bindings.append(build_g4a_scenario_binding(
                scenario_id=scenario_id,
                g2b_snapshot=payload,
                g2b_snapshot_path=str(path),
                g2b_snapshot_sha256=snapshot_sha,
            ))
            g2_predecessors[scenario_id] = {"path": str(path), "sha256": snapshot_sha}

        storage_path = args.storage_path.expanduser().resolve()
        if not storage_path.exists():
            raise ReconstructionValidationError(f"I4.3G4A storage path does not exist: {storage_path}")
        usage = shutil.disk_usage(storage_path)
        minimum_free_bytes = int(float(args.minimum_free_gib) * 1024**3)
        if minimum_free_bytes <= 0:
            raise ReconstructionValidationError("--minimum-free-gib must be greater than zero.")

        snapshot = build_i4_3g4a_preflight_snapshot(
            i4_3g1_snapshot=g1,
            i4_3g3c_closure=g3c,
            scenario_bindings=bindings,
            runtime_probes=_runtime_probes(args),
            free_disk_bytes=int(usage.free),
            minimum_free_disk_bytes=minimum_free_bytes,
        )
        snapshot["predecessor_evidence"] = {
            "i4_3g1_snapshot": {"path": str(g1_path), "sha256": g1_sha},
            "i4_3g3c_final_closure": {"path": str(g3c_path), "sha256": g3c_sha},
            "g2b_reused_contexts": g2_predecessors,
            "g1_predecessor_evidence": g1.get("predecessor_evidence", {}),
        }
        snapshot["storage"]["storage_path"] = str(storage_path)
        target = _write_json(out / SNAPSHOT_NAME, snapshot)

        summary = snapshot["binding_summary"]
        runtime = snapshot["runtime_summary"]
        storage = snapshot["storage"]
        decision = snapshot["decision"]
        authority = snapshot["authority"]
        print(f"I4.3G4A single-image source/runtime/evidence preflight: {'PASS' if snapshot['pass'] else 'NOT READY'}")
        print(f"bound_scenarios={summary['bound_scenario_count']}/3 all_g4_sources_bound={summary['all_g4_sources_bound']}")
        print(f"ready_providers={runtime['ready_provider_count']}/6 all_single_image_runtimes_ready={runtime['all_single_image_runtimes_ready']}")
        print(
            f"storage_free_gib={storage['free_disk_gib']:.2f} minimum_free_gib={storage['minimum_free_disk_gib']:.2f} "
            f"storage_ready={storage['storage_ready_for_g4b']}"
        )
        print("status_contract=pass,degraded,rejected quality; failed,cancelled,unavailable infrastructure")
        print("g1_snapshot_unchanged=True g4_manual_review_required_for_all_three=True")
        print(
            "g4b_authorized={authorized} sfm_started=False provider_inference_started=False matched_view_started=False".format(
                authorized=decision["i4_3g4b_production_native_execution_authorized"]
            )
        )
        print(
            "camera_pose_authority_owner={owner} provider_selected={selected} provider_promoted={promoted} "
            "auto_policy_implemented={auto} hybrid_authorized={hybrid} metric_scale_claimed={metric} "
            "measurement_authoritative={measurement} certified_metrology={metrology} surface_mesh_generated={mesh}".format(
                owner=authority["camera_pose_authority_owner"],
                selected=authority["provider_selected"],
                promoted=authority["provider_promoted"],
                auto=authority["auto_policy_implemented"],
                hybrid=authority["hybrid_authorized"],
                metric=authority["metric_scale_claimed"],
                measurement=authority["measurement_authoritative"],
                metrology=authority["certified_metrology"],
                mesh=authority["surface_mesh_generated"],
            )
        )
        print(f"snapshot={target}")
        return 0 if snapshot["pass"] else 3
    except Exception as exc:  # noqa: BLE001 - CLI boundary produces deterministic failure evidence
        print(f"I4.3G4A ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
