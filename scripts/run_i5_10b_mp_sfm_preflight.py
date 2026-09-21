#!/usr/bin/env python3
"""Run execution-free I5.10B MP-SfM runtime/evidence preflight."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.reconstruction.errors import ReconstructionValidationError  # noqa: E402
from core.reconstruction.i5_difficult_handheld_robustness import I51EvidenceBinding  # noqa: E402
from core.reconstruction.i5_mp_sfm_preflight import (  # noqa: E402
    build_i5_10b_mp_sfm_preflight,
    revalidate_i5_10b_source,
    verify_i5_10b_repository_anchors,
)
from core.reconstruction.sfm_environment import probe_pycolmap_environment  # noqa: E402

OUTPUT_NAME = "i5_10b_mp_sfm_runtime_evidence_preflight.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(path: Path, *, label: str) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ReconstructionValidationError(f"I5.10B could not read {label}: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ReconstructionValidationError(f"I5.10B {label} root must be an object: {path}")
    return value


def _binding(path: Path, expected_sha: str, *, label: str, announce_large: bool = False) -> I51EvidenceBinding:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise ReconstructionValidationError(f"I5.10B {label} missing: {resolved}")
    expected = str(expected_sha).strip().lower()
    if len(expected) != 64 or any(ch not in "0123456789abcdef" for ch in expected):
        raise ReconstructionValidationError(f"I5.10B {label} expected SHA256 is invalid.")
    if announce_large:
        print(f"I5.10B verifying immutable checkpoint SHA256: {resolved}", flush=True)
    actual = _sha256(resolved)
    if actual != expected:
        raise ReconstructionValidationError(
            f"I5.10B {label} SHA256 mismatch: expected={expected} actual={actual}"
        )
    if announce_large:
        print(f"I5.10B checkpoint SHA256 verified: {actual}", flush=True)
    return I51EvidenceBinding(str(resolved), actual)


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--i5-10a-admission", type=Path, required=True)
    p.add_argument("--i5-10a-sha256", required=True)
    p.add_argument("--i5-4a-admission", type=Path, required=True)
    p.add_argument("--i5-4a-sha256", required=True)
    p.add_argument("--i5-4b-replay", type=Path, required=True)
    p.add_argument("--i5-4b-sha256", required=True)
    p.add_argument("--i5-4c-closure", type=Path, required=True)
    p.add_argument("--i5-4c-sha256", required=True)
    p.add_argument("--post-i5-10a-checkpoint", type=Path, required=True)
    p.add_argument("--post-i5-10a-checkpoint-sha256", required=True)
    p.add_argument("--preflight-receipt", type=Path, required=True)
    p.add_argument("--preflight-receipt-sha256", required=True)
    p.add_argument("--sfm-python", required=True)
    p.add_argument("--runtime-probe-timeout-seconds", type=float, default=30.0)
    p.add_argument("--output-dir", type=Path, required=True)
    return p


def run(args: argparse.Namespace) -> Path:
    if args.runtime_probe_timeout_seconds <= 0:
        raise ReconstructionValidationError("I5.10B runtime probe timeout must be positive.")
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)

    a10_bind = _binding(args.i5_10a_admission, args.i5_10a_sha256, label="I5.10A admission")
    a4_bind = _binding(args.i5_4a_admission, args.i5_4a_sha256, label="I5.4A admission")
    b4_bind = _binding(args.i5_4b_replay, args.i5_4b_sha256, label="I5.4B replay")
    c4_bind = _binding(args.i5_4c_closure, args.i5_4c_sha256, label="I5.4C closure")
    receipt_bind = _binding(args.preflight_receipt, args.preflight_receipt_sha256, label="I5.10B preflight receipt")
    checkpoint_bind = _binding(
        args.post_i5_10a_checkpoint,
        args.post_i5_10a_checkpoint_sha256,
        label="post-I5.10A source checkpoint",
        announce_large=True,
    )

    a10 = _json(Path(a10_bind.path), label="I5.10A admission")
    a4 = _json(Path(a4_bind.path), label="I5.4A admission")
    b4 = _json(Path(b4_bind.path), label="I5.4B replay")
    c4 = _json(Path(c4_bind.path), label="I5.4C closure")
    receipt = _json(Path(receipt_bind.path), label="I5.10B preflight receipt")

    dataset = Path(str(a4.get("source_identity", {}).get("directory", ""))).expanduser().resolve()
    print(f"I5.10B revalidating full I5.4 source identity: {dataset}", flush=True)
    source = revalidate_i5_10b_source(a4, dataset)
    print(
        f"I5.10B source verified: images={source['full_source_image_count']} "
        f"window=60 source_sha256={source['full_source_asset_set_sha256']} "
        f"window_sha256={source['window_asset_set_sha256']}",
        flush=True,
    )

    runtime = probe_pycolmap_environment(str(Path(args.sfm_python).expanduser().resolve()), timeout_s=float(args.runtime_probe_timeout_seconds)).as_dict()
    anchors = verify_i5_10b_repository_anchors(ROOT)
    result = build_i5_10b_mp_sfm_preflight(
        i5_10a_admission=a10,
        i5_10a_binding=a10_bind,
        i5_4a_admission=a4,
        i5_4a_binding=a4_bind,
        i5_4b_replay=b4,
        i5_4b_binding=b4_bind,
        i5_4c_closure=c4,
        i5_4c_binding=c4_bind,
        source_revalidation=source,
        runtime_probe=runtime,
        post_i5_10a_checkpoint_binding=checkpoint_bind,
        preflight_receipt=receipt,
        preflight_receipt_binding=receipt_bind,
        repository_anchors=anchors,
    )
    target = output / OUTPUT_NAME
    target.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")

    baseline = result["historical_baseline"]
    probe = result["runtime_preflight"]
    plan = result["candidate_plan"]
    print("I5.10B MP-SfM Runtime/Evidence Preflight: PASS")
    print(
        f"historical_baseline=sequential_overlap10 {baseline['registered_images']}/{baseline['input_images']}_registered "
        f"quality={baseline['quality_status']} bottleneck={baseline['primary_bottleneck']}"
    )
    print(
        f"source_identity=VERIFIED full_source_images={source['full_source_image_count']} "
        f"window_images={source['window_image_count']} selected_source_sha256_verified=True"
    )
    print(
        f"pycolmap={probe['pycolmap_version']} distribution={probe.get('distribution','')} "
        f"cuda_devices={probe['cuda_devices']} device_mode={probe['device_mode']} "
        "match_sequential=True match_exhaustive=True"
    )
    print(
        f"candidate=exhaustive input_images={plan['input_images']} exhaustive_pairs={plan['exhaustive_pair_count']} "
        f"attempts={plan['attempt_count']} baseline_rerun={plan['baseline_rerun']}"
    )
    print("candidate_command_template=" + " ".join(plan["command_template"]))
    print(
        "execution_free=True real_mp_sfm_execution=False sfm_rerun=False mvs_rerun=False "
        "threshold_relaxation=False retry_until_pass=False production_default_modified=False"
    )
    print("source_control=NO_GIT_GITEA immutable_post_i5_10a_checkpoint_bound=True")
    print("pycolmap_authority=True metric_scale=False metrology=False auto=False hybrid=False")
    print("I5.10C real controlled evaluation: AUTHORIZED / NOT STARTED")
    print("I5.10D and I5.11-I5.14: DEFERRED")
    print(f"repository_anchor_count={len(anchors)}")
    print(f"evidence={target}")
    return target


def main() -> int:
    try:
        run(_parser().parse_args())
    except (ReconstructionValidationError, OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
        print(f"I5.10B ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
