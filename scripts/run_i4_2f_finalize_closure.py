#!/usr/bin/env python3
"""I4.2F: freeze the TempleRing evidence chain, persist the manual verdict, and close I4.2."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.reconstruction.errors import ReconstructionValidationError  # noqa: E402
from core.reconstruction.temple_ring_closure import (  # noqa: E402
    I42F_ALLOWED_VERDICTS,
    build_i4_2f_closure,
    require_empty_or_absent_output,
    sha256_file,
    validate_i4_2f_evidence,
    write_json,
)

_DEFAULT_E = Path.home() / "Desktop/i4_2_evidence_20260824/i4_2e_detail_qualification_attempt04"
_DEFAULT_PREVIEW = Path.home() / "Desktop/i4_2_evidence_20260824/i4_2e_detail_preview_attempt04"
_DEFAULT_OUTPUT = Path.home() / "Desktop/i4_2_evidence_20260824/i4_2f_closure_attempt01"


def _args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--i4-2e-dir", type=Path, default=_DEFAULT_E)
    p.add_argument("--preview-dir", type=Path, default=_DEFAULT_PREVIEW)
    p.add_argument("--output-dir", type=Path, default=_DEFAULT_OUTPUT)
    p.add_argument("--verdict", choices=I42F_ALLOWED_VERDICTS, required=True)
    p.add_argument("--confirm-manual-verdict", action="store_true")
    return p.parse_args()


def _freeze_summaries(output: Path, evidence: dict) -> dict[str, dict[str, str]]:
    frozen_dir = output / "evidence" / "frozen_summaries"
    frozen_dir.mkdir(parents=True, exist_ok=True)
    upstream = evidence["upstream"]
    sources = {
        "i4_2a_summary": Path(upstream["i4_2a_summary"]),
        "i4_2b_summary": Path(upstream["i4_2b_summary"]),
        "i4_2c_summary": Path(upstream["i4_2c_summary"]),
        "i4_2d_summary": Path(upstream["i4_2d_summary"]),
        "i4_2e_summary": Path(evidence["i4_2e_summary"]),
        "i4_2e_preview_summary": Path(evidence["i4_2e_preview_summary"]),
    }
    result: dict[str, dict[str, str]] = {}
    for label, source in sources.items():
        destination = frozen_dir / source.name
        shutil.copy2(source, destination)
        source_hash = sha256_file(source)
        frozen_hash = sha256_file(destination)
        if frozen_hash != source_hash:
            raise ReconstructionValidationError(f"I4.2F frozen-copy hash mismatch for {label}.")
        result[label] = {
            "source": str(source.resolve()),
            "source_sha256": source_hash,
            "frozen_copy": str(destination.resolve()),
            "frozen_sha256": frozen_hash,
        }
    return result



def _freeze_review_renders(output: Path, evidence: dict) -> dict[str, dict[str, dict[str, str]]]:
    review_dir = output / "evidence" / "review_renders"
    review_dir.mkdir(parents=True, exist_ok=True)
    render_paths = evidence["preview"]["render_paths"]
    render_hashes = evidence["preview"]["render_sha256"]
    result: dict[str, dict[str, dict[str, str]]] = {}
    for view in ("oblique", "front", "side", "top"):
        result[view] = {}
        for kind in ("ab", "loss_overlay"):
            source = Path(render_paths[view][kind])
            destination = review_dir / f"i4_2e_{view}_{kind}.png"
            shutil.copy2(source, destination)
            expected = render_hashes[view][kind]
            frozen_hash = sha256_file(destination)
            if frozen_hash != expected:
                raise ReconstructionValidationError(f"I4.2F frozen review-render hash mismatch for {view}/{kind}.")
            result[view][kind] = {
                "source": str(source.resolve()),
                "source_sha256": expected,
                "frozen_copy": str(destination.resolve()),
                "frozen_sha256": frozen_hash,
            }
    return result


def main() -> int:
    args = _args()
    output = args.output_dir.expanduser().resolve()
    try:
        require_empty_or_absent_output(output)
        evidence = validate_i4_2f_evidence(i4_2e_dir=args.i4_2e_dir, preview_dir=args.preview_dir)
        closure = build_i4_2f_closure(
            evidence_validation=evidence,
            verdict=str(args.verdict),
            confirmed=bool(args.confirm_manual_verdict),
        )
        output.mkdir(parents=True, exist_ok=True)
        frozen = _freeze_summaries(output, evidence)
        frozen_review = _freeze_review_renders(output, evidence)
        closure["frozen_summaries"] = frozen
        closure["frozen_review_renders"] = frozen_review
        manifest = {
            "schema_version": 1,
            "phase": "I4.2F-evidence-freeze",
            "summaries": frozen,
            "review_renders": frozen_review,
            "source_artifact_sha256": evidence["source_artifact_sha256"],
            "review_render_sha256": evidence["preview"]["render_sha256"],
        }
        manifest_path = write_json(output / "i4_2f_evidence_manifest.json", manifest)
        closure["evidence_manifest"] = {
            "path": str(manifest_path),
            "sha256": sha256_file(manifest_path),
        }
        closure_path = write_json(output / "i4_2f_closure.json", closure)
    except (OSError, json.JSONDecodeError, ReconstructionValidationError) as exc:
        print(f"I4.2F TempleRing final closure: FAIL — {exc}", file=sys.stderr)
        return 3

    if closure["pass"]:
        print("I4.2F TempleRing final closure: PASS — I4.2 COMPLETE")
    else:
        print("I4.2F TempleRing final closure: BLOCKED — I4.2 REMAINS OPEN")
    print(f"verdict={closure['manual_review']['verdict']}")
    print(
        "d12_safe_on_temple_ring_within_tested_transfer="
        + str(closure["scientific_conclusion"]["d12_sparse_safe_on_temple_ring_within_tested_transfer"])
    )
    print("production_filter_policy_changed=False dense_provider_promoted=False")
    print("metric_scale_claimed=False measurement_authoritative=False certified_metrology=False surface_mesh_generated=False")
    print("i4_3_provider_architecture_authorized=" + str(closure["decision"]["i4_3_provider_architecture_authorized"]))
    print(f"closure={closure_path}")
    return 0 if closure["pass"] else 4


if __name__ == "__main__":
    raise SystemExit(main())
