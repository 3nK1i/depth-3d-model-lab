#!/usr/bin/env python3
"""I4.1.9A A3.1: persist the explicit manual DA3-vs-MVS visual verdict and close the House positive control."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.reconstruction.errors import ReconstructionValidationError  # noqa: E402
from core.reconstruction.mvs_visual_ab_closure import (  # noqa: E402
    MVS_VISUAL_AB_ALLOWED_VERDICTS,
    build_a3_1_manual_closure,
    require_absent_output,
    sha256_file,
    validate_a3_manual_review_evidence,
    write_json,
)

_DEFAULT_A3 = Path("/tmp/eia_i4_1_9a_house_mvs_positive_control/a3_visual_ab")
_DEFAULT_VERDICT = "mvs_better_but_imperfect"


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--a3-dir", type=Path, default=_DEFAULT_A3)
    parser.add_argument("--verdict", choices=MVS_VISUAL_AB_ALLOWED_VERDICTS, default=_DEFAULT_VERDICT)
    parser.add_argument(
        "--confirm-manual-verdict",
        action="store_true",
        help="Explicitly confirm that the human reviewer accepts the selected A3 visual verdict.",
    )
    return parser.parse_args()


def main() -> int:
    args = _args()
    root = args.a3_dir.expanduser().resolve()
    summary_path = root / "i4_1_9a_a3_summary.json"
    output = root / "i4_1_9a_a3_1_closure.json"
    try:
        require_absent_output(output)
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if not isinstance(summary, dict):
            raise ReconstructionValidationError("A3.1 summary must contain a JSON object.")
        evidence = validate_a3_manual_review_evidence(summary, a3_dir=root)
        closure = build_a3_1_manual_closure(
            summary,
            evidence_validation=evidence,
            verdict=str(args.verdict),
            confirmed=bool(args.confirm_manual_verdict),
            a3_summary_sha256=sha256_file(summary_path),
        )
        write_json(output, closure)
    except (OSError, json.JSONDecodeError, ReconstructionValidationError) as exc:
        print(f"I4.1.9A A3.1 Manual Verdict Closure: FAIL — {exc}", file=sys.stderr)
        return 3

    print("I4.1.9A A3.1 Manual Verdict Closure: PASS — I4.1.9A COMPLETE")
    print(f"verdict={closure['manual_review']['verdict']}")
    print("controlled_comparison_revalidated=True same_authoritative_sfm_frame=True shared_camera=True")
    print("mvs_static_object_orbit_provider_candidate_supported=" + str(closure["decision"]["mvs_static_object_orbit_provider_candidate_supported"]))
    print("dense_provider_promoted=False metric_scale_claimed=False measurement_authoritative=False surface_mesh_generated=False")
    print("i4_2_detail_preservation_authorized=" + str(closure["decision"]["i4_2_detail_preservation_authorized"]))
    print("i4_3_provider_architecture_authorized=False")
    print(f"closure={output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
