#!/usr/bin/env python3
"""Persist/reload the already-qualified F4 real-provider result without rerunning backends."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.reconstruction import (
    persistent_scene_from_pipeline_payload,
    read_persistent_reconstruction_scene,
    verify_persistent_scene_file_references,
    write_persistent_reconstruction_scene,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--f4-result", default="/tmp/phase8_7f4_real_backends/pipeline_result.json")
    parser.add_argument("--f4-summary", default="/tmp/phase8_7f4_real_backends/summary.json")
    parser.add_argument("--output-dir", default="/tmp/phase8_7f5_real_persistence")
    return parser


def _overview(scene, path: Path) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.set_xlim(0, 12); ax.set_ylim(0, 5); ax.axis("off")
    rows = [
        ("Canonical Geometry", scene.canonical_geometry_artifact_id, "authoritative"),
        ("Semantic + QC", "semantic_geometry / inspection_qc", "domain authority"),
        ("Gaussian", scene.gaussian_artifact.artifact_id if scene.gaussian_artifact else "missing", "appearance_only"),
        ("NeRF", scene.nerf_artifact.artifact_id if scene.nerf_artifact else "missing", "appearance_only"),
    ]
    for index, (label, artifact, authority) in enumerate(rows):
        y = 4.1 - index * 1.0
        patch = FancyBboxPatch((0.4, y-0.35), 11.2, 0.7, boxstyle="round,pad=0.03")
        ax.add_patch(patch)
        ax.text(0.7, y, label, va="center", fontsize=12, fontweight="bold")
        ax.text(3.0, y, artifact, va="center", fontsize=9)
        ax.text(10.0, y, authority, va="center", fontsize=10)
    ax.text(0.5, 0.15, f"scene={scene.scene_id}  state={scene.state.value}  fingerprint={scene.scene_fingerprint_sha256[:20]}...", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main() -> int:
    args = _parser().parse_args()
    result_path = Path(args.f4_result).expanduser().resolve()
    summary_path = Path(args.f4_summary).expanduser().resolve()
    out = Path(args.output_dir).expanduser().resolve()
    if not result_path.is_file() or not summary_path.is_file():
        raise SystemExit("ERROR: qualified F4 pipeline_result.json and summary.json are required; do not rerun F4 automatically.")
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    result = json.loads(result_path.read_text(encoding="utf-8"))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    snapshots = {
        "gaussian_scene": summary.get("gaussian_scene", {}),
        "nerf_scene": summary.get("nerf_scene", {}),
    }
    scene = persistent_scene_from_pipeline_payload(result, payload_snapshots_by_kind=snapshots)
    manifest = write_persistent_reconstruction_scene(scene, out / "scene")
    reloaded = read_persistent_reconstruction_scene(manifest)
    refs = verify_persistent_scene_file_references(reloaded)
    overview = out / "persistent_scene_overview.png"
    _overview(reloaded, overview)
    payload = {
        "phase": "8.7F5",
        "pass": True,
        "source": "qualified_phase8_7f4_real_provider_result",
        "scene_id": reloaded.scene_id,
        "scene_fingerprint_sha256": reloaded.scene_fingerprint_sha256,
        "state": reloaded.state.value,
        "pipeline_status": reloaded.pipeline_status.value,
        "artifact_count": len(reloaded.artifacts),
        "provider_count": len(reloaded.provider_provenance),
        "fallback_evidence_count": len(reloaded.fallback_evidence),
        "file_reference_count": len(refs),
        "canonical_geometry_artifact_id": reloaded.canonical_geometry_artifact_id,
        "canonical_geometry_authoritative": True,
        "gaussian_authority": reloaded.gaussian_artifact.authority.value if reloaded.gaussian_artifact else "missing",
        "nerf_authority": reloaded.nerf_artifact.authority.value if reloaded.nerf_artifact else "missing",
        "certified_metrology": False,
        "manifest": str(manifest),
        "overview": str(overview),
        "whole_engine_hardening_deferred_to": "8.7G",
        "reconstruction_ui_enabled": False,
    }
    (out / "summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("Phase 8.7F5 qualified F4 → persistent scene validation: PASS")
    print(f"state={payload['state']} artifacts={payload['artifact_count']} providers={payload['provider_count']} file_refs={payload['file_reference_count']}")
    print(f"scene_id={payload['scene_id']} fingerprint={payload['scene_fingerprint_sha256'][:16]}...")
    print(f"gaussian_authority={payload['gaussian_authority']} nerf_authority={payload['nerf_authority']}")
    print("canonical_geometry_authoritative=True certified_metrology=False")
    print("whole_engine_hardening_deferred_to=8.7G reconstruction_ui_enabled=False")
    print(f"manifest={manifest}")
    print(f"overview={overview}")
    print(f"summary={out/'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
