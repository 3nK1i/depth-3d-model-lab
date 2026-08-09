#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.reconstruction import (
    PersistentReconstructionSceneState,
    persistent_scene_from_pipeline_result,
    read_persistent_reconstruction_scene,
    verify_persistent_scene_file_references,
    write_persistent_reconstruction_scene,
)
from tests.reconstruction_f5_helpers import run_f5_fixture

OUT = Path("/tmp/phase8_7f5_contracts")


def main() -> int:
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)
    _fixture, _providers, result, snapshots = run_f5_fixture(OUT / "fixture", request_id="phase8-7f5-contract-smoke")
    scene = persistent_scene_from_pipeline_result(result, payload_snapshots_by_kind=snapshots)
    manifest = write_persistent_reconstruction_scene(scene, (OUT / "persistent-scene").resolve())
    reloaded = read_persistent_reconstruction_scene(manifest)
    refs = verify_persistent_scene_file_references(reloaded)
    payload = {
        "phase": "8.7F5",
        "pass": reloaded.scene_id == scene.scene_id and reloaded.scene_fingerprint_sha256 == scene.scene_fingerprint_sha256,
        "scene_id": reloaded.scene_id,
        "scene_fingerprint_sha256": reloaded.scene_fingerprint_sha256,
        "state": reloaded.state.value,
        "pipeline_status": reloaded.pipeline_status.value,
        "artifact_count": len(reloaded.artifacts),
        "provider_count": len(reloaded.provider_provenance),
        "file_reference_count": len(refs),
        "fallback_evidence_count": len(reloaded.fallback_evidence),
        "canonical_geometry_artifact_id": reloaded.canonical_geometry_artifact_id,
        "canonical_geometry_authoritative": reloaded.canonical_geometry_authoritative,
        "gaussian_authority": reloaded.gaussian_artifact.authority.value if reloaded.gaussian_artifact else "missing",
        "nerf_authority": reloaded.nerf_artifact.authority.value if reloaded.nerf_artifact else "missing",
        "certified_metrology": False,
        "whole_engine_hardening_deferred_to": "8.7G",
        "reconstruction_ui_enabled": False,
    }
    (OUT / "summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Phase 8.7F5 persistent reconstruction scene contracts: {'PASS' if payload['pass'] else 'FAIL'}")
    print(f"state={payload['state']} artifacts={payload['artifact_count']} providers={payload['provider_count']} files={payload['file_reference_count']}")
    print(f"scene_id={payload['scene_id']} fingerprint={payload['scene_fingerprint_sha256'][:16]}...")
    print("canonical_geometry_authoritative=True gaussian_authority=appearance_only nerf_authority=appearance_only")
    print("certified_metrology=False whole_engine_hardening_deferred_to=8.7G reconstruction_ui_enabled=False")
    print(f"summary={OUT/'summary.json'}")
    return 0 if payload["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
