#!/usr/bin/env python3
"""Dependency-light Phase 9.5D persistent NeRF smoke; no NeRF backend."""
from __future__ import annotations

import json
from pathlib import Path
import tempfile
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.reconstruction import (
    current_process_nerf_heavy_modules,
    persist_nerf_representation,
    reload_persistent_nerf_representation,
)
from tests.phase9_5d_helpers import make_phase9_5d_source


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="phase9_5d_smoke_") as tmp:
        root = Path(tmp)
        scene, artifact, preview, runtime, manifest, summary, console = make_phase9_5d_source(root)
        result = persist_nerf_representation(
            base_scene=scene,
            nerf_artifact=artifact,
            preview_path=preview,
            runtime_evidence_path=runtime,
            source_optimization_manifest_path=manifest,
            source_optimization_summary_path=summary,
            source_worker_console_path=console,
            expected_dataset_fingerprint_sha256=artifact.dataset_fingerprint_sha256,
            output_dir=(root / "persistent").resolve(),
        )
        reloaded = reload_persistent_nerf_representation(result.scene_manifest_path)
        payload = {
            "phase": "9.5D",
            "status": "PASS",
            "persistent_scene_id": result.scene.scene_id,
            "parent_scene_id": result.parent_scene_id,
            "nerf_artifact_available": result.scene.nerf_artifact is not None,
            "checkpoint_persisted": Path(result.artifact.checkpoint_path).is_file(),
            "config_persisted": Path(result.artifact.config_path).is_file(),
            "dataset_persisted": Path(result.dataset_root, "transforms.json").is_file(),
            "restart_reload_verified": reloaded.restart_reload_verified,
            "canonical_geometry_authoritative": result.canonical_geometry_authoritative,
            "nerf_canonical_geometry_authoritative": result.nerf_canonical_geometry_authoritative,
            "metric_scale_authoritative": result.metric_scale_authoritative,
            "measurement_authoritative": result.measurement_authoritative,
            "qc_authoritative": result.qc_authoritative,
            "certified_metrology": result.certified_metrology,
            "backend_started": result.backend_started,
            "rendering_started": result.rendering_started,
            "production_heavy_modules": list(current_process_nerf_heavy_modules()),
        }
        required = (
            payload["nerf_artifact_available"] is True
            and payload["checkpoint_persisted"] is True
            and payload["config_persisted"] is True
            and payload["dataset_persisted"] is True
            and payload["restart_reload_verified"] is True
            and payload["canonical_geometry_authoritative"] is True
            and payload["nerf_canonical_geometry_authoritative"] is False
            and payload["certified_metrology"] is False
            and payload["backend_started"] is False
            and payload["rendering_started"] is False
            and payload["production_heavy_modules"] == []
        )
        if not required:
            payload["status"] = "FAIL"
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0 if payload["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
