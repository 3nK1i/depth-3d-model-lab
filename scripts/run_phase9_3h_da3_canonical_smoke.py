#!/usr/bin/env python3
"""Dependency-light Phase 9.3H DA3 -> canonical persistent-scene smoke."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.offline_reconstruction_semantics import semantic_presentation_from_persistent_scene
from app.reconstruction_canonical_cameras import load_persistent_canonical_camera_trajectory
from app.reconstruction_canonical_geometry import load_canonical_viewer_mesh
from app.reconstruction_canonical_scene_presentation import build_canonical_scene_presentation
from app.reconstruction_dense_provider_presentation import build_dense_provider_presentation
from core.reconstruction import read_persistent_reconstruction_scene, verify_persistent_scene_file_references
from core.reconstruction.da3 import DA3_PROVIDER_ID
from tests.phase9_3h_helpers import make_phase9_3h_da3_fixture


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="phase9_3h_smoke_") as tmp:
        root = Path(tmp).resolve()
        input_path, result_path, _manifest = make_phase9_3h_da3_fixture(root / "evidence")
        output = root / "job"
        completed = subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts" / "run_phase9_3h_da3_canonical_job.py"),
                "--session-id",
                "44444444-4444-4444-8444-444444444444",
                "--output-dir",
                str(output),
                "--reuse-da3-input-manifest",
                str(input_path),
                "--reuse-da3-result-manifest",
                str(result_path),
                "--pixel-stride",
                "2",
            ],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
        if completed.returncode != 0:
            raise RuntimeError(completed.stdout + completed.stderr)
        result = json.loads((output / "offline_result.json").read_text(encoding="utf-8"))
        scene = read_persistent_reconstruction_scene(result["persistent_scene_manifest"])
        verify_persistent_scene_file_references(scene)
        mesh = load_canonical_viewer_mesh(scene)
        cameras = load_persistent_canonical_camera_trajectory(scene)
        semantics = semantic_presentation_from_persistent_scene(scene)
        canonical = build_canonical_scene_presentation(scene)
        providers = build_dense_provider_presentation(scene, canonical)
        da3 = next((item for item in providers.entries if item.provider_id == DA3_PROVIDER_ID), None)
        if mesh is None or cameras is None or semantics is None or da3 is None:
            raise RuntimeError("Phase 9.3H smoke could not reload the complete bounded persistent presentation.")
        payload = {
            "status": "PASS",
            "scene_id": scene.scene_id,
            "scene_state": scene.state.value,
            "canonical_geometry_authoritative": scene.canonical_geometry_authoritative,
            "certified_metrology": scene.certified_metrology,
            "canonical_vertex_count": mesh.vertex_count,
            "canonical_face_count": mesh.face_count,
            "canonical_camera_count": cameras.persisted_pose_count,
            "scale": canonical.scale_text,
            "metric_geometry_status": result["metric_geometry_status"],
            "unit_to_metre": result["unit_to_metre"],
            "calibration_required": result["calibration_required"],
            "semantic_context_available": result["semantic_context_available"],
            "inspection_qc_available": result["inspection_qc_available"],
            "semantic_geometry_fabricated": result["semantic_geometry_fabricated"],
            "inspection_qc_fabricated": result["inspection_qc_fabricated"],
            "semantic_structure": semantics.structure_text,
            "da3_scene_bound": da3.scene_bound,
            "da3_external": da3.external,
            "da3_authority": da3.authority_text,
            "pipeline": providers.integration_text,
            "da3_runtime_probed": result["da3_runtime_probed"],
            "da3_backend_started": result["da3_backend_started"],
            "heavy_backend_started": False,
            "real_da3_execution_deferred_to_9_3j": True,
        }
        if not payload["canonical_geometry_authoritative"] or payload["certified_metrology"]:
            raise RuntimeError("9.3H smoke changed canonical/metrology authority.")
        if payload["metric_geometry_status"] != "arbitrary" or payload["unit_to_metre"] is not None:
            raise RuntimeError("9.3H smoke invented metric scale.")
        if payload["calibration_required"]:
            raise RuntimeError("9.3H smoke unexpectedly made calibration mandatory.")
        if payload["semantic_context_available"] or payload["inspection_qc_available"]:
            raise RuntimeError("9.3H smoke fabricated semantic/QC availability.")
        if payload["semantic_geometry_fabricated"] or payload["inspection_qc_fabricated"]:
            raise RuntimeError("9.3H smoke fabricated semantic/QC evidence.")
        if not payload["da3_scene_bound"] or "Supporting geometry only" not in payload["da3_authority"]:
            raise RuntimeError("9.3H smoke lost DA3 supporting-evidence provenance.")
        if payload["da3_runtime_probed"] or payload["da3_backend_started"] or payload["heavy_backend_started"]:
            raise RuntimeError("Dependency-light 9.3H smoke unexpectedly started/probed a heavy backend.")
        print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
