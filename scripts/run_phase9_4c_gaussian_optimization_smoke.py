#!/usr/bin/env python3
"""Dependency-light Phase 9.4C optimization execution contract smoke."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.reconstruction import (
    GaussianCanonicalSeedEvidence,
    build_gaussian_seed_from_canonical_arrays,
    current_process_gaussian_heavy_modules,
)


def main() -> int:
    vertices = np.asarray(
        [[0.0, 0.0, 1.0], [0.1, 0.0, 1.0], [0.0, 0.1, 1.0], [0.0, 0.0, 1.1]],
        dtype=np.float32,
    )
    faces = np.asarray([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]], dtype=np.uint32)
    colors = np.ones((4, 4), dtype=np.float32)
    seed = build_gaussian_seed_from_canonical_arrays(vertices, faces, colors, max_gaussians=4)
    with tempfile.TemporaryDirectory(prefix="phase9_4c_") as tmp:
        surface = (Path(tmp) / "canonical.obj").resolve()
        surface.write_text("# deterministic smoke placeholder\n", encoding="utf-8")
        evidence = GaussianCanonicalSeedEvidence(
            source_surface_path=str(surface),
            source_surface_sha256="c" * 64,
            gaussian_count=seed.gaussian_count,
            source_vertex_count=len(vertices),
            source_face_count=len(faces),
        )
        payload = {
            "status": "PASS",
            "phase": "9.4C",
            "gaussian_seed_count": seed.gaussian_count,
            "seed_positions_are_canonical_vertices": bool(
                np.array_equal(seed.positions_reconstruction, vertices)
            ),
            "seed_evidence": evidence.as_dict(),
            "gaussian_scene_role": "appearance_only",
            "canonical_geometry_authoritative": False,
            "metric_scale_authoritative": False,
            "measurement_authoritative": False,
            "qc_authoritative": False,
            "certified_metrology": False,
            "production_heavy_modules": list(current_process_gaussian_heavy_modules()),
            "real_backend_started": False,
            "optimization_started": False,
            "rendering_started": False,
            "real_execution_deferred_to_target_gpu": True,
            "process_group_required": True,
            "termination_order": ["SIGTERM", "SIGKILL"],
        }
    if payload["production_heavy_modules"]:
        payload["status"] = "FAIL"
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
