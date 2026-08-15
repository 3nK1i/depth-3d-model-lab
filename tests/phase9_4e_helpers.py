from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from core.contracts.validation import metadata_as_dict
from core.reconstruction import persist_gaussian_representation, persistent_scene_from_pipeline_payload
from tests.phase9_4d_helpers import make_phase9_4d_source


def _with_materialized_canonical_surface(scene, surface_path: Path):
    """Keep the 9.4E UI fixture faithful to a real viewer-ready persistent scene."""
    surface_path = surface_path.resolve()
    surface_path.parent.mkdir(parents=True, exist_ok=True)
    surface_path.write_text(
        "v -1 -1 0 0.55 0.45 0.35\n"
        "v 1 -1 0 0.65 0.50 0.40\n"
        "v 1 1 0 0.70 0.60 0.50\n"
        "v -1 1 0 0.50 0.55 0.45\n"
        "f 1 2 3\n"
        "f 1 3 4\n",
        encoding="utf-8",
    )

    snapshots = {}
    stages = []
    for artifact in scene.artifacts:
        snapshot = metadata_as_dict(artifact.payload_snapshot)  # type: ignore[arg-type]
        if artifact.kind.value == "surface_mesh":
            snapshot.update(
                {
                    "surface_obj_path": str(surface_path),
                    "surface_vertex_count": 4,
                    "surface_face_count": 2,
                    "canonical_geometry_authoritative": True,
                    "certified_metrology": False,
                }
            )
        snapshots[artifact.kind.value] = snapshot
        stages.append(
            {
                "stage": artifact.stage.value,
                "state": "completed",
                "required": True,
                "provider_attempts": [artifact.provider_id],
                "provider_id": artifact.provider_id,
                "fallback_used": False,
                "reason": "",
                "artifact": artifact.as_dict(),
            }
        )

    return persistent_scene_from_pipeline_payload(
        {
            "request_id": scene.request_id,
            "session_id": scene.session_id,
            "status": scene.pipeline_status.value,
            "canonical_geometry_artifact_id": scene.canonical_geometry_artifact_id,
            "canonical_geometry_authoritative": True,
            "certified_metrology": False,
            "stages": stages,
            "provider_provenance": [item.as_dict() for item in scene.provider_provenance],
        },
        payload_snapshots_by_kind=snapshots,
        created_at_utc=scene.created_at_utc,
    )


def make_phase9_4e_persistent_scene(tmp_path: Path, *, gaussian_count: int = 64):
    scene, artifact, preview, metrics, manifest, summary = make_phase9_4d_source(tmp_path / "source")
    scene = _with_materialized_canonical_surface(scene, tmp_path / "canonical_surface.obj")
    count = int(gaussian_count)
    if count != artifact.gaussian_count:
        raise ValueError("Phase 9.4E fixture currently preserves the 9.4D artifact count of 64.")
    rng = np.random.default_rng(9405)
    means = np.stack(
        [
            np.linspace(-0.8, 0.8, count, dtype=np.float32),
            0.18 * np.sin(np.linspace(0.0, np.pi * 4.0, count, dtype=np.float32)),
            np.linspace(-0.25, 0.25, count, dtype=np.float32),
        ],
        axis=1,
    )
    scales = np.full((count, 3), 0.035, dtype=np.float32)
    scales[:, 0] *= np.linspace(0.7, 1.4, count, dtype=np.float32)
    colors = rng.uniform(0.1, 0.95, size=(count, 3)).astype(np.float32)
    opacities = np.linspace(0.25, 0.95, count, dtype=np.float32)
    quats = np.zeros((count, 4), dtype=np.float32)
    quats[:, 0] = 1.0
    torch.save(
        {
            "schema_version": 1,
            "backend": "gsplat_external",
            "backend_version": "1.5.3",
            "dataset_fingerprint_sha256": artifact.dataset_fingerprint_sha256,
            "means_reconstruction": torch.from_numpy(means),
            "quats_wxyz": torch.from_numpy(quats),
            "scales_reconstruction": torch.from_numpy(scales),
            "opacities": torch.from_numpy(opacities),
            "colors_rgb": torch.from_numpy(colors),
            "pose_convention": artifact.pose_convention,
            "canonical_geometry_authoritative": False,
            "certified_metrology": False,
        },
        artifact.scene_path,
    )
    result = persist_gaussian_representation(
        base_scene=scene,
        gaussian_artifact=artifact,
        preview_path=preview,
        qualification_metrics_path=metrics,
        source_artifact_manifest_path=manifest,
        source_optimization_summary_path=summary,
        expected_dataset_fingerprint_sha256=artifact.dataset_fingerprint_sha256,
        output_dir=(tmp_path / "persistent").resolve(),
    )
    return result.scene, Path(result.scene_manifest_path), result.artifact
