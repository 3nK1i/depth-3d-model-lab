from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from core.reconstruction.single_image_provider_matched_view_review import (
    I43G4C_BLIND_LABELS,
    I43G4C_PROVIDER_IMPLEMENTATIONS,
    I43G4C_REVIEW_DIMENSIONS,
    build_manual_review_template,
    sha256_file,
)
from tests.i4_3g4a_helpers import passing_g1_snapshot
from tests.i4_3g4b_helpers import provider_evidence


_WORKER_NAMES = {
    "metric3d_v2": "single_image_depth_result_manifest.json",
    "unidepth_v2": "unidepth_v2_result_manifest.json",
    "moge_2": "moge2_result_manifest.json",
    "depth_pro": "depth_pro_result_manifest.json",
    "depth_anything_v2": "depth_anything_v2_result_manifest.json",
    "marigold": "marigold_result_manifest.json",
}


def write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def artifact(path: Path) -> dict[str, str]:
    return {"path": str(path.resolve()), "sha256": sha256_file(path)}


def completed_review(*, scenario_id: str, commitment: str) -> dict[str, Any]:
    review = build_manual_review_template(
        scenario_id=scenario_id,
        assignment_commitment_sha256=commitment,
        common_keyframe_ids=("kf-0", "kf-1"),
        canonical_available_labels=I43G4C_BLIND_LABELS,
    )
    for index, dimension in enumerate(I43G4C_REVIEW_DIMENSIONS):
        review["dimension_reviews"][dimension] = {
            "preferred_blind_labels": [I43G4C_BLIND_LABELS[index % len(I43G4C_BLIND_LABELS)]],
            "not_assessable_blind_labels": [],
            "notes": "reviewed",
            "dimension_confirmation": True,
        }
    review["reviewer_confirmation"] = True
    return review


def synthetic_g4b_snapshot(
    root: Path,
    *,
    scenario_id: str = "g4_single_image_complete_room",
    rejected: tuple[str, ...] = (),
) -> tuple[Path, str]:
    root.mkdir(parents=True, exist_ok=True)
    sfm_path = write_json(root / "sfm.json", {"backend_id": "pycolmap", "workspace": {}, "landmarks": []})
    provider_refs: dict[str, Any] = {}
    statuses: dict[str, str] = {}
    selected = ["kf-0", "kf-1"]

    for implementation in I43G4C_PROVIDER_IMPLEMENTATIONS:
        status = "rejected" if implementation in rejected else "pass"
        statuses[implementation] = status
        provider_root = root / implementation
        worker_root = provider_root / "worker"
        worker_root.mkdir(parents=True, exist_ok=True)
        frame_records = []
        artifact_paths: list[Path] = []
        for index, keyframe_id in enumerate(selected):
            frame = worker_root / f"frame_{index:05d}_{keyframe_id}.npz"
            np.savez_compressed(
                frame,
                depth=np.full((4, 5), float(index + 1), dtype=np.float32),
                valid_mask=np.ones((4, 5), dtype=bool),
            )
            frame_records.append({
                "order": index,
                "keyframe_id": keyframe_id,
                "evidence_npz_path": str(frame.resolve()),
            })
            if status == "rejected":
                artifact_paths.append(frame)
        worker_manifest = write_json(
            worker_root / _WORKER_NAMES[implementation],
            {"frame_records": frame_records},
        )
        artifact_paths.append(worker_manifest)
        input_manifest = write_json(provider_root / "input_manifest.json", {"sequence": selected})
        artifact_paths.append(input_manifest)
        if status != "rejected":
            canonical = provider_root / f"canonical_{implementation}_fused.npz"
            np.savez_compressed(
                canonical,
                points_reconstruction=np.asarray([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32),
                colors_rgb=np.asarray([[255, 0, 0], [0, 255, 0], [0, 0, 255]], dtype=np.uint8),
            )
            artifact_paths.append(canonical)

        evidence = provider_evidence(implementation, status=status)
        evidence["scenario_id"] = scenario_id
        evidence["adaptive_plan_identity"] = {
            "plan_path": str((provider_root / "plan.json").resolve()),
            "plan_sha256": "1" * 64,
            "selected_view_ids": selected,
        }
        evidence["artifacts"] = [artifact(path) for path in artifact_paths]
        evidence_path = write_json(provider_root / "production_native.json", evidence)
        provider_refs[implementation] = artifact(evidence_path)

    snapshot = {
        "schema_version": 1,
        "phase": "I4.3G4B",
        "status": "production_native_complete_with_provider_rejection" if rejected else "production_native_complete",
        "pass": True,
        "scenario": {
            "scenario_id": scenario_id,
            "authoritative_sfm_snapshot_path": str(sfm_path.resolve()),
            "authoritative_sfm_snapshot_sha256": sha256_file(sfm_path),
            "authoritative_sfm_frame_id": "sfm-frame",
        },
        "providers": provider_refs,
        "production_statuses": statuses,
        "accounting": {
            "terminal_provider_count": 6,
            "all_required_providers_accounted": True,
        },
        "manual_review": {
            "required": True,
            "dimensions": list(I43G4C_REVIEW_DIMENSIONS),
            "deferred_to": "I4.3G4C",
            "production_rejection_cannot_be_overridden": True,
        },
        "authoritative_sfm": {
            "path": str(sfm_path.resolve()),
            "sha256": sha256_file(sfm_path),
        },
        "authority": {
            "camera_pose_authority_owner": "pycolmap_sfm",
            "same_authoritative_sfm_shared_by_all_six_providers": True,
            "authoritative_sfm_modified": False,
            "provider_selected": False,
            "provider_promoted": False,
            "auto_policy_implemented": False,
            "hybrid_authorized": False,
            "metric_scale_claimed": False,
            "measurement_authoritative": False,
            "certified_metrology": False,
            "surface_mesh_generated": False,
            "permanent_global_view_cap": None,
        },
    }
    path = write_json(root / "i4_3g4b_production_native_snapshot.json", snapshot)
    return path, sha256_file(path)


__all__ = [
    "passing_g1_snapshot",
    "completed_review",
    "synthetic_g4b_snapshot",
    "write_json",
    "artifact",
]
