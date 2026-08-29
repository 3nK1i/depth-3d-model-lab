from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from core.reconstruction.temporal_provider_matched_view_review import sha256_file
from tests.i4_3g3a_helpers import passing_g1_snapshot
from tests.i4_3g3b_helpers import provider_evidence


def write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def artifact(path: Path) -> dict[str, Any]:
    return {"path": str(path.resolve()), "sha256": sha256_file(path), "size_bytes": path.stat().st_size}


def g3c_fixture(
    tmp_path: Path,
    *,
    scenario_id: str = "g3_temporal_static_sweep",
    rolling_status: str = "pass",
    vda_status: str = "pass",
) -> dict[str, Any]:
    root = tmp_path / scenario_id
    workspace = root / "workspace"
    images_dir = workspace / "images"
    images_dir.mkdir(parents=True)

    keyframes = ("kf000", "kf001", "kf002")
    indices = (10, 11, 12)
    timestamps = (1_000_000_000, 1_100_000_000, 1_200_000_000)
    workspace_images: list[dict[str, Any]] = []
    for order, (keyframe, index, timestamp) in enumerate(zip(keyframes, indices, timestamps, strict=True)):
        image = np.full((12, 16, 3), 20 + order * 30, dtype=np.uint8)
        rel = Path("images") / f"{keyframe}.png"
        path = workspace / rel
        assert cv2.imwrite(str(path), image)
        workspace_images.append(
            {
                "order": order,
                "keyframe_id": keyframe,
                "frame_id": f"frame-{order}",
                "source_sequence_index": index,
                "timestamp_ns": timestamp,
                "staged_image_path": rel.as_posix(),
                "staged_sha256": sha256_file(path),
            }
        )

    sfm = {
        "workspace": {
            "workspace_root": str(workspace.resolve()),
            "images": workspace_images,
        }
    }
    sfm_path = write_json(root / "sfm_result.json", sfm)
    sfm_sha = sha256_file(sfm_path)
    source_sha = "a" * 64
    frame_id = f"g3c:{scenario_id}:pycolmap"

    rd_summary = {
        "depth_canonicalization": {
            "transform": "provider_native_polarity_sign_flip",
            "multiplier": -1.0,
            "sequence_order_changed": False,
            "physical_metric_scale_claimed": False,
        }
    }
    rd_summary_path = write_json(root / "i4_3d2_rolling_depth_execution_summary.json", rd_summary)
    vda_summary_path = write_json(root / "i4_3d1_vda_execution_summary.json", {"status": vda_status})

    rd_npz = root / "rolling_depth_evidence.npz"
    vda_npz = root / "vda_depth_evidence.npz"
    np.savez_compressed(rd_npz, depth=np.linspace(-1.0, 1.0, 3 * 12 * 16, dtype=np.float32).reshape(3, 12, 16))
    np.savez_compressed(vda_npz, depth=np.linspace(0.2, 5.0, 3 * 12 * 16, dtype=np.float32).reshape(3, 12, 16))

    evidence_paths: dict[str, Path] = {}
    for implementation, status, summary_path, depth_path in (
        ("rolling_depth", rolling_status, rd_summary_path, rd_npz),
        ("video_depth_anything", vda_status, vda_summary_path, vda_npz),
    ):
        evidence = provider_evidence(implementation, scenario_id=scenario_id)
        evidence["status"] = status
        evidence["source_identity"] = {"source_file_sha256": source_sha}
        evidence["authoritative_sfm_identity"] = {
            "snapshot_path": str(sfm_path.resolve()),
            "snapshot_sha256": sfm_sha,
            "frame_id": frame_id,
        }
        evidence["temporal_sequence"].update(
            {
                "ordered_keyframe_ids": list(keyframes),
                "ordered_source_sequence_indices": list(indices),
                "ordered_timestamps_ns": list(timestamps),
            }
        )
        evidence["artifacts"] = [artifact(summary_path), artifact(depth_path)]
        if status == "rejected":
            evidence["metrics"]["execution_completed"] = False
            evidence["rejection"] = {
                "terminal": True,
                "reason": "global_sequence_alignment_rejected:sequence_affine_residual_exceeds_limit",
                "canonical_geometry_generated": False,
            }
        evidence_path = write_json(root / f"{implementation}_production_native.json", evidence)
        evidence_paths[implementation] = evidence_path

    has_rejection = rolling_status == "rejected" or vda_status == "rejected"
    production = {
        "phase": "I4.3G3B",
        "status": "production_native_complete_with_provider_rejection" if has_rejection else "production_native_complete",
        "pass": True,
        "scenario": {"scenario_id": scenario_id, "source_file_sha256": source_sha},
        "providers": {
            "rolling_depth": {"evidence": artifact(evidence_paths["rolling_depth"])},
            "video_depth_anything": {"evidence": artifact(evidence_paths["video_depth_anything"])},
            "same_authoritative_sfm_shared": True,
        },
        "decision": {
            "production_native_complete": True,
            "provider_terminal_outcomes_complete": True,
            "provider_quality_rejection_present": has_rejection,
            "matched_view_diagnostic_authorized_for_g3c": True,
            "scenario_verdict_pending_g3c_manual_review": True,
        },
        "authority": {
            "camera_pose_authority_owner": "pycolmap_sfm",
            "same_authoritative_sfm_shared_by_both_providers": True,
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
    production_path = write_json(root / "i4_3g3b_production_native_snapshot.json", production)
    g1_path = write_json(root / "i4_3g1_snapshot.json", passing_g1_snapshot())
    return {
        "g1_path": g1_path,
        "g1_sha256": sha256_file(g1_path),
        "production_path": production_path,
        "production_sha256": sha256_file(production_path),
        "sfm_path": sfm_path,
        "scenario_id": scenario_id,
        "source_sha256": source_sha,
    }
