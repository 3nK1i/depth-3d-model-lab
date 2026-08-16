#!/usr/bin/env python3
"""Evaluate the Phase 9.6H final real-scene Digital Twin release gate.

This script verifies already-produced E2/E3/F evidence and explicit manual UI
confirmations.  It never starts a heavyweight backend.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys
from typing import Any

import cv2

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.reconstruction_canonical_cameras import load_persistent_canonical_camera_trajectory  # noqa: E402
from app.reconstruction_release_gate import (  # noqa: E402
    MANUAL_BOOLEAN_FIELDS,
    PHASE96H,
    RELEASE_GATE_SCHEMA,
    qualified_release_limitations,
    validate_phase96h_evidence,
)
from core.reconstruction import read_persistent_reconstruction_scene, verify_persistent_scene_file_references  # noqa: E402


def _json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"Expected JSON object: {path}")
    return payload


def _hash_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _probe_video(path: Path) -> dict[str, Any]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return {"path": str(path), "readable": False, "duration_s": 0.0, "fps": 0.0, "frame_count": 0, "width": 0, "height": 0}
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        duration = (frames / fps) if fps > 0.0 and frames > 0 else 0.0
        return {
            "path": str(path),
            "readable": True,
            "duration_s": duration,
            "fps": fps,
            "frame_count": frames,
            "width": width,
            "height": height,
        }
    finally:
        cap.release()


def _verify_state_export_refs(payload: dict[str, Any]) -> tuple[bool, list[dict[str, Any]]]:
    checks: list[dict[str, Any]] = []
    ok = True
    for row in list(payload.get("referenced_payloads", [])):
        path = Path(str(row.get("path", ""))).expanduser()
        expected_size = int(row.get("size_bytes", -1) or -1)
        expected_sha = str(row.get("sha256", "")).strip().lower()
        exists = path.is_file()
        size_ok = exists and path.stat().st_size == expected_size
        sha_ok = size_ok and _hash_file(path) == expected_sha
        checks.append({"path": str(path), "exists": exists, "size_ok": size_ok, "sha256_ok": sha_ok})
        ok = ok and exists and size_ok and sha_ok
    return ok, checks


def _parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--video", type=Path, required=True)
    p.add_argument("--e2-summary", type=Path, required=True)
    p.add_argument("--e3-summary", type=Path, required=True)
    p.add_argument("--state-export", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, default=Path("/tmp/phase9_6h_release_gate"))
    p.add_argument("--appearance-path", choices=("nerf", "gaussian+nerf"), default="nerf")
    for field in MANUAL_BOOLEAN_FIELDS:
        p.add_argument("--" + field.replace("_", "-"), action="store_true")
    return p.parse_args()


def main() -> int:
    args = _parse()
    video_path = args.video.expanduser().resolve()
    e2_path = args.e2_summary.expanduser().resolve()
    e3_path = args.e3_summary.expanduser().resolve()
    state_path = args.state_export.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)

    try:
        for path, label in ((video_path, "video"), (e2_path, "E2 summary"), (e3_path, "E3 summary"), (state_path, "state export")):
            if not path.is_file():
                raise RuntimeError(f"Missing {label}: {path}")
        video = _probe_video(video_path)
        e2 = _json(e2_path)
        e3 = _json(e3_path)
        state_export = _json(state_path)

        scene_manifest = Path(str(e3.get("persistent_scene_manifest", ""))).expanduser().resolve()
        if not scene_manifest.is_file():
            raise RuntimeError(f"E3 persistent scene manifest is missing: {scene_manifest}")
        scene = read_persistent_reconstruction_scene(scene_manifest)
        verify_persistent_scene_file_references(scene)
        cameras = load_persistent_canonical_camera_trajectory(scene)
        camera_count = int(cameras.persisted_pose_count) if cameras is not None else 0

        refs_ok, ref_checks = _verify_state_export_refs(state_export)
        manual = {field: bool(getattr(args, field)) for field in MANUAL_BOOLEAN_FIELDS}
        manual["appearance_path"] = args.appearance_path

        evidence = {
            "schema": RELEASE_GATE_SCHEMA,
            "phase": PHASE96H,
            "video": video,
            "e2_full_scene": e2,
            "e3_nerf": e3,
            "persistent_scene": {
                "scene_id": scene.scene_id,
                "scene_fingerprint_sha256": scene.scene_fingerprint_sha256,
                "canonical_geometry_authoritative": scene.canonical_geometry_authoritative,
                "certified_metrology": scene.certified_metrology,
                "camera_count": camera_count,
                "references_verified": True,
                "manifest": str(scene_manifest),
            },
            "state_export": state_export,
            "integrity": {
                "persistent_scene_manifest_exists": scene_manifest.is_file(),
                "state_export_references_verified": refs_ok,
                "reference_checks": ref_checks,
            },
            "manual": manual,
            "limitations": qualified_release_limitations(),
        }
        result = validate_phase96h_evidence(evidence)
        summary = {
            **evidence,
            "pass": result.passed,
            "reasons": list(result.reasons),
            "release_candidate_scope": "functional_digital_twin_release_gate",
            "v0_9_0_tag_authorized_by_script": False,
        }
        summary_path = output / "phase9_6h_release_gate_summary.json"
        summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps({"pass": result.passed, "reasons": list(result.reasons), "summary": str(summary_path)}, indent=2))
        return 0 if result.passed else 1
    except Exception as exc:
        print(f"Phase 9.6H release-gate error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
