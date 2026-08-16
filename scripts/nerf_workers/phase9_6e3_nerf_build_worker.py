#!/usr/bin/env python3
"""Direct external Nerfstudio build/train/persist worker for Phase 9.6E3."""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.reconstruction import (  # noqa: E402
    NerfSceneArtifact,
    persist_nerf_representation,
    read_persistent_reconstruction_scene,
)
from scripts.nerf_workers.nerfstudio_worker import _run as run_qualified_nerfstudio_worker  # noqa: E402

PROTOCOL = "eia.digital-twin.nerf-build.v1"
PHASE = "9.6E3"


def _emit(job_id: str, event: str, *, scene_id: str, stage: str, message: str, progress: float | None = None, result_path: str = "") -> None:
    payload: dict[str, object] = {
        "protocol": PROTOCOL,
        "job_id": job_id,
        "scene_id": scene_id,
        "event": event,
        "stage": stage,
        "message": message,
        "canonical_geometry_authoritative": False,
        "metric_scale_authoritative": False,
        "measurement_authoritative": False,
        "qc_authoritative": False,
        "certified_metrology": False,
    }
    if progress is not None:
        payload["progress_fraction"] = float(progress)
    if result_path:
        payload["result_path"] = result_path
    print(json.dumps(payload, sort_keys=True), flush=True)


def _hash(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_object(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"Expected JSON object: {path}")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", required=True, type=Path)
    args = parser.parse_args()
    request_path = args.request.expanduser().resolve()
    request = _load_object(request_path)
    job_id = str(request.get("job_id", "")).strip()
    scene_id = str(request.get("scene_id", "")).strip()
    root = request_path.parent.resolve()
    if request.get("phase") != PHASE or request.get("protocol") != PROTOCOL or not job_id or not scene_id:
        raise SystemExit("Invalid Phase 9.6E3 build request identity.")

    try:
        _emit(job_id, "started", scene_id=scene_id, stage="preparing", message="Preparing exact canonical NeRF training evidence.", progress=0.05)
        worker_request = request.get("nerf_worker_request")
        if not isinstance(worker_request, dict):
            raise RuntimeError("NeRF build request is missing nerf_worker_request.")
        inner_request = root / "nerfstudio_worker_request.json"
        inner_result = root / "nerfstudio_worker_result.json"
        inner_request.write_text(json.dumps(worker_request, indent=2, sort_keys=True) + "\n", encoding="utf-8")

        _emit(job_id, "progress", scene_id=scene_id, stage="training", message=f"Training Nerfacto appearance ({int(request['training_steps'])} bounded steps).", progress=0.25)
        run_qualified_nerfstudio_worker(inner_request, inner_result)
        artifact = NerfSceneArtifact.from_payload(_load_object(inner_result))
        artifact_root = Path(artifact.artifact_dir).resolve()
        preview = artifact_root / "preview.png"
        runtime_evidence = artifact_root / "e2_runtime_evidence.json"
        config = Path(artifact.config_path).resolve()
        checkpoint = Path(artifact.checkpoint_path).resolve()
        for path, label in ((preview, "preview"), (runtime_evidence, "runtime evidence"), (config, "config"), (checkpoint, "checkpoint")):
            if not path.is_file() or path.stat().st_size <= 0:
                raise RuntimeError(f"NeRF {label} is missing after training: {path}")

        _emit(job_id, "progress", scene_id=scene_id, stage="persisting", message="Persisting trained NeRF appearance as an immutable scene revision.", progress=0.88)
        optimization_manifest = root / "phase9_6e3_optimization_manifest.json"
        optimization_manifest.write_text(
            json.dumps(
                {
                    "phase": "9.5C",
                    "phase9_6e3_orchestration": True,
                    "request_id": artifact.request_id,
                    "dataset_fingerprint_sha256": artifact.dataset_fingerprint_sha256,
                    "render_capable": artifact.render_capable,
                    "device": artifact.device,
                    "config_sha256": _hash(config),
                    "checkpoint_sha256": _hash(checkpoint),
                    "preview_sha256": _hash(preview),
                    "runtime_evidence_sha256": _hash(runtime_evidence),
                    "canonical_geometry_authoritative": False,
                    "metric_scale_authoritative": False,
                    "measurement_authoritative": False,
                    "qc_authoritative": False,
                    "certified_metrology": False,
                },
                indent=2,
                sort_keys=True,
            ) + "\n",
            encoding="utf-8",
        )
        base_scene = read_persistent_reconstruction_scene(Path(str(request["source_scene_manifest"])).expanduser().resolve())
        if base_scene.scene_id != scene_id:
            raise RuntimeError("NeRF build source scene identity changed after preparation.")
        persisted = persist_nerf_representation(
            base_scene=base_scene,
            nerf_artifact=artifact,
            preview_path=preview,
            runtime_evidence_path=runtime_evidence,
            source_optimization_manifest_path=optimization_manifest,
            output_dir=root / "persistent-scene",
            expected_dataset_fingerprint_sha256=str(request["dataset_fingerprint_sha256"]),
            replace_existing=True,
        )
        summary = {
            "schema_version": 1,
            "phase": PHASE,
            "pass": True,
            "job_id": job_id,
            "source_scene_id": scene_id,
            "persistent_scene_id": persisted.scene.scene_id,
            "persistent_scene_manifest": persisted.scene_manifest_path,
            "rebuild": bool(request.get("rebuild", False)),
            "preset_key": str(request.get("preset_key", "")),
            "preset_label": str(request.get("preset_label", "")),
            "training_steps": artifact.training_steps,
            "input_frame_count": artifact.input_frame_count,
            "training_duration_s": artifact.training_duration_s,
            "preview_path": str(Path(persisted.artifact.artifact_dir) / "preview.png"),
            "checkpoint_path": persisted.artifact.checkpoint_path,
            "dataset_fingerprint_sha256": artifact.dataset_fingerprint_sha256,
            "nerf_scene_role": artifact.scene_role.value,
            "canonical_geometry_authoritative": False,
            "metric_scale_authoritative": False,
            "measurement_authoritative": False,
            "qc_authoritative": False,
            "certified_metrology": False,
            "quality_certified": False,
        }
        summary_path = root / "phase9_6e3_nerf_build_summary.json"
        summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        _emit(job_id, "ready", scene_id=scene_id, stage="completed", message=f"NeRF appearance trained and persisted from {artifact.input_frame_count} canonical views.", progress=1.0, result_path=str(summary_path))
        return 0
    except Exception as exc:
        _emit(job_id, "status", scene_id=scene_id, stage="failed", message=str(exc), progress=None)
        print(f"Phase 9.6E3 NeRF build error: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
