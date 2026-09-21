#!/usr/bin/env python3
"""Real I6.7B COLMAP-MVS progressive dense-emission qualification.

Consumes one already-admitted/frozen OfflineSfmResult JSON and executes the
existing I4.3B MVS adapter through the I6.7B generic emission wrapper.  This
script is validation-only: it does not select a provider through Auto, admit a
final dense scene, persist canonical dense geometry, or touch the application
controller/UI.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.reconstruction.dense_provider_architecture import (  # noqa: E402
    DenseProviderImplementation,
    DenseProviderRunStatus,
    DenseProviderSelectionDecision,
    DenseProviderSelectionMode,
    DenseProviderSelectionRequest,
)
from core.reconstruction.errors import ReconstructionValidationError  # noqa: E402
from core.reconstruction.mvs_provider import (  # noqa: E402
    I43B_PROVIDER_ID,
    MvsProductionProviderAdapter,
    MvsProviderRunContext,
)
from core.reconstruction.reconstruction_progressive_dense_contract import (  # noqa: E402
    ProgressiveDenseSnapshot,
    validate_progressive_dense_snapshot_transition,
)
from core.reconstruction.reconstruction_progressive_dense_emission import (  # noqa: E402
    I67B_MAX_PREVIEW_POINTS,
    canonical_offline_sfm_result_sha256,
    execute_dense_provider_with_progressive_emission,
)
from core.reconstruction.sfm_result_serialization import offline_sfm_result_from_dict  # noqa: E402


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ReconstructionValidationError(f"JSON payload must be an object: {path}")
    return payload


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sfm_payload(root: dict[str, Any]) -> dict[str, Any]:
    nested = root.get("sfm_result")
    if isinstance(nested, dict):
        return nested
    return root


class _JsonlSnapshotSink:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.snapshots: list[ProgressiveDenseSnapshot] = []
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            raise ReconstructionValidationError(f"Refusing to overwrite progressive snapshot evidence: {path}")

    def publish(self, snapshot: ProgressiveDenseSnapshot) -> None:
        if not isinstance(snapshot, ProgressiveDenseSnapshot):
            raise TypeError("snapshot must be ProgressiveDenseSnapshot")
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(snapshot.as_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n")
        self.snapshots.append(snapshot)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sfm-result-json", type=Path, required=True)
    parser.add_argument("--reconstruction-frame-id", required=True)
    parser.add_argument("--sfm-python", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--timeout-minutes", type=float, default=120.0)
    args = parser.parse_args()

    sfm_json = args.sfm_result_json.expanduser().resolve()
    sfm_python = args.sfm_python.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    frame_id = str(args.reconstruction_frame_id).strip()
    if not sfm_json.is_file():
        raise ReconstructionValidationError(f"Offline SfM JSON is missing: {sfm_json}")
    if not sfm_python.is_file():
        raise ReconstructionValidationError(f"PyCOLMAP Python is missing: {sfm_python}")
    if not frame_id:
        raise ReconstructionValidationError("--reconstruction-frame-id must be non-empty")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ReconstructionValidationError(f"Output directory must be empty or absent: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    source_payload = _sfm_payload(_read_json(sfm_json))
    sfm = offline_sfm_result_from_dict(source_payload)
    registered = int(sfm.statistics.registered_keyframes)
    if registered < 3:
        raise ReconstructionValidationError("I6.7B real smoke requires at least three registered sparse views.")

    request = DenseProviderSelectionRequest(
        mode=DenseProviderSelectionMode.EXPLICIT_PROVIDER,
        available_view_count=registered,
        authoritative_sfm_available=True,
        authoritative_sfm_frame_id=frame_id,
        requested_provider_id=I43B_PROVIDER_ID,
        allow_fallback=False,
    )
    decision = DenseProviderSelectionDecision(
        request_mode=DenseProviderSelectionMode.EXPLICIT_PROVIDER,
        requested_provider_id=I43B_PROVIDER_ID,
        selected_provider_id=I43B_PROVIDER_ID,
        selected_implementation=DenseProviderImplementation.COLMAP_MVS,
        considered_provider_ids=(I43B_PROVIDER_ID,),
        automatic=False,
        engineer_override_applied=False,
        fallback_used=False,
        reason_codes=("explicit_i6_7b_real_qualification",),
    )

    snapshot_path = output_dir / "progressive_dense_snapshots.jsonl"
    sink = _JsonlSnapshotSink(snapshot_path)
    provider_output = output_dir / "provider_colmap_mvs"
    context = MvsProviderRunContext(
        sfm_result=sfm,
        reconstruction_frame_id=frame_id,
        output_dir=provider_output,
        pycolmap_python=sfm_python,
        worker_script=ROOT / "scripts" / "reconstruction_pycolmap_dense_provider_mvs_worker.py",
        worker_timeout_s=max(60.0, float(args.timeout_minutes) * 60.0),
    )

    started = time.monotonic()
    envelope = execute_dense_provider_with_progressive_emission(
        adapter_factory=lambda progress_sink: MvsProductionProviderAdapter(
            context,
            progress_sink=progress_sink,
        ),
        request=request,
        decision=decision,
        source_sfm_result=sfm,
        reconstruction_frame_id=frame_id,
        snapshot_sink=sink,
    )
    wall_time_s = time.monotonic() - started

    snapshots = tuple(sink.snapshots)
    if not snapshots:
        raise ReconstructionValidationError("I6.7B real smoke emitted no progressive dense snapshots.")
    for previous, current in zip(snapshots, snapshots[1:]):
        validate_progressive_dense_snapshot_transition(previous, current)

    if envelope.result.status not in {DenseProviderRunStatus.COMPLETED, DenseProviderRunStatus.DEGRADED}:
        failure_reason = envelope.result.failure_reason or "unspecified provider failure"
        raise ReconstructionValidationError(
            "I6.7B real MVS execution did not complete: "
            f"status={envelope.result.status.value} failure_reason={failure_reason}"
        )

    geometry = tuple(snapshot for snapshot in snapshots if snapshot.geometry is not None)
    if not geometry:
        raise ReconstructionValidationError("I6.7B real smoke emitted no actual dense geometry preview.")
    final_preview = geometry[-1]
    if final_preview.statistics.current_dense_points <= 0:
        raise ReconstructionValidationError("I6.7B real geometry preview contains no dense points.")
    if final_preview.geometry is None or len(final_preview.geometry.point_samples) > I67B_MAX_PREVIEW_POINTS:
        raise ReconstructionValidationError("I6.7B real geometry preview exceeded its visualization-only bound.")
    if envelope.result.final_preview_revision != final_preview.revision:
        raise ReconstructionValidationError("I6.7B execution result/final preview revision drifted.")
    if envelope.report.emission_failure_count != 0:
        raise ReconstructionValidationError(
            f"I6.7B real progressive emission recorded failures: {envelope.report.failure_records}"
        )

    for snapshot in snapshots:
        if snapshot.final_dense_result_admitted:
            raise ReconstructionValidationError("Progressive path incorrectly admitted a final dense result.")
        if snapshot.persistent_dense_scene or snapshot.canonical_geometry_authoritative:
            raise ReconstructionValidationError("Progressive path incorrectly promoted provisional geometry.")
        if snapshot.metric_scale_claimed or snapshot.measurement_authoritative or snapshot.certified_metrology:
            raise ReconstructionValidationError("Progressive path incorrectly claimed metric/metrology authority.")
        if snapshot.surface_mesh_generated or snapshot.provider_promotion_performed:
            raise ReconstructionValidationError("Progressive path crossed a future-phase/provider-promotion boundary.")

    fused = Path(envelope.result.canonical_geometry_ref).expanduser().resolve()
    if not fused.is_file():
        raise ReconstructionValidationError(f"Real MVS canonical geometry is missing: {fused}")

    report = {
        "schema_version": 1,
        "phase": "I6.7B_DENSE_PROVIDER_EXECUTION_AND_PROGRESSIVE_EMISSION",
        "status": "pass",
        "pass": True,
        "source_sfm": {
            "path": str(sfm_json),
            "sha256": _sha256_file(sfm_json),
            "session_id": sfm.session_id,
            "registered_views": registered,
            "source_sparse_result_sha256": canonical_offline_sfm_result_sha256(sfm),
            "reconstruction_frame_id": frame_id,
        },
        "provider": {
            "provider_id": I43B_PROVIDER_ID,
            "implementation": DenseProviderImplementation.COLMAP_MVS.value,
            "selection_mode": DenseProviderSelectionMode.EXPLICIT_PROVIDER.value,
            "auto_used": False,
            "wall_time_s": wall_time_s,
            "result": envelope.result.as_dict(),
        },
        "progressive": {
            "snapshot_count": len(snapshots),
            "geometry_snapshot_count": len(geometry),
            "revisions": [item.revision for item in snapshots],
            "stages": [item.stage.value for item in snapshots],
            "final_preview_revision": final_preview.revision,
            "current_dense_points": final_preview.statistics.current_dense_points,
            "preview_point_count": len(final_preview.geometry.point_samples),
            "max_preview_points": I67B_MAX_PREVIEW_POINTS,
            "emission_report": {
                "emitted_snapshot_count": envelope.report.emitted_snapshot_count,
                "emitted_geometry_snapshot_count": envelope.report.emitted_geometry_snapshot_count,
                "dropped_status_snapshot_count": envelope.report.dropped_status_snapshot_count,
                "emission_failure_count": envelope.report.emission_failure_count,
                "failure_records": list(envelope.report.failure_records),
            },
        },
        "artifacts": {
            "snapshots_jsonl": str(snapshot_path),
            "snapshots_jsonl_sha256": _sha256_file(snapshot_path),
            "canonical_geometry": str(fused),
            "canonical_geometry_sha256": _sha256_file(fused),
        },
        "authority": {
            "final_dense_result_admitted_by_progressive_path": False,
            "persistent_dense_scene_created": False,
            "provider_selection_modified": False,
            "production_auto_activated": False,
            "pycolmap_camera_authority_preserved": True,
            "metric_scale_claimed": False,
            "measurement_authoritative": False,
            "certified_metrology_claimed": False,
            "surface_mesh_generated": False,
        },
    }
    report_path = output_dir / "i6_7b_progressive_dense_emission_report.json"
    _write_json(report_path, report)

    print("I6_7B_REAL_PROGRESSIVE_DENSE_EMISSION=PASS")
    print(f"PROVIDER={I43B_PROVIDER_ID}")
    print(f"REGISTERED_VIEWS={registered}")
    print(f"SNAPSHOT_COUNT={len(snapshots)}")
    print(f"GEOMETRY_SNAPSHOT_COUNT={len(geometry)}")
    print("REVISIONS=" + ",".join(str(item.revision) for item in snapshots))
    print("STAGES=" + ",".join(item.stage.value for item in snapshots))
    print(f"DENSE_POINT_COUNT={final_preview.statistics.current_dense_points}")
    print(f"PREVIEW_POINT_COUNT={len(final_preview.geometry.point_samples)}")
    print(f"MAX_PREVIEW_POINTS={I67B_MAX_PREVIEW_POINTS}")
    print(f"FINAL_PREVIEW_REVISION={final_preview.revision}")
    print(f"FINAL_PROVIDER_STATUS={envelope.result.status.value}")
    print("PROGRESSIVE_FINAL_ADMISSION=NO")
    print("PERSISTENT_DENSE_SCENE_CREATED=NO")
    print("PRODUCTION_AUTO_ACTIVATED=NO")
    print("PYCOLMAP_CAMERA_AUTHORITY_PRESERVED=YES")
    print("METRIC_SCALE_CLAIMED=NO")
    print("SURFACE_MESH_GENERATED=NO")
    print(f"REPORT={report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
