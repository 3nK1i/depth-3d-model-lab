#!/usr/bin/env python3
"""Replay frozen I6.7B progressive-dense evidence through I6.7C transport.

This is a no-provider-rerun qualification.  It consumes the already-frozen
I6.7B JSONL snapshots and proves the dedicated line codec, run binding,
latest-only coalescing, and transport-memory bounds on real progressive data.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.reconstruction.errors import ReconstructionValidationError
from core.reconstruction.reconstruction_progressive_dense_contract import (
    ProgressiveDenseSnapshot,
)
from core.reconstruction.reconstruction_progressive_dense_transport import (
    I67C_MAX_RETAINED_PAYLOAD_BYTES,
    I67C_MAX_TRANSPORT_SNAPSHOT_BYTES,
    ProgressiveDenseOfferDisposition,
    ProgressiveDenseSnapshotBuffer,
    ProgressiveDenseTransportBinding,
    ProgressiveDenseTransportLineSink,
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ReconstructionValidationError(f"Expected JSON object: {path}")
    return payload


def _load_snapshots(path: Path) -> tuple[ProgressiveDenseSnapshot, ...]:
    snapshots: list[ProgressiveDenseSnapshot] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, start=1):
            text = raw.strip()
            if not text:
                continue
            try:
                payload = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ReconstructionValidationError(
                    f"Invalid snapshot JSONL at line {line_number}: {exc}"
                ) from exc
            if not isinstance(payload, dict):
                raise ReconstructionValidationError(
                    f"Snapshot JSONL line {line_number} is not an object."
                )
            snapshots.append(ProgressiveDenseSnapshot.from_payload(payload))
    if not snapshots:
        raise ReconstructionValidationError("I6.7C replay requires at least one snapshot.")
    return tuple(snapshots)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshots-jsonl", required=True, type=Path)
    parser.add_argument("--i6-7b-report", required=True, type=Path)
    args = parser.parse_args()

    snapshots_path = args.snapshots_jsonl.expanduser().resolve()
    report_path = args.i6_7b_report.expanduser().resolve()
    if not snapshots_path.is_file():
        raise ReconstructionValidationError(f"Missing I6.7B snapshots JSONL: {snapshots_path}")
    if not report_path.is_file():
        raise ReconstructionValidationError(f"Missing I6.7B report: {report_path}")

    report = _load_json(report_path)
    snapshots = _load_snapshots(snapshots_path)

    if report.get("pass") is not True or report.get("status") != "pass":
        raise ReconstructionValidationError("I6.7B source report is not a passing frozen qualification.")
    artifacts = report.get("artifacts", {})
    if not isinstance(artifacts, dict):
        raise ReconstructionValidationError("I6.7B artifacts block is invalid.")
    expected_snapshots_sha = str(artifacts.get("snapshots_jsonl_sha256", "")).strip().lower()
    actual_snapshots_sha = _sha256_file(snapshots_path)
    if expected_snapshots_sha != actual_snapshots_sha:
        raise ReconstructionValidationError("I6.7B snapshots JSONL SHA256 does not match its report.")

    progressive = report.get("progressive", {})
    if not isinstance(progressive, dict):
        raise ReconstructionValidationError("I6.7B progressive report block is invalid.")
    expected_count = int(progressive.get("snapshot_count", -1))
    if expected_count != len(snapshots):
        raise ReconstructionValidationError("I6.7B report/snapshot count mismatch.")
    expected_revisions = tuple(int(item) for item in progressive.get("revisions", []))
    actual_revisions = tuple(item.revision for item in snapshots)
    if expected_revisions != actual_revisions:
        raise ReconstructionValidationError("I6.7B report/revision sequence mismatch.")

    buffer = ProgressiveDenseSnapshotBuffer()
    buffer.reset(ProgressiveDenseTransportBinding.from_snapshot(snapshots[0]))

    dispositions: list[ProgressiveDenseOfferDisposition] = []
    sink = ProgressiveDenseTransportLineSink(
        lambda line: dispositions.append(buffer.offer_line(line))
    )

    # Deliver the first snapshot so the replay exercises the strict
    # last-delivered -> coalesced-pending transition path and the two-snapshot
    # retained-memory ceiling.  Then intentionally apply maximum backpressure:
    # all intermediate updates coalesce into one latest pending snapshot.
    sink.publish(snapshots[0])
    first_delivered = buffer.take_pending()
    if first_delivered is None or first_delivered.revision != snapshots[0].revision:
        raise ReconstructionValidationError("I6.7C replay could not deliver the first snapshot.")

    for snapshot in snapshots[1:]:
        sink.publish(snapshot)

    final_delivered = buffer.take_pending()
    if final_delivered is None:
        raise ReconstructionValidationError("I6.7C replay produced no final delivered snapshot.")
    expected_final = snapshots[-1]
    if final_delivered.snapshot_sha256 != expected_final.snapshot_sha256:
        raise ReconstructionValidationError("I6.7C latest-only buffer did not preserve the final snapshot.")
    if final_delivered.geometry is None:
        raise ReconstructionValidationError("I6.7C final real replay lost the bounded geometry preview.")

    stats = buffer.stats
    if stats.peak_pending_payload_bytes > I67C_MAX_TRANSPORT_SNAPSHOT_BYTES:
        raise ReconstructionValidationError("I6.7C pending payload exceeded its hard byte bound.")
    if stats.peak_retained_payload_bytes > I67C_MAX_RETAINED_PAYLOAD_BYTES:
        raise ReconstructionValidationError("I6.7C retained payload exceeded its hard memory bound.")
    if stats.rejected != 0:
        raise ReconstructionValidationError("I6.7C replay rejected frozen valid I6.7B snapshots.")
    if stats.delivered != 2:
        raise ReconstructionValidationError("I6.7C replay expected exactly two delivered snapshots.")
    expected_coalesced = max(0, len(snapshots) - 2)
    if stats.coalesced != expected_coalesced:
        raise ReconstructionValidationError("I6.7C real replay coalescing count drifted.")

    authority = report.get("authority", {})
    if not isinstance(authority, dict):
        raise ReconstructionValidationError("I6.7B authority block is invalid.")
    forbidden_true = (
        "final_dense_result_admitted_by_progressive_path",
        "persistent_dense_scene_created",
        "production_auto_activated",
        "metric_scale_claimed",
        "measurement_authoritative",
        "certified_metrology_claimed",
        "surface_mesh_generated",
    )
    if any(authority.get(name) is True for name in forbidden_true):
        raise ReconstructionValidationError("I6.7C replay source crossed a frozen authority boundary.")
    if authority.get("pycolmap_camera_authority_preserved") is not True:
        raise ReconstructionValidationError("I6.7C replay source does not preserve PyCOLMAP authority.")

    print("I6_7C_REAL_TRANSPORT_REPLAY=PASS")
    print(f"SOURCE_REPORT={report_path}")
    print(f"SOURCE_SNAPSHOTS={snapshots_path}")
    print(f"SOURCE_SNAPSHOTS_SHA256={actual_snapshots_sha}")
    print(f"INPUT_SNAPSHOT_COUNT={len(snapshots)}")
    print(f"FIRST_REVISION={snapshots[0].revision}")
    print(f"FINAL_REVISION={final_delivered.revision}")
    print(f"FINAL_STAGE={final_delivered.stage.value}")
    print(f"FINAL_HAS_GEOMETRY={'YES' if final_delivered.geometry is not None else 'NO'}")
    print(f"FINAL_PREVIEW_POINT_COUNT={len(final_delivered.geometry.point_samples)}")
    print(f"FINAL_DENSE_POINT_COUNT={final_delivered.statistics.current_dense_points}")
    print(f"TRANSPORT_ACCEPTED={stats.accepted}")
    print(f"TRANSPORT_COALESCED={stats.coalesced}")
    print(f"TRANSPORT_REJECTED={stats.rejected}")
    print(f"TRANSPORT_DELIVERED={stats.delivered}")
    print(f"PEAK_PENDING_PAYLOAD_BYTES={stats.peak_pending_payload_bytes}")
    print(f"MAX_PENDING_PAYLOAD_BYTES={I67C_MAX_TRANSPORT_SNAPSHOT_BYTES}")
    print(f"PEAK_RETAINED_PAYLOAD_BYTES={stats.peak_retained_payload_bytes}")
    print(f"MAX_RETAINED_PAYLOAD_BYTES={I67C_MAX_RETAINED_PAYLOAD_BYTES}")
    print("LATEST_ONLY_PENDING_BUFFER=YES")
    print("BOUNDED_PENDING_SNAPSHOT_COUNT=1")
    print("FINAL_GEOMETRY_PRESERVED=YES")
    print("PROGRESSIVE_FINAL_ADMISSION=NO")
    print("PERSISTENT_DENSE_SCENE_CREATED=NO")
    print("PRODUCTION_AUTO_ACTIVATED=NO")
    print("PYCOLMAP_CAMERA_AUTHORITY_PRESERVED=YES")
    print("METRIC_SCALE_CLAIMED=NO")
    print("SURFACE_MESH_GENERATED=NO")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
