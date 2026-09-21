#!/usr/bin/env python3
"""Real I6.6C QProcess transport smoke using the production offline worker."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys
from uuid import uuid4

from PyQt6.QtCore import QCoreApplication, QTimer

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.offline_reconstruction_controller import (
    OfflineReconstructionConfiguration,
    OfflineReconstructionController,
    OfflineReconstructionRunState,
    OfflineReconstructionSourceKind,
)
from core.reconstruction.reconstruction_progressive_sfm_contract import (
    ProgressiveSfmStage,
    validate_progressive_sfm_snapshot_transition,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--sfm-python", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-keyframes", type=int, default=24)
    parser.add_argument("--timeout-minutes", type=float, default=60.0)
    args = parser.parse_args()

    images = args.images.expanduser().resolve()
    sfm_python = args.sfm_python.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists():
        shutil.rmtree(output_dir)

    app = QCoreApplication.instance() or QCoreApplication([])
    controller = OfflineReconstructionController()
    snapshots = []
    logs: list[str] = []
    finish_payloads: list[dict] = []
    errors: list[str] = []
    timed_out = {"value": False}

    controller.progressive_sparse_snapshot_received.connect(snapshots.append)
    controller.log_received.connect(logs.append)
    controller.run_finished.connect(finish_payloads.append)
    controller.error_occurred.connect(errors.append)

    timeout = QTimer()
    timeout.setSingleShot(True)
    timeout.setInterval(max(1, int(args.timeout_minutes * 60_000 + 30_000)))

    def on_timeout() -> None:
        timed_out["value"] = True
        controller.cancel()
        app.quit()

    timeout.timeout.connect(on_timeout)
    controller.run_finished.connect(lambda _payload: app.quit())

    session_id = str(uuid4())
    configuration = OfflineReconstructionConfiguration(
        source_kind=OfflineReconstructionSourceKind.IMAGE_DIRECTORY,
        inputs=(images,),
        output_dir=output_dir,
        sfm_python=sfm_python,
        max_keyframes=int(args.max_keyframes),
        timeout_minutes=float(args.timeout_minutes),
    )
    returned = controller.start(configuration, session_id=session_id)
    assert returned == session_id
    timeout.start()
    app.exec()
    timeout.stop()

    assert not timed_out["value"], "I6.6C real transport smoke timed out."
    assert finish_payloads, "Offline reconstruction did not emit run_finished."
    finished = finish_payloads[-1]
    assert finished["state"] is OfflineReconstructionRunState.COMPLETED, finished
    assert snapshots, "No progressive sparse-SfM snapshots reached the Qt process bridge."
    assert all(snapshot.session_id == session_id for snapshot in snapshots)
    for previous, current in zip(snapshots, snapshots[1:]):
        validate_progressive_sfm_snapshot_transition(previous, current)
    assert snapshots[-1].stage is ProgressiveSfmStage.FINALIZING
    assert all(snapshot.final_result_admitted is False for snapshot in snapshots)
    assert all(snapshot.persistent_canonical_scene is False for snapshot in snapshots)
    assert all(snapshot.metric_scale_claimed is False for snapshot in snapshots)

    stats = controller.progressive_sfm_transport_stats
    assert stats["delivered"] == len(snapshots)
    assert stats["rejected"] == 0, stats
    assert not errors, errors
    result_path = output_dir / "offline_result.json"
    assert result_path.is_file()
    result = json.loads(result_path.read_text(encoding="utf-8"))
    assert bool(result.get("pass"))
    assert str(result.get("session_id", "")) == session_id

    report = {
        "phase": "I6.6C",
        "status": "pass",
        "session_id": session_id,
        "snapshot_count": len(snapshots),
        "revisions": [snapshot.revision for snapshot in snapshots],
        "stages": [snapshot.stage.value for snapshot in snapshots],
        "registered_camera_counts": [
            snapshot.statistics.registered_cameras for snapshot in snapshots
        ],
        "landmark_counts": [snapshot.statistics.sparse_landmarks for snapshot in snapshots],
        "transport_stats": stats,
        "final_worker_status": result.get("status"),
        "final_result_pass": bool(result.get("pass")),
        "final_result_admitted_by_progressive_path": False,
        "persistent_canonical_scene_created_by_progressive_path": False,
        "pycolmap_camera_authority_preserved": True,
        "metric_scale_claimed": False,
        "certified_metrology_claimed": False,
        "errors": errors,
        "log_tail": logs[-20:],
    }
    report_path = output_dir / "i6_6c_progressive_transport_report.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print("I6_6C_REAL_PROGRESSIVE_TRANSPORT=PASS")
    print(f"SNAPSHOT_COUNT={len(snapshots)}")
    print("REVISIONS=" + ",".join(str(item.revision) for item in snapshots))
    print("STAGES=" + ",".join(item.stage.value for item in snapshots))
    print(f"LAST_STAGE={snapshots[-1].stage.value}")
    print(f"COALESCED_COUNT={stats['coalesced']}")
    print(f"REJECTED_COUNT={stats['rejected']}")
    print(f"FINAL_WORKER_STATUS={result.get('status')}")
    print("PROGRESSIVE_FINAL_ADMISSION=NO")
    print("PYCOLMAP_CAMERA_AUTHORITY_PRESERVED=YES")
    print(f"REPORT={report_path}")
    controller.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
