#!/usr/bin/env python3
"""Visually replay immutable I6.7B snapshots into I6.7D's passive GPU viewer.

No provider execution, model inference, source admission, or scene persistence.
Close the window after inspecting the bounded preview and its provisional label.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from PyQt6.QtCore import QTimer
from PyQt6.QtWidgets import QApplication

from core.reconstruction.reconstruction_progressive_dense_transport import (
    ProgressiveDenseTransportBinding, ProgressiveDenseOfferDisposition,
    progressive_dense_transport_line,
)
from scripts.run_i6_7c_progressive_dense_transport_replay import (
    _load_json, _load_snapshots, _sha256_file,
)
from app.widgets.reconstruction_page import ReconstructionPage


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--snapshots-jsonl', required=True, type=Path)
    parser.add_argument('--i6-7b-report', required=True, type=Path)
    args = parser.parse_args()
    snapshots_path = args.snapshots_jsonl.resolve()
    report_path = args.i6_7b_report.resolve()
    report = _load_json(report_path)
    snapshots = _load_snapshots(snapshots_path)
    if report.get('pass') is not True or report.get('status') != 'pass':
        raise RuntimeError('Frozen source report is not a passing qualification.')
    if report.get('artifacts', {}).get('snapshots_jsonl_sha256') != _sha256_file(snapshots_path):
        raise RuntimeError('I6.7B report/snapshot SHA256 mismatch.')
    if len(snapshots) != int(report.get('progressive', {}).get('snapshot_count', -1)):
        raise RuntimeError('I6.7B report/snapshot count mismatch.')
    if [x.revision for x in snapshots] != report['progressive']['revisions']:
        raise RuntimeError('I6.7B report/revision sequence mismatch.')
    if not snapshots[-1].geometry:
        raise RuntimeError('Frozen source has no final bounded preview.')

    app = QApplication.instance() or QApplication(sys.argv[:1])
    page = ReconstructionPage()
    page.setWindowTitle('I6.7D • Frozen dense replay • provisional / non-metric')
    page.begin_progressive_dense_preview(ProgressiveDenseTransportBinding.from_snapshot(snapshots[0]))
    page.show()
    cursor = iter(snapshots)
    pump = QTimer(page)
    result = {'pass': False, 'finished': False}

    def submit_next() -> None:
        snapshot = next(cursor, None)
        if snapshot is None:
            pump.stop()
            QTimer.singleShot(100, print_final)
            return
        disposition = page.offer_progressive_dense_transport_line(progressive_dense_transport_line(snapshot))
        if disposition not in (ProgressiveDenseOfferDisposition.ACCEPTED,
                               ProgressiveDenseOfferDisposition.COALESCED):
            raise RuntimeError('Frozen snapshot rejected: ' + page._progressive_dense_buffer.last_rejection_reason)

    def print_final() -> None:
        canvas = page._reconstruction_canvas
        result['finished'] = True
        result['pass'] = bool(
            canvas.progressive_dense_revision == snapshots[-1].revision
            and canvas.progressive_dense_display_count == len(snapshots[-1].geometry.point_samples)
            and page._progressive_dense_geometry_uploads == 1
            and not canvas.dense_rgb_points_loaded
        )
        print('I6_7D_FROZEN_DENSE_VIEWER_REPLAY=' + (
            'PASS' if result['pass'] else 'FAIL'
        ), flush=True)
        print('FINAL_REVISION=' + str(canvas.progressive_dense_revision), flush=True)
        print('DISPLAY_POINTS=' + str(canvas.progressive_dense_display_count), flush=True)
        print('GEOMETRY_GPU_UPLOADS=' + str(page._progressive_dense_geometry_uploads), flush=True)
        print('VIEWER_APPLY_LATEST_MS=' + f'{page._progressive_dense_latest_apply_ms:.3f}', flush=True)
        print('VIEWER_APPLY_PEAK_MS=' + f'{page._progressive_dense_peak_apply_ms:.3f}', flush=True)
        print('STATUS_LABEL=' + page._viewport_state.text(), flush=True)

    pump.timeout.connect(submit_next)
    pump.start(100)  # human-visible replay; not a throughput benchmark
    rc = app.exec()
    page.shutdown_viewport()
    return 0 if rc == 0 and result['finished'] and result['pass'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
