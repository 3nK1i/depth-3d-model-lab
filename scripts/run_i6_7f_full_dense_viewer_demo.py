#!/usr/bin/env python3
"""I6.7F real desktop GPU replay from ONE already-admitted sparse/dense run.

Never reruns SfM/MVS, never persists a scene or promotes provisional geometry.
Close the window after orbit/pan/zoom manual validation; compare console stats.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys
from time import monotonic

from PyQt6.QtCore import QThread, pyqtSignal
from PyQt6.QtWidgets import QApplication, QHBoxLayout, QLabel, QMainWindow, QPushButton, QVBoxLayout, QWidget

from app.reconstruction_full_dense_viewer import prepare_full_dense_viewer_frame
from app.widgets.viewport.vispy_canvas import ReconstructionCanvas


class Loader(QThread):
    ready = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self, admission: dict):
        super().__init__()
        self.admission = admission

    def run(self):
        try:
            frame = prepare_full_dense_viewer_frame(
                self.admission, expected_session_id=self.admission['sparse_session_id'])
        except Exception as exc:
            self.failed.emit(str(exc)[:600])
        else:
            self.ready.emit(frame)



def build_demo_admission(*, offline_output_dir: Path, session_id: str,
                         dense_run_id: str, manifest_sha256: str,
                         geometry_sha256: str) -> dict[str, object]:
    """Build the *complete* viewer input from the existing bounded admission.

    This early manifest read only supplies the claimed vertex count; the normal
    independent verifier still checks the source sparse result, manifest,
    complete PLY and point count before preparing any GPU buffers.
    """
    manifest = offline_output_dir / 'dense_runs' / dense_run_id / 'final_dense_admission.json'
    if not manifest.is_file() or manifest.is_symlink() or manifest.stat().st_size > 64 * 1024:
        raise ValueError('missing, symlinked, or oversized final admission manifest')
    raw = manifest.read_bytes()
    if len(raw) > 64 * 1024 or sha256(raw).hexdigest() != manifest_sha256:
        raise ValueError('final admission manifest SHA256 mismatch; preserve source evidence')
    payload = json.loads(raw)
    if not isinstance(payload, dict):
        raise ValueError('final admission manifest is not an object')
    count = payload.get('complete_geometry_vertex_count')
    if type(count) is not int or count <= 0:
        raise ValueError('final admission manifest lacks a valid complete vertex count')
    return dict(
        sparse_session_id=session_id, dense_run_id=dense_run_id,
        manifest_path=str(manifest), manifest_sha256=manifest_sha256,
        complete_geometry_sha256=geometry_sha256,
        complete_geometry_vertex_count=count,
        final_dense_result_admitted=True,
    )

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--offline-output-dir', type=Path, required=True)
    parser.add_argument('--session-id', required=True)
    parser.add_argument('--dense-run-id', required=True)
    parser.add_argument('--manifest-sha256', required=True)
    parser.add_argument('--geometry-sha256', required=True)
    args = parser.parse_args()
    try:
        admission = build_demo_admission(
            offline_output_dir=args.offline_output_dir,
            session_id=args.session_id, dense_run_id=args.dense_run_id,
            manifest_sha256=args.manifest_sha256,
            geometry_sha256=args.geometry_sha256,
        )
    except (OSError, ValueError, TypeError) as exc:
        print('I6_7F_REAL_FULL_DENSE_VIEWER=FAIL: ' + str(exc)[:400], flush=True)
        return 1
    app = QApplication.instance() or QApplication(sys.argv[:1])
    window = QMainWindow()
    window.setWindowTitle('I6.7F • Complete admitted dense cloud • non-metric / non-persistent')
    outer = QWidget()
    layout = QVBoxLayout(outer)
    label = QLabel('Verifying complete dense artifact and preparing full-resolution GPU buffers…')
    layout.addWidget(label)
    canvas = ReconstructionCanvas(window)
    layout.addWidget(canvas.native, stretch=1)
    controls = QHBoxLayout()
    for mode in ('orbit', 'pan', 'zoom'):
        button = QPushButton(mode.title())
        button.clicked.connect(lambda _checked=False, m=mode: canvas.set_interaction_mode(m))
        controls.addWidget(button)
    fit = QPushButton('Fit complete cloud')
    fit.clicked.connect(canvas.fit_admitted_full_dense)
    controls.addWidget(fit)
    layout.addLayout(controls)
    window.setCentralWidget(outer)
    window.resize(1400, 900)
    window.show()
    loaded = []
    result = {'rc': 1}
    worker = Loader(admission)

    def on_ready(frame):
        try:
            t0 = monotonic()
            canvas.show_admitted_full_dense(frame)
            submit_ms = (monotonic() - t0) * 1000
            label.setText(f'Full admitted RGB cloud: {frame.point_count:,} / {frame.point_count:,} points • non-metric / non-persistent')
            print('I6_7F_REAL_FULL_DENSE_VIEWER=PASS', flush=True)
            print(f'FULL_VERIFIED_POINTS={frame.point_count}', flush=True)
            print(f'FULL_DISPLAY_POINTS={canvas.admitted_full_dense_display_count}', flush=True)
            print(f'FULL_MARKER_SUBMISSIONS={canvas.admitted_full_dense_gpu_submissions}', flush=True)
            print(f'CPU_PREPARE_MS={frame.preparation_ms:.3f}', flush=True)
            print(f'QT_MARKER_SUBMISSION_MS={submit_ms:.3f}', flush=True)
            print('RENDER_FPS_MEASURED=NO', flush=True)
            print('PERSISTENT_SCENE_CREATED=NO', flush=True)
            print('METRIC_SCALE_CLAIMED=NO', flush=True)
            result['rc'] = 0
        except Exception as exc:
            on_failed(str(exc))

    def on_failed(detail):
        label.setText('Complete dense viewer unavailable: ' + str(detail)[:400])
        print('I6_7F_REAL_FULL_DENSE_VIEWER=FAIL: ' + str(detail)[:400], flush=True)
        result['rc'] = 1

    worker.ready.connect(on_ready)
    worker.failed.connect(on_failed)
    worker.start()
    try:
        app.exec()
    finally:
        worker.wait()  # never destroy running QThread
        canvas.clear_admitted_full_dense(update_canvas=False)
        canvas.close()
    return result['rc']


if __name__ == '__main__':
    raise SystemExit(main())
