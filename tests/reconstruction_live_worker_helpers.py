from __future__ import annotations

from pathlib import Path
import textwrap


def make_fake_stella_worker_layout(tmp_path: Path) -> tuple[Path, Path, Path]:
    prefix = tmp_path / "stella-prefix"
    (prefix / "lib").mkdir(parents=True)
    (prefix / "lib" / "libstella_vslam.so").write_bytes(b"fake")
    vocab = tmp_path / "orb_vocab.fbow"
    vocab.write_bytes(b"fake-vocabulary")
    worker = tmp_path / "fake_live_worker.py"
    worker.write_text(
        textwrap.dedent(
            r'''#!/usr/bin/env python3
import argparse
import json
import os
import socket
import time

p = argparse.ArgumentParser()
p.add_argument('--socket', required=True)
p.add_argument('--session-id', required=True)
p.add_argument('--vocab', required=True)
p.add_argument('--config', required=True)
p.add_argument('--backend-id', required=True)
a = p.parse_args()

try:
    os.unlink(a.socket)
except FileNotFoundError:
    pass
srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
srv.bind(a.socket)
os.chmod(a.socket, 0o600)
srv.listen(1)
conn, _ = srv.accept()
worker_seq = 0
buf = b''
request = None
mode = os.environ.get('EIA_FAKE_STELLA_MODE', 'normal')

def send(kind, payload):
    global worker_seq
    raw = json.dumps({
        'protocol_version': 1,
        'message_type': kind,
        'session_id': a.session_id,
        'sequence_number': worker_seq,
        'payload': payload,
    }, sort_keys=True, separators=(',', ':')).encode() + b'\n'
    worker_seq += 1
    conn.sendall(raw)

def recv():
    global buf
    while b'\n' not in buf:
        data = conn.recv(65536)
        if not data:
            raise SystemExit(0)
        buf += data
    line, buf = buf.split(b'\n', 1)
    return json.loads(line)

send('hello', {
    'backend_id': a.backend_id,
    'worker_version': 'fake-v1',
    'stella_version': 'fake',
    'stella_commit': 'fake',
    'pose_input_convention': 'T_camera_from_reconstruction',
    'pose_output_convention': 'T_reconstruction_from_camera',
    'pixel_transport': 'posix_shared_memory',
    'control_transport': 'af_unix_json_lines',
})
send('ready', {'protocol_version': 1, 'metric_scale': False})

while True:
    message = recv()
    kind = message['message_type']
    payload = message['payload']
    if kind == 'start':
        request = payload['request']
        send('started', {
            'loop_detector_enabled': False,
            'phase8_global_optimization_feature_enabled': False,
            'internal_global_optimization_thread_may_exist': True,
            'mapping_enabled': request['enable_mapping'],
            'scale_status': 'arbitrary',
            'unit_to_metre': None,
        })
    elif kind == 'frame_available':
        if mode == 'crash_frame':
            os._exit(17)
        if mode in {'hang_frame', 'cancel_frame'}:
            time.sleep(20)
            continue
        frame = payload['frame']
        if mode == 'stale_frame':
            send('error', {
                'code': 'stale_frame',
                'message': 'fake overwritten generation',
                'retryable': True,
                'frame_id': frame['frame_id'],
                'sequence_index': frame['sequence_index'],
            })
            continue
        pose = {
            'pose_id': 'fake-pose-' + str(frame['sequence_index']),
            'camera_id': frame['camera_id'],
            'frame_id': frame['frame_id'],
            'timestamp_ns': frame['capture_timestamp_ns'],
            'camera_frame_id': request['camera']['camera_frame']['frame_id'],
            'reconstruction_frame_id': 'arbitrary_gauge_live_reconstruction',
            'pose_convention': 'T_reconstruction_from_camera',
            'T_reconstruction_from_camera': [
                [1.0, 0.0, 0.0, 0.1],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            'quaternion_order': 'wxyz',
            'gauge': {
                'scale_status': 'arbitrary',
                'alignment_status': 'arbitrary',
                'unit_to_metre': None,
                'scale_source': '',
                'relative_scale_uncertainty': None,
            },
        }
        send('observation', {'observation': {
            'session_id': a.session_id,
            'frame_id': frame['frame_id'],
            'sequence_index': frame['sequence_index'],
            'tracking_state': 'tracking',
            'quality': {
                'tracked_features': 0,
                'inlier_features': 0,
                'inlier_ratio': None,
                'tracking_confidence': None,
                'processing_latency_ns': 1000000,
                'reason': 'fake_pose_valid',
            },
            'pose': pose,
            'is_keyframe': frame['sequence_index'] == 0,
            'keyframe_id': 'fake-keyframe-0' if frame['sequence_index'] == 0 else '',
            'map_snapshot': {
                'revision': 1,
                'keyframe_count': 1,
                'landmark_count': 85,
            },
        }})
    elif kind == 'stop':
        send('stopped', {'clean': True})
    elif kind == 'shutdown':
        break

conn.close()
srv.close()
try:
    os.unlink(a.socket)
except FileNotFoundError:
    pass
'''
        ),
        encoding="utf-8",
    )
    worker.chmod(0o755)
    return prefix, worker, vocab
